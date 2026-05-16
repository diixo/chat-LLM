from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from msc_processed_utils import ASSISTANT_START_MARKER, END_OF_TEXT, available_speaker_indices, iter_jsonl, orient_dialogue_turns, parse_memory_items_from_target_text, parse_session_number, render_dialogue_block, write_jsonl_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage B response SFT data from normalized MSC dialogue JSONL and optional gold memory JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw normalized MSC dialogue JSONL.")
    parser.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="Optional processed memory summary JSONL for gold-memory response examples.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output processed JSONL.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on written processed examples.",
    )
    parser.add_argument(
        "--require-memory",
        action="store_true",
        help="Skip response examples when no matching memory summary is found.",
    )
    return parser.parse_args()


def build_memory_index(memory_path: Path | None) -> dict[tuple[str, int, int], dict[str, Any]]:
    if memory_path is None:
        return {}

    memory_index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in iter_jsonl(memory_path):
        metadata = record.get("metadata") or {}
        initial_data_id = metadata.get("initial_data_id")
        source_session_number = parse_session_number(
            metadata.get("source_session_number") or metadata.get("source_session_folder") or metadata.get("session_id")
        )
        target_speaker_index = metadata.get("target_speaker_index")
        if initial_data_id is None or source_session_number is None or target_speaker_index is None:
            continue

        memory_items = (record.get("target") or {}).get("memory")
        if not isinstance(memory_items, list) or not memory_items:
            memory_items = parse_memory_items_from_target_text(record.get("target_text"))

        memory_index[(str(initial_data_id), source_session_number, int(target_speaker_index))] = {
            "id": record.get("id"),
            "memory": memory_items,
            "source_session_number": source_session_number,
            "source_session_folder": metadata.get("source_session_folder"),
        }

    return memory_index


def build_text(memory_items: list[str], current_dialogue: list[dict[str, str]], assistant_response: str) -> tuple[str, str]:
    memory_block = "\n".join(memory_items)
    dialogue_block = render_dialogue_block(current_dialogue)
    target_text = assistant_response + END_OF_TEXT
    text = (
        "<|task|> response\n"
        "<|memory|>\n"
        f"{memory_block}\n\n"
        "<|dialog|>\n"
        f"{dialogue_block}\n\n"
        "<|assistant|>\n"
        f"{target_text}"
    )
    return text, target_text


def build_dataset(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    memory_index = build_memory_index(args.memory)

    written_examples = 0
    skipped_examples = 0
    memory_hits = 0
    memory_misses = 0

    with args.output.open("w", encoding="utf-8") as output_handle:
        for raw_record in iter_jsonl(args.input):
            raw_metadata = raw_record.get("metadata") or {}
            current_dialogue = raw_metadata.get("current_dialogue") or []
            if not current_dialogue:
                skipped_examples += 1
                continue

            split = raw_record.get("split") or "unknown"
            source_session_folder = raw_metadata.get("session_folder")
            source_session_number = parse_session_number(source_session_folder or raw_metadata.get("session_id"))
            initial_data_id = raw_metadata.get("initial_data_id")
            speaker_indices = available_speaker_indices(current_dialogue)

            for target_user_speaker_index in speaker_indices:
                oriented_dialogue = orient_dialogue_turns(current_dialogue, target_user_speaker_index)
                previous_session_number = None
                if source_session_number is not None and source_session_number > 1:
                    previous_session_number = source_session_number - 1

                memory_record = None
                memory_items: list[str] = []
                if args.memory is not None and initial_data_id is not None and previous_session_number is not None:
                    memory_record = memory_index.get(
                        (str(initial_data_id), previous_session_number, target_user_speaker_index)
                    )
                    if memory_record is not None:
                        memory_items = list(memory_record.get("memory") or [])
                        memory_hits += 1
                    else:
                        memory_misses += 1
                        if args.require_memory:
                            continue

                dialogue_history: list[dict[str, str]] = []
                assistant_speaker_index = next(
                    (index for index in speaker_indices if index != target_user_speaker_index),
                    None,
                )

                for target_turn_index, turn in enumerate(oriented_dialogue):
                    role = turn["role"]
                    content = turn["content"]

                    if role == "assistant":
                        if dialogue_history and dialogue_history[-1]["role"] == "user":
                            if args.max_examples is not None and written_examples >= args.max_examples:
                                print(f"Reached max examples limit ({args.max_examples}).")
                                print_summary(args.output, written_examples, skipped_examples, memory_hits, memory_misses)
                                return 0

                            prompt_dialogue = [
                                {"role": prompt_turn["role"], "content": prompt_turn["content"]}
                                for prompt_turn in dialogue_history
                            ]
                            text, target_text = build_text(memory_items, prompt_dialogue, content)
                            processed_record = {
                                "id": f"msc_response_{split}_{written_examples + 1:06d}",
                                "task": "response_sft",
                                "source": "msc",
                                "split": split,
                                "input": {
                                    "memory": memory_items,
                                    "current_dialogue": prompt_dialogue,
                                },
                                "target": {
                                    "assistant_response": content,
                                },
                                "text": text,
                                "target_start_marker": ASSISTANT_START_MARKER,
                                "target_text": target_text,
                                "metadata": {
                                    "episode_id": raw_record.get("episode_id"),
                                    "example_id": raw_record.get("example_id"),
                                    "session_id": raw_metadata.get("session_id"),
                                    "source_session_folder": source_session_folder,
                                    "source_session_number": source_session_number,
                                    "initial_data_id": initial_data_id,
                                    "target_user_speaker_index": target_user_speaker_index,
                                    "assistant_speaker_index": assistant_speaker_index,
                                    "target_turn_index": target_turn_index,
                                    "num_memory_items": len(memory_items),
                                    "num_dialog_turns": len(prompt_dialogue),
                                    "memory_source_id": memory_record.get("id") if memory_record else None,
                                    "memory_source_session_number": previous_session_number,
                                },
                            }
                            write_jsonl_record(output_handle, processed_record)
                            written_examples += 1

                        dialogue_history.append({"role": role, "content": content})
                        continue

                    dialogue_history.append({"role": role, "content": content})

    print_summary(args.output, written_examples, skipped_examples, memory_hits, memory_misses)
    return 0


def print_summary(
    output_path: Path,
    written_examples: int,
    skipped_examples: int,
    memory_hits: int,
    memory_misses: int,
) -> None:
    print(f"Wrote {written_examples} examples to {output_path.as_posix()}")
    print(f"Skipped {skipped_examples} examples")
    print(f"Memory hits: {memory_hits}")
    print(f"Memory misses: {memory_misses}")


def main() -> int:
    args = parse_args()
    return build_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())