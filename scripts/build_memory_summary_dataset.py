from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from msc_processed_utils import END_OF_TEXT, MEMORY_START_MARKER, available_speaker_indices, extract_memory_items_from_annotated_dialogue, iter_jsonl, orient_dialogue_turns, parse_session_number, render_dialogue_block, write_jsonl_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage A memory-summary training data from normalized MSC persona summary JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw normalized MSC persona summary JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output processed JSONL.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on written processed examples.",
    )
    return parser.parse_args()


def build_text(previous_dialogue: list[dict[str, str]], memory_items: list[str]) -> tuple[str, str]:
    dialogue_block = render_dialogue_block(previous_dialogue)
    target_text = "\n".join(memory_items) + END_OF_TEXT
    text = (
        "<|task|> memory_summary\n"
        "<|dialog|>\n"
        f"{dialogue_block}\n\n"
        "<|memory|>\n"
        f"{target_text}"
    )
    return text, target_text


def build_record(
    raw_record: dict[str, Any],
    target_speaker_index: int,
    output_index: int,
) -> dict[str, Any] | None:
    raw_metadata = raw_record.get("metadata") or {}
    annotated_dialogue = raw_metadata.get("annotated_dialogue") or []
    if not annotated_dialogue:
        return None

    previous_dialogue = orient_dialogue_turns(annotated_dialogue, target_speaker_index)
    memory_items = extract_memory_items_from_annotated_dialogue(annotated_dialogue, target_speaker_index)
    if not previous_dialogue or not memory_items:
        return None

    split = raw_record.get("split") or "unknown"
    source_session_folder = raw_metadata.get("session_folder")
    source_session_number = parse_session_number(source_session_folder or raw_metadata.get("session_id"))
    text, target_text = build_text(previous_dialogue, memory_items)

    return {
        "id": f"msc_memory_{split}_{output_index:06d}",
        "task": "memory_summary",
        "source": "msc",
        "split": split,
        "input": {
            "previous_dialogue": previous_dialogue,
        },
        "target": {
            "memory": memory_items,
        },
        "text": text,
        "target_start_marker": MEMORY_START_MARKER,
        "target_text": target_text,
        "metadata": {
            "episode_id": raw_record.get("episode_id"),
            "example_id": raw_record.get("example_id"),
            "session_id": raw_metadata.get("session_id"),
            "source_session_folder": source_session_folder,
            "source_session_number": source_session_number,
            "initial_data_id": raw_metadata.get("initial_data_id"),
            "target_speaker_index": target_speaker_index,
            "num_memory_items": len(memory_items),
            "num_dialog_turns": len(previous_dialogue),
            "followup": raw_metadata.get("followup"),
            "newfact": raw_metadata.get("newfact"),
        },
    }


def build_dataset(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written_examples = 0
    skipped_examples = 0

    with args.output.open("w", encoding="utf-8") as output_handle:
        for raw_record in iter_jsonl(args.input):
            raw_metadata = raw_record.get("metadata") or {}
            annotated_dialogue = raw_metadata.get("annotated_dialogue") or []
            speaker_indices = available_speaker_indices(annotated_dialogue)

            for target_speaker_index in speaker_indices:
                if args.max_examples is not None and written_examples >= args.max_examples:
                    print(f"Reached max examples limit ({args.max_examples}).")
                    print_summary(args.output, written_examples, skipped_examples)
                    return 0

                processed_record = build_record(raw_record, target_speaker_index, written_examples + 1)
                if processed_record is None:
                    skipped_examples += 1
                    continue

                write_jsonl_record(output_handle, processed_record)
                written_examples += 1

    print_summary(args.output, written_examples, skipped_examples)
    return 0


def print_summary(output_path: Path, written_examples: int, skipped_examples: int) -> None:
    print(f"Wrote {written_examples} examples to {output_path.as_posix()}")
    print(f"Skipped {skipped_examples} examples")


def main() -> int:
    args = parse_args()
    return build_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())