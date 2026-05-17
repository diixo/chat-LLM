from __future__ import annotations

from pathlib import Path
from typing import Any

from msc_processed_utils import ASSISTANT_START_MARKER, END_OF_TEXT, dedupe_preserve_order, normalize_whitespace, render_dialogue_block, rewrite_fact_to_user_perspective, write_jsonl_record


CONVAI2_SPLITS: tuple[tuple[str, Path, Path], ...] = (
    (
        "train",
        Path("data/convai2/train_self_original_no_cands.txt"),
        Path("datasets/processed/convai2_response_sft/train.jsonl"),
    ),
    (
        "valid",
        Path("data/convai2/valid_self_original_no_cands.txt"),
        Path("datasets/processed/convai2_response_sft/valid.jsonl"),
    ),
)
MAX_EXAMPLES: int | None = None


def build_initial_data_id(split: str, episode_index: int) -> str:
    if split == "train":
        return f"train:ordered_{episode_index}"
    return f"{split}_{episode_index}"


def load_episodes(input_path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    persona_lines: list[str] = []
    dialogue_pairs: list[tuple[str, str]] = []

    def flush_episode() -> None:
        if not persona_lines and not dialogue_pairs:
            return

        episodes.append(
            {
                "personality": list(persona_lines),
                "dialogue_pairs": list(dialogue_pairs),
            }
        )

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            number_token, separator, content = stripped_line.partition(" ")
            if not separator or not number_token.isdigit():
                raise ValueError(f"Invalid ConvAI2 line format in {input_path.as_posix()}: {stripped_line}")

            if int(number_token) == 1 and (persona_lines or dialogue_pairs):
                flush_episode()
                persona_lines = []
                dialogue_pairs = []

            content = content.strip()
            if content.startswith("your persona: "):
                persona_text = normalize_whitespace(content[len("your persona: ") :])
                if persona_text:
                    persona_lines.append(persona_text)
                continue

            fields = content.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Expected a dialogue pair in {input_path.as_posix()}: {stripped_line}")

            user_text = normalize_whitespace(fields[0])
            assistant_text = normalize_whitespace(fields[1])
            if not user_text or not assistant_text:
                continue

            dialogue_pairs.append((user_text, assistant_text))

    flush_episode()
    return episodes


def build_memory_items(personality: Any) -> list[str]:
    if not isinstance(personality, list):
        return []

    rewritten_items = [
        rewrite_fact_to_user_perspective(item)
        for item in personality
        if normalize_whitespace(item if isinstance(item, str) else None)
    ]
    return dedupe_preserve_order(rewritten_items)


def build_prompt_dialogue(dialogue_pairs: Any, dialogue_pair_index: int) -> list[dict[str, str]]:
    if not isinstance(dialogue_pairs, list) or dialogue_pair_index >= len(dialogue_pairs):
        return []

    prompt_dialogue: list[dict[str, str]] = []
    for pair_index, raw_pair in enumerate(dialogue_pairs[:dialogue_pair_index]):
        if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
            continue

        user_text, assistant_text = raw_pair
        prompt_dialogue.append({"role": "user", "content": user_text})
        prompt_dialogue.append({"role": "assistant", "content": assistant_text})

    current_pair = dialogue_pairs[dialogue_pair_index]
    if not isinstance(current_pair, tuple) or len(current_pair) != 2:
        return []

    current_user_text = normalize_whitespace(current_pair[0])
    if not current_user_text:
        return []

    prompt_dialogue.append({"role": "user", "content": current_user_text})
    return prompt_dialogue


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


def build_record(
    episode: dict[str, Any],
    episode_index: int,
    dialogue_pair_index: int,
    split: str,
    output_index: int,
    input_path: Path,
) -> dict[str, Any] | None:
    memory_items = build_memory_items(episode.get("personality"))
    if not memory_items:
        return None

    dialogue_pairs = episode.get("dialogue_pairs")
    if not isinstance(dialogue_pairs, list) or dialogue_pair_index >= len(dialogue_pairs):
        return None

    prompt_dialogue = build_prompt_dialogue(dialogue_pairs, dialogue_pair_index)
    if not prompt_dialogue:
        return None

    raw_pair = dialogue_pairs[dialogue_pair_index]
    if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
        return None

    assistant_response = normalize_whitespace(raw_pair[1])
    if not assistant_response:
        return None

    text, target_text = build_text(memory_items, prompt_dialogue, assistant_response)
    initial_data_id = build_initial_data_id(split, episode_index)
    episode_id = f"convai2_{initial_data_id}"
    example_id = f"{episode_id}:turn_{dialogue_pair_index:03d}"

    return {
        "id": f"convai2_response_{split}_{output_index:06d}",
        "task": "response_sft",
        "source": "convai2",
        "split": split,
        "input": {
            "memory": memory_items,
            "current_dialogue": prompt_dialogue,
        },
        "target": {
            "assistant_response": assistant_response,
        },
        "text": text,
        "target_start_marker": ASSISTANT_START_MARKER,
        "target_text": target_text,
        "metadata": {
            "episode_id": episode_id,
            "example_id": example_id,
            "initial_data_id": initial_data_id,
            "source_file": input_path.as_posix(),
            "convai2_episode_index": episode_index,
            "dialogue_pair_index": dialogue_pair_index,
            "num_memory_items": len(memory_items),
            "num_dialog_turns": len(prompt_dialogue),
        },
    }


def build_dataset(
    split: str,
    input_path: Path,
    output_path: Path,
    max_examples: int | None = MAX_EXAMPLES,
) -> tuple[int, int]:
    episodes = load_episodes(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_examples = 0
    skipped_examples = 0

    with output_path.open("w", encoding="utf-8") as output_handle:
        for episode_index, episode in enumerate(episodes):
            dialogue_pairs = episode.get("dialogue_pairs")
            if not isinstance(dialogue_pairs, list):
                skipped_examples += 1
                continue

            for dialogue_pair_index, _dialogue_pair in enumerate(dialogue_pairs):
                if max_examples is not None and written_examples >= max_examples:
                    print(f"Reached max examples limit ({max_examples}).")
                    print_summary(output_path, written_examples, skipped_examples)
                    return written_examples, skipped_examples

                processed_record = build_record(
                    episode=episode,
                    episode_index=episode_index,
                    dialogue_pair_index=dialogue_pair_index,
                    split=split,
                    output_index=written_examples + 1,
                    input_path=input_path,
                )
                if processed_record is None:
                    skipped_examples += 1
                    continue

                write_jsonl_record(output_handle, processed_record)
                written_examples += 1

    print_summary(output_path, written_examples, skipped_examples)
    return written_examples, skipped_examples


def print_summary(output_path: Path, written_examples: int, skipped_examples: int) -> None:
    print(f"Wrote {written_examples} examples to {output_path.as_posix()}")
    print(f"Skipped {skipped_examples} examples")


def print_total_summary(total_written: int, total_skipped: int) -> None:
    print(f"Total written examples: {total_written}")
    print(f"Total skipped examples: {total_skipped}")


def main() -> int:
    total_written = 0
    total_skipped = 0

    for split, input_path, output_path in CONVAI2_SPLITS:
        written_examples, skipped_examples = build_dataset(
            split=split,
            input_path=input_path,
            output_path=output_path,
        )
        total_written += written_examples
        total_skipped += skipped_examples

    print_total_summary(total_written, total_skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())