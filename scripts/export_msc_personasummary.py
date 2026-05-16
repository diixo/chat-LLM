from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local MSC persona summary files into the project's raw JSONL schema."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/msc_v0.1/msc/msc_personasummary"),
        help="Root directory that contains session_* folders.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        required=True,
        help="Dataset split to export from every matching session folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file in the project's raw unified format.",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Optional session filter. Accepts values like 1 or session_1. Repeatable.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of written examples across all input files.",
    )
    return parser.parse_args()


def normalize_session_filters(session_filters: Iterable[str]) -> set[str]:
    normalized = set()
    for value in session_filters:
        cleaned = value.strip()
        if not cleaned:
            continue
        if cleaned.startswith("session_"):
            normalized.add(cleaned)
        else:
            normalized.add(f"session_{cleaned}")
    return normalized


def discover_input_files(
    input_root: Path,
    split: str,
    session_filters: set[str],
) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    input_files: list[Path] = []
    for session_dir in sorted(input_root.glob("session_*")):
        if not session_dir.is_dir():
            continue
        if session_filters and session_dir.name not in session_filters:
            continue

        candidate = session_dir / f"{split}.txt"
        if candidate.exists():
            input_files.append(candidate)

    if not input_files:
        filters_text = ", ".join(sorted(session_filters)) if session_filters else "all sessions"
        raise FileNotFoundError(
            f"No input files found for split '{split}' under {input_root} ({filters_text})."
        )

    return input_files


def parse_session_number(session_folder: str) -> int | None:
    match = re.search(r"session_(\d+)$", session_folder)
    if not match:
        return None
    return int(match.group(1))


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def parse_speaker_index(raw_id: Any) -> int | None:
    if not isinstance(raw_id, str):
        return None
    match = re.match(r"bot_(\d+)$", raw_id.strip())
    if not match:
        return None
    return int(match.group(1))


def normalize_turn(turn: dict[str, Any], turn_index: int) -> dict[str, Any]:
    speaker_id = (turn.get("id") or f"bot_{turn_index % 2}").strip()
    speaker_index = parse_speaker_index(speaker_id)

    return {
        "turn_index": turn_index,
        "speaker_id": speaker_id,
        "speaker_index": speaker_index,
        "text": normalize_whitespace(turn.get("text")),
        "convai2_id": turn.get("convai2_id"),
        "persona_text": normalize_whitespace(turn.get("persona_text")),
        "agg_persona_list": [
            normalize_whitespace(item)
            for item in (turn.get("agg_persona_list") or [])
            if normalize_whitespace(item)
        ],
        "problem_data": turn.get("problem_data") or {},
    }


def flatten_dialog(dialog: list[dict[str, Any]]) -> str:
    lines = []
    for turn_index, turn in enumerate(dialog):
        text = normalize_whitespace(turn.get("text"))
        if not text:
            continue
        speaker_id = (turn.get("id") or f"bot_{turn_index % 2}").strip()
        lines.append(f"{speaker_id}: {text}")
    return "\n".join(lines)


def build_example_id(initial_data_id: str | None, session_folder: str, line_number: int) -> tuple[str, str]:
    base_id = initial_data_id or f"unknown_episode_{line_number}"
    episode_id = f"{base_id}:{session_folder}"
    example_id = f"{episode_id}:line_{line_number}"
    return episode_id, example_id


def normalize_record(record: dict[str, Any], input_file: Path, line_number: int, split: str) -> dict[str, Any] | None:
    dialog = record.get("dialog")
    if not isinstance(dialog, list) or not dialog:
        return None

    session_folder = input_file.parent.name
    session_number = parse_session_number(session_folder)
    initial_data_id = record.get("initial_data_id")
    episode_id, example_id = build_example_id(initial_data_id, session_folder, line_number)
    annotated_dialogue = [normalize_turn(turn, idx) for idx, turn in enumerate(dialog)]

    return {
        "source": "local_msc_personasummary",
        "task": "msc_persona_summary",
        "split": split,
        "episode_id": episode_id,
        "example_id": example_id,
        "text": flatten_dialog(dialog),
        "labels": [],
        "eval_labels": [],
        "label_candidates": [],
        "metadata": {
            "session_id": session_number,
            "session_folder": session_folder,
            "speaker": None,
            "previous_persona": None,
            "initial_data_id": initial_data_id,
            "dataset_name": "msc_personasummary",
            "source_file": input_file.as_posix(),
            "dialogue_turn_count": len(annotated_dialogue),
            "followup": normalize_whitespace(record.get("followup")),
            "newfact": normalize_whitespace(record.get("newfact")),
            "init_personachat": record.get("init_personachat"),
            "annotated_dialogue": annotated_dialogue,
            "raw_local_example": record,
        },
    }


def export_examples(args: argparse.Namespace) -> int:
    session_filters = normalize_session_filters(args.session)
    input_files = discover_input_files(args.input_root, args.split, session_filters)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written_examples = 0
    skipped_examples = 0

    with args.output.open("w", encoding="utf-8") as output_handle:
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8") as input_handle:
                for line_number, raw_line in enumerate(input_handle, start=1):
                    if args.max_examples is not None and written_examples >= args.max_examples:
                        print(
                            f"Reached max examples limit ({args.max_examples}).",
                            file=sys.stderr,
                        )
                        print_summary(args.output, input_files, written_examples, skipped_examples)
                        return 0

                    stripped_line = raw_line.strip()
                    if not stripped_line:
                        continue

                    try:
                        record = json.loads(stripped_line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in {input_file} at line {line_number}: {error}"
                        ) from error

                    normalized_record = normalize_record(record, input_file, line_number, args.split)
                    if normalized_record is None:
                        skipped_examples += 1
                        print(
                            f"Skipped {input_file.as_posix()} line {line_number}: missing dialog.",
                            file=sys.stderr,
                        )
                        continue

                    output_handle.write(json.dumps(normalized_record, ensure_ascii=False) + "\n")
                    written_examples += 1

    print_summary(args.output, input_files, written_examples, skipped_examples)
    return 0


def print_summary(
    output_path: Path,
    input_files: list[Path],
    written_examples: int,
    skipped_examples: int,
) -> None:
    input_labels = ", ".join(path.as_posix() for path in input_files)
    print(f"Input files: {input_labels}")
    print(f"Wrote {written_examples} examples to {output_path.as_posix()}")
    print(f"Skipped {skipped_examples} examples")


def main() -> int:
    args = parse_args()
    return export_examples(args)


if __name__ == "__main__":
    raise SystemExit(main())