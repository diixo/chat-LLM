from __future__ import annotations

import json
from pathlib import Path


INPUT_OUTPUT_PAIRS: tuple[tuple[Path, Path], ...] = (
    (
        Path("datasets/processed/msc_response_sft/train.jsonl"),
        Path("datasets/processed/msc_response_sft/train.json"),
    ),
    (
        Path("datasets/processed/msc_response_sft/valid.jsonl"),
        Path("datasets/processed/msc_response_sft/valid.json"),
    ),
)


def load_jsonl_records(input_path: Path) -> list[dict]:
    records: list[dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            try:
                records.append(json.loads(stripped_line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {input_path.as_posix()} at line {line_number}: {error}"
                ) from error

    return records


def convert_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path.as_posix()}")

    records = load_jsonl_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    print(f"Converted {input_path.as_posix()} -> {output_path.as_posix()} ({len(records)} records)")


def main() -> int:
    for input_path, output_path in INPUT_OUTPUT_PAIRS:
        convert_file(input_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())