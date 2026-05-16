from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


IGNORE_INDEX = -100


class TokenizerProtocol(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ...


@dataclass(slots=True)
class ProcessedExample:
    id: str
    task: str
    source: str
    split: str
    text: str
    target_start_marker: str
    target_text: str
    metadata: dict[str, Any]
    input: dict[str, Any] | None = None
    target: dict[str, Any] | None = None


class ProcessedSFTDataset:
    def __init__(
        self,
        path: str | Path,
        tokenizer: TokenizerProtocol,
        *,
        max_length: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_examples(self.path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        prompt_text = build_prompt_text(example.text, example.target_text)
        prompt_ids = tokenize_text(
            self.tokenizer,
            prompt_text,
            max_length=self.max_length,
        )
        full_ids = tokenize_text(
            self.tokenizer,
            example.text,
            max_length=self.max_length,
        )

        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = [IGNORE_INDEX] * prompt_length + full_ids[prompt_length:]

        if len(labels) < len(full_ids):
            labels.extend([IGNORE_INDEX] * (len(full_ids) - len(labels)))

        return {
            "id": example.id,
            "task": example.task,
            "source": example.source,
            "split": example.split,
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "target_start_marker": example.target_start_marker,
            "target_text": example.target_text,
            "metadata": example.metadata,
            "input": example.input,
            "target": example.target,
        }

    @staticmethod
    def _load_examples(path: Path) -> list[ProcessedExample]:
        examples: list[ProcessedExample] = []

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue

                try:
                    record = json.loads(stripped_line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path} at line {line_number}: {error}") from error

                examples.append(
                    ProcessedExample(
                        id=record["id"],
                        task=record["task"],
                        source=record["source"],
                        split=record["split"],
                        text=record["text"],
                        target_start_marker=record["target_start_marker"],
                        target_text=record["target_text"],
                        metadata=record.get("metadata") or {},
                        input=record.get("input"),
                        target=record.get("target"),
                    )
                )

        return examples


class ProcessedSFTCollator:
    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        *,
        pad_to_multiple_of: int | None = None,
        return_tensors: str = "pt",
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("ProcessedSFTCollator requires at least one feature.")

        pad_token_id = resolve_pad_token_id(self.tokenizer)
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        padded_input_ids: list[list[int]] = []
        padded_attention_masks: list[list[int]] = []
        padded_labels: list[list[int]] = []
        example_ids: list[str] = []
        tasks: list[str] = []
        metadata: list[dict[str, Any]] = []

        for feature in features:
            current_length = len(feature["input_ids"])
            pad_length = max_length - current_length

            padded_input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
            padded_attention_masks.append(feature["attention_mask"] + [0] * pad_length)
            padded_labels.append(feature["labels"] + [IGNORE_INDEX] * pad_length)
            example_ids.append(feature["id"])
            tasks.append(feature["task"])
            metadata.append(feature["metadata"])

        batch: dict[str, Any] = {
            "id": example_ids,
            "task": tasks,
            "metadata": metadata,
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels,
        }

        if self.return_tensors == "pt":
            try:
                import torch
            except ImportError as error:
                raise ImportError(
                    "PyTorch is required when return_tensors='pt'. Install torch or set return_tensors='python'."
                ) from error

            batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
            batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        elif self.return_tensors != "python":
            raise ValueError(f"Unsupported return_tensors value: {self.return_tensors}")

        return batch


def build_prompt_text(full_text: str, target_text: str) -> str:
    if not full_text.endswith(target_text):
        raise ValueError("Example text does not end with target_text; cannot derive prompt_text safely.")
    return full_text[: len(full_text) - len(target_text)]


def tokenize_text(
    tokenizer: TokenizerProtocol,
    text: str,
    *,
    max_length: int | None,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=max_length is not None,
        max_length=max_length,
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, list):
        raise ValueError("Tokenizer must return a dictionary with a list under 'input_ids'.")
    return [int(token_id) for token_id in input_ids]


def resolve_pad_token_id(tokenizer: TokenizerProtocol) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("Tokenizer must provide pad_token_id or eos_token_id for batch padding.")