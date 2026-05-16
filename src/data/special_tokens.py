from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PIPELINE_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|memory|>",
    "<|dialog|>",
    "<|task|>",
)

VERIFICATION_SAMPLE = (
    "<|task|> response\n"
    "<|memory|>\n"
    "The user likes hiking.\n\n"
    "<|dialog|>\n"
    "<|system|> Keep responses short.\n"
    "<|user|> Guess what I did this weekend?\n"
    "<|assistant|>\n"
)


@dataclass(slots=True)
class SpecialTokenCheckResult:
    token: str
    token_ids: list[int]
    decoded_token: str
    standalone_single_token: bool
    exact_roundtrip: bool
    context_occurrences_match: bool

    @property
    def is_stable(self) -> bool:
        return (
            self.standalone_single_token
            and self.exact_roundtrip
            and self.context_occurrences_match
        )


def register_pipeline_special_tokens(tokenizer: Any) -> int:
    return int(
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(PIPELINE_SPECIAL_TOKENS)}
        )
    )


def verify_pipeline_special_tokens(tokenizer: Any) -> list[SpecialTokenCheckResult]:
    sample_ids = tokenizer(VERIFICATION_SAMPLE, add_special_tokens=False)["input_ids"]
    decoded_sample = decode_tokens(tokenizer, sample_ids)
    results: list[SpecialTokenCheckResult] = []

    for token in PIPELINE_SPECIAL_TOKENS:
        token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        decoded_token = decode_tokens(tokenizer, token_ids)
        results.append(
            SpecialTokenCheckResult(
                token=token,
                token_ids=[int(token_id) for token_id in token_ids],
                decoded_token=decoded_token,
                standalone_single_token=len(token_ids) == 1,
                exact_roundtrip=decoded_token == token,
                context_occurrences_match=decoded_sample.count(token) == VERIFICATION_SAMPLE.count(token),
            )
        )

    return results


def assert_pipeline_special_tokens(tokenizer: Any) -> list[SpecialTokenCheckResult]:
    results = verify_pipeline_special_tokens(tokenizer)
    unstable = [result for result in results if not result.is_stable]
    if unstable:
        raise ValueError(format_special_token_report(unstable))
    return results


def format_special_token_report(results: Sequence[SpecialTokenCheckResult]) -> str:
    lines = []
    for result in results:
        lines.append(
            (
                f"{result.token}: ids={result.token_ids}, "
                f"single={result.standalone_single_token}, "
                f"roundtrip={result.exact_roundtrip}, "
                f"context={result.context_occurrences_match}"
            )
        )
    return "\n".join(lines)


def save_pipeline_special_tokens(path: str | Path, extra_tokens: Iterable[str] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = list(PIPELINE_SPECIAL_TOKENS)
    if extra_tokens:
        tokens.extend(extra_tokens)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"special_tokens": tokens}, handle, ensure_ascii=False, indent=2)


def decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)