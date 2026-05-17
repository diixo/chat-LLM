from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


END_OF_TEXT = "<|endoftext|>"
MEMORY_START_MARKER = "<|memory|>\n"
ASSISTANT_START_MARKER = "<|assistant|>\n"

_SESSION_NUMBER_PATTERN = re.compile(r"(?:session_)?(\d+)$", re.IGNORECASE)
_SPEAKER_PATTERNS = (
    re.compile(r"^speaker\s+(\d+)$", re.IGNORECASE),
    re.compile(r"^bot_(\d+)$", re.IGNORECASE),
)
_PUNCTUATION_SPACING_FIXES: tuple[tuple[str, str], ...] = (
    (" .", "."),
    (" ,", ","),
    (" ?", "?"),
    (" !", "!"),
    (" ' ", "'"),
)
_THIRD_PERSON_IRREGULAR_VERBS: dict[str, str] = {
    "am": "is",
    "are": "is",
    "be": "is",
    "do": "does",
    "go": "goes",
    "have": "has",
}
_FACT_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^I've been\b", re.IGNORECASE), "The user has been"),
    (re.compile(r"^I am\b", re.IGNORECASE), "The user is"),
    (re.compile(r"^I'm\b", re.IGNORECASE), "The user is"),
    (re.compile(r"^I was\b", re.IGNORECASE), "The user was"),
    (re.compile(r"^I have\b", re.IGNORECASE), "The user has"),
    (re.compile(r"^I have never\b", re.IGNORECASE), "The user has never"),
    (re.compile(r"^I've\b", re.IGNORECASE), "The user has"),
    (re.compile(r"^I had\b", re.IGNORECASE), "The user had"),
    (re.compile(r"^I like\b", re.IGNORECASE), "The user likes"),
    (re.compile(r"^I love\b", re.IGNORECASE), "The user loves"),
    (re.compile(r"^I enjoy\b", re.IGNORECASE), "The user enjoys"),
    (re.compile(r"^I need\b", re.IGNORECASE), "The user needs"),
    (re.compile(r"^I want\b", re.IGNORECASE), "The user wants"),
    (re.compile(r"^I plan\b", re.IGNORECASE), "The user plans"),
    (re.compile(r"^I work as\b", re.IGNORECASE), "The user works as"),
    (re.compile(r"^I work\b", re.IGNORECASE), "The user works"),
    (re.compile(r"^I live\b", re.IGNORECASE), "The user lives"),
    (re.compile(r"^I take\b", re.IGNORECASE), "The user takes"),
    (re.compile(r"^I learned\b", re.IGNORECASE), "The user learned"),
    (re.compile(r"^I value\b", re.IGNORECASE), "The user values"),
    (re.compile(r"^I check\b", re.IGNORECASE), "The user checks"),
    (re.compile(r"^I changed\b", re.IGNORECASE), "The user changed"),
    (re.compile(r"^I gained\b", re.IGNORECASE), "The user gained"),
    (re.compile(r"^I grow\b", re.IGNORECASE), "The user grows"),
    (re.compile(r"^I picked\b", re.IGNORECASE), "The user picked"),
    (re.compile(r"^I started\b", re.IGNORECASE), "The user started"),
    (re.compile(r"^I booked\b", re.IGNORECASE), "The user booked"),
    (re.compile(r"^I troubleshoot\b", re.IGNORECASE), "The user troubleshoots"),
    (re.compile(r"^I test\b", re.IGNORECASE), "The user tests"),
    (re.compile(r"^I used to\b", re.IGNORECASE), "The user used to"),
    (re.compile(r"^I do not\b", re.IGNORECASE), "The user does not"),
    (re.compile(r"^I don't\b", re.IGNORECASE), "The user does not"),
    (re.compile(r"^I can\b", re.IGNORECASE), "The user can"),
    (re.compile(r"^I will\b", re.IGNORECASE), "The user will"),
    (re.compile(r"^I finally\b", re.IGNORECASE), "The user finally"),
    (re.compile(r"^My\b", re.IGNORECASE), "The user's"),
)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            try:
                yield json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {error}") from error


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_session_number(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = _SESSION_NUMBER_PATTERN.search(value.strip())
        if match:
            return int(match.group(1))
    return None


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = normalize_whitespace(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def uppercase_first_character(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def normalize_reply_text(text: str | None, add_trailing_punctuation: bool = False) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""

    normalized = cleaned.lower()
    for spaced, compact in _PUNCTUATION_SPACING_FIXES:
        normalized = normalized.replace(compact, spaced).replace("  ", " ")

    tokens = normalized.split(" ")
    for index, token in enumerate(tokens):
        if not token:
            continue
        if index == 0:
            tokens[index] = uppercase_first_character(token)
        elif token in {"i", "i'm", "i've", "i'll", "i'd"}:
            tokens[index] = uppercase_first_character(token)
        elif token in {"?", ".", "!"} and index + 1 < len(tokens):
            tokens[index + 1] = uppercase_first_character(tokens[index + 1])

    normalized = f" {' '.join(tokens)} "
    for spaced, compact in _PUNCTUATION_SPACING_FIXES:
        normalized = normalized.replace(spaced, compact)

    normalized = normalize_whitespace(normalized)
    if add_trailing_punctuation and normalized and normalized[-1] not in '!.?)"\'':
        normalized += "."
    return normalized


def ensure_terminal_punctuation(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""
    if cleaned.endswith((".", "!", "?")):
        return cleaned
    return f"{cleaned}."


def parse_speaker_index(raw_value: Any) -> int | None:
    if isinstance(raw_value, int):
        return raw_value
    if not isinstance(raw_value, str):
        return None

    cleaned = raw_value.strip()
    for pattern in _SPEAKER_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            number = int(match.group(1))
            if cleaned.lower().startswith("speaker"):
                return number - 1
            return number
    return None


def available_speaker_indices(turns: list[dict[str, Any]]) -> list[int]:
    fallback_order: dict[str, int] = {}
    indices: list[int] = []
    seen: set[int] = set()

    for turn in turns:
        speaker_index = turn.get("speaker_index")
        if speaker_index is None:
            speaker_value = turn.get("speaker_id") or turn.get("speaker")
            speaker_index = parse_speaker_index(speaker_value)
            if speaker_index is None:
                key = str(speaker_value) if speaker_value is not None else f"unknown_{len(fallback_order)}"
                speaker_index = fallback_order.setdefault(key, len(fallback_order))

        if speaker_index not in seen:
            seen.add(speaker_index)
            indices.append(speaker_index)

    return indices


def orient_dialogue_turns(
    turns: list[dict[str, Any]],
    target_user_speaker_index: int,
) -> list[dict[str, str]]:
    fallback_order: dict[str, int] = {}
    oriented_turns: list[dict[str, str]] = []

    for turn in turns:
        content = normalize_reply_text(turn.get("text") or turn.get("content"))
        if not content:
            continue

        speaker_index = turn.get("speaker_index")
        if speaker_index is None:
            speaker_value = turn.get("speaker_id") or turn.get("speaker")
            speaker_index = parse_speaker_index(speaker_value)
            if speaker_index is None:
                key = str(speaker_value) if speaker_value is not None else f"unknown_{len(fallback_order)}"
                speaker_index = fallback_order.setdefault(key, len(fallback_order))

        role = "user" if speaker_index == target_user_speaker_index else "assistant"
        oriented_turns.append({"role": role, "content": content})

    return oriented_turns


def render_dialogue_block(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"<|{turn['role']}|> {turn['content']}" for turn in turns)


def split_fact_candidates(text: str | None) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part for part in parts if part]


def conjugate_third_person_singular(verb: str) -> str:
    lowered = verb.lower()
    if lowered in _THIRD_PERSON_IRREGULAR_VERBS:
        return _THIRD_PERSON_IRREGULAR_VERBS[lowered]
    if lowered.endswith("y") and len(lowered) > 1 and lowered[-2] not in "aeiou":
        return f"{lowered[:-1]}ies"
    if lowered.endswith(("s", "sh", "ch", "x", "z", "o")):
        return f"{lowered}es"
    return f"{lowered}s"


def rewrite_generic_first_person_fact(text: str) -> str | None:
    match = re.match(r"^I\s+([A-Za-z']+)(.*)$", text, flags=re.IGNORECASE)
    if not match:
        return None

    verb = match.group(1)
    remainder = match.group(2) or ""
    conjugated_verb = conjugate_third_person_singular(verb)
    return f"The user {conjugated_verb}{remainder}"


def rewrite_fact_to_user_perspective(fact: str) -> str:
    cleaned = normalize_whitespace(fact)
    if not cleaned:
        return ""
    if cleaned.lower().startswith("the user"):
        rewritten = rewrite_embedded_first_person_pronouns(cleaned)
        return ensure_terminal_punctuation(normalize_reply_text(rewritten))

    for pattern, replacement in _FACT_REWRITES:
        if pattern.search(cleaned):
            rewritten = pattern.sub(replacement, cleaned, count=1)
            rewritten = rewrite_embedded_first_person_pronouns(rewritten)
            return ensure_terminal_punctuation(normalize_reply_text(rewritten))

    generic_rewrite = rewrite_generic_first_person_fact(cleaned)
    if generic_rewrite is not None:
        generic_rewrite = rewrite_embedded_first_person_pronouns(generic_rewrite)
        return ensure_terminal_punctuation(normalize_reply_text(generic_rewrite))

    return ensure_terminal_punctuation(normalize_reply_text(cleaned))


def rewrite_embedded_first_person_pronouns(text: str) -> str:
    rewritten = re.sub(r"\bmy\b", "their", text, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bmine\b", "theirs", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bme\b", "them", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\band am\b", "and is", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bbut am\b", "but is", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r", am\b", ", is", rewritten, flags=re.IGNORECASE)
    return rewritten


def extract_memory_items_from_annotated_dialogue(
    turns: list[dict[str, Any]],
    target_speaker_index: int,
) -> list[str]:
    latest_agg_list: list[str] | None = None

    for turn in turns:
        if turn.get("speaker_index") != target_speaker_index:
            continue

        agg_persona_list = turn.get("agg_persona_list") or []
        cleaned_agg = [rewrite_fact_to_user_perspective(item) for item in agg_persona_list if normalize_whitespace(item)]
        cleaned_agg = dedupe_preserve_order(cleaned_agg)
        if cleaned_agg:
            latest_agg_list = cleaned_agg

    if latest_agg_list:
        return latest_agg_list

    fallback_items: list[str] = []
    for turn in turns:
        if turn.get("speaker_index") != target_speaker_index:
            continue
        fallback_items.extend(split_fact_candidates(turn.get("persona_text")))
        problem_data = turn.get("problem_data") or {}
        fallback_items.extend(split_fact_candidates(problem_data.get("persona")))

    rewritten_items = [rewrite_fact_to_user_perspective(item) for item in fallback_items]
    return dedupe_preserve_order(rewritten_items)


def parse_memory_items_from_target_text(target_text: str | None) -> list[str]:
    raw_text = (target_text or "").replace(END_OF_TEXT, "")
    if not raw_text.strip():
        return []
    return [
        normalize_whitespace(line)
        for line in raw_text.splitlines()
        if normalize_whitespace(line)
    ]