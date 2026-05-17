from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from build_convai2_response_dataset import build_record, load_episodes  # noqa: E402
from build_response_sft_dataset import enrich_convai2_session1_record  # noqa: E402


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def iter_first_jsonl_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            records.append(json.loads(stripped_line))
            if len(records) >= limit:
                break
    return records


def initial_data_id_to_episode_index(initial_data_id: str) -> int:
    train_prefix = "train:ordered_"
    if initial_data_id.startswith(train_prefix):
        return int(initial_data_id[len(train_prefix) :])

    split, separator, suffix = initial_data_id.partition("_")
    if separator and split in {"valid", "test"} and suffix.isdigit():
        return int(suffix)

    raise ValueError(f"Unsupported initial_data_id format: {initial_data_id}")


def flatten_episode_turns(episode: dict[str, Any]) -> list[str]:
    turns: list[str] = []
    for user_text, assistant_text in episode.get("dialogue_pairs") or []:
        turns.append(user_text)
        turns.append(assistant_text)
    return turns


class ConvAI2Session1MappingTest(unittest.TestCase):
    def assert_msc_history_matches_convai2(self, split: str, convai2_relpath: str, msc_relpath: str) -> None:
        convai2_path = PROJECT_ROOT / convai2_relpath
        msc_path = PROJECT_ROOT / msc_relpath

        if not convai2_path.exists() or not msc_path.exists():
            self.skipTest(f"Required local dataset files are missing for split {split}.")

        episodes = load_episodes(convai2_path)
        raw_records = iter_first_jsonl_records(msc_path, limit=3)
        self.assertGreaterEqual(len(raw_records), 1)

        for output_index, raw_record in enumerate(raw_records, start=1):
            metadata = raw_record.get("metadata") or {}
            initial_data_id = metadata.get("initial_data_id")
            previous_dialogue_history = metadata.get("previous_dialogue_history") or []

            with self.subTest(split=split, initial_data_id=initial_data_id):
                self.assertIsInstance(initial_data_id, str)
                self.assertGreaterEqual(len(previous_dialogue_history), 1)

                episode_index = initial_data_id_to_episode_index(initial_data_id)
                self.assertLess(episode_index, len(episodes))

                episode = episodes[episode_index]
                previous_dialogue = previous_dialogue_history[0].get("dialog") or []
                msc_turns = [turn.get("text", "") for turn in previous_dialogue]
                convai2_turns = flatten_episode_turns(episode)

                self.assertEqual(len(convai2_turns), len(msc_turns))
                for convai2_turn, msc_turn in zip(convai2_turns, msc_turns):
                    self.assertEqual(normalize_tokens(convai2_turn), normalize_tokens(msc_turn))

                first_turn_record = build_record(
                    episode=episode,
                    episode_index=episode_index,
                    dialogue_pair_index=0,
                    split=split,
                    output_index=output_index,
                    input_path=convai2_path,
                )
                self.assertIsNotNone(first_turn_record)

                enriched_record = enrich_convai2_session1_record(first_turn_record)
                self.assertEqual(enriched_record["metadata"]["initial_data_id"], initial_data_id)
                self.assertEqual(enriched_record["metadata"]["source_session_number"], 1)
                self.assertEqual(enriched_record["metadata"]["memory_source_kind"], "convai2_persona")
                self.assertEqual(
                    normalize_tokens(enriched_record["input"]["current_dialogue"][0]["content"]),
                    normalize_tokens(msc_turns[0]),
                )
                self.assertEqual(
                    normalize_tokens(enriched_record["target"]["assistant_response"]),
                    normalize_tokens(msc_turns[1]),
                )

    def test_train_and_valid_session1_records_resolve_to_matching_convai2_episodes(self) -> None:
        self.assert_msc_history_matches_convai2(
            split="train",
            convai2_relpath="data/convai2/train_self_original_no_cands.txt",
            msc_relpath="datasets/raw/msc/train.jsonl",
        )
        self.assert_msc_history_matches_convai2(
            split="valid",
            convai2_relpath="data/convai2/valid_self_original_no_cands.txt",
            msc_relpath="datasets/raw/msc/valid.jsonl",
        )


if __name__ == "__main__":
    unittest.main()