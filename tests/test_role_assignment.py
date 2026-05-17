from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from build_convai2_response_dataset import build_record as build_convai2_record, load_episodes  # noqa: E402
from msc_processed_utils import orient_dialogue_turns  # noqa: E402


def find_jsonl_record(
    path: Path,
    *,
    initial_data_id: str,
    session_folder: str,
) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            record = json.loads(stripped_line)
            metadata = record.get("metadata") or {}
            if metadata.get("initial_data_id") == initial_data_id and metadata.get("session_folder") == session_folder:
                return record

    return None


def find_jsonl_record_by_id(path: Path, record_id: str) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            record = json.loads(stripped_line)
            if record.get("id") == record_id:
                return record

    return None


class RoleAssignmentTest(unittest.TestCase):
    def test_convai2_session1_roles_match_known_source_phrases(self) -> None:
        convai2_path = PROJECT_ROOT / "data/convai2/train_self_original_no_cands.txt"
        if not convai2_path.exists():
            self.skipTest("ConvAI2 train file is missing.")

        episode = load_episodes(convai2_path)[0]

        first_record = build_convai2_record(
            episode=episode,
            episode_index=0,
            dialogue_pair_index=0,
            split="train",
            output_index=1,
            input_path=convai2_path,
        )
        self.assertIsNotNone(first_record)
        self.assertEqual(
            first_record["input"]["current_dialogue"],
            [
                {
                    "role": "user",
                    "content": "Hi, how are you doing? I'm getting ready to do some cheetah chasing to stay in shape.",
                }
            ],
        )
        self.assertEqual(
            first_record["target"]["assistant_response"],
            "You must be very fast. Hunting is one of my favorite hobbies.",
        )

        second_record = build_convai2_record(
            episode=episode,
            episode_index=0,
            dialogue_pair_index=1,
            split="train",
            output_index=2,
            input_path=convai2_path,
        )
        self.assertIsNotNone(second_record)
        self.assertEqual(
            second_record["input"]["current_dialogue"][:3],
            [
                {
                    "role": "user",
                    "content": "Hi, how are you doing? I'm getting ready to do some cheetah chasing to stay in shape.",
                },
                {
                    "role": "assistant",
                    "content": "You must be very fast. Hunting is one of my favorite hobbies.",
                },
                {
                    "role": "user",
                    "content": "I am! For my hobby I like to do canning or some whittling.",
                },
            ],
        )
        self.assertEqual(
            second_record["target"]["assistant_response"],
            "I also remodel homes when I am not out bow hunting.",
        )

    def test_msc_history_orientation_matches_source_speaker_labels(self) -> None:
        msc_train_path = PROJECT_ROOT / "datasets/raw/msc/train.jsonl"
        if not msc_train_path.exists():
            self.skipTest("Raw MSC train JSONL is missing.")

        raw_record = find_jsonl_record(
            msc_train_path,
            initial_data_id="train:ordered_0",
            session_folder="session_2",
        )
        if raw_record is None:
            self.skipTest("Could not find the expected MSC raw record for train:ordered_0 / session_2.")

        previous_dialogue_history = (raw_record.get("metadata") or {}).get("previous_dialogue_history") or []
        self.assertGreaterEqual(len(previous_dialogue_history), 1)

        previous_dialogue_turns = previous_dialogue_history[0].get("dialog") or []

        oriented_for_speaker0 = orient_dialogue_turns(previous_dialogue_turns, target_user_speaker_index=0)
        self.assertEqual(
            oriented_for_speaker0[:3],
            [
                {
                    "role": "user",
                    "content": "Hi, how are you doing? I'm getting ready to do some cheetah chasing to stay in shape.",
                },
                {
                    "role": "assistant",
                    "content": "You must be very fast. Hunting is one of my favorite hobbies.",
                },
                {
                    "role": "user",
                    "content": "I am! For my hobby I like to do canning or some whittling.",
                },
            ],
        )

        oriented_for_speaker1 = orient_dialogue_turns(previous_dialogue_turns, target_user_speaker_index=1)
        self.assertEqual(
            oriented_for_speaker1[:3],
            [
                {
                    "role": "assistant",
                    "content": "Hi, how are you doing? I'm getting ready to do some cheetah chasing to stay in shape.",
                },
                {
                    "role": "user",
                    "content": "You must be very fast. Hunting is one of my favorite hobbies.",
                },
                {
                    "role": "assistant",
                    "content": "I am! For my hobby I like to do canning or some whittling.",
                },
            ],
        )

    def test_final_stage_b_artifact_preserves_known_role_assignments(self) -> None:
        stage_b_train_path = PROJECT_ROOT / "datasets/processed/msc_response_sft/train.jsonl"
        stage_b_valid_path = PROJECT_ROOT / "datasets/processed/msc_response_sft/valid.jsonl"
        if not stage_b_train_path.exists() or not stage_b_valid_path.exists():
            self.skipTest("Final Stage B processed artifacts are missing.")

        convai2_record = find_jsonl_record_by_id(stage_b_train_path, "convai2_response_train_000001")
        if convai2_record is None:
            self.skipTest("Could not find convai2_response_train_000001 in final Stage B train artifact.")

        self.assertEqual(convai2_record.get("source"), "convai2")
        self.assertEqual(
            convai2_record["input"]["current_dialogue"],
            [
                {
                    "role": "user",
                    "content": "Hi, how are you doing? I'm getting ready to do some cheetah chasing to stay in shape.",
                }
            ],
        )
        self.assertEqual(
            convai2_record["target"]["assistant_response"],
            "You must be very fast. Hunting is one of my favorite hobbies.",
        )

        msc_record = find_jsonl_record_by_id(stage_b_valid_path, "msc_response_valid_007813")
        if msc_record is None:
            self.skipTest("Could not find msc_response_valid_007813 in final Stage B valid artifact.")

        self.assertEqual(msc_record.get("source"), "msc")
        self.assertEqual(
            msc_record["input"]["current_dialogue"],
            [
                {
                    "role": "user",
                    "content": "What kind of dogs do you have?",
                }
            ],
        )
        self.assertEqual(
            msc_record["target"]["assistant_response"],
            "One is a terrier and one is a sheep dog. How long have you been volunteering?",
        )

        msc_reverse_record = find_jsonl_record_by_id(stage_b_valid_path, "msc_response_valid_007819")
        if msc_reverse_record is None:
            self.skipTest("Could not find msc_response_valid_007819 in final Stage B valid artifact.")

        self.assertEqual(msc_reverse_record.get("source"), "msc")
        self.assertEqual(
            msc_reverse_record["input"]["current_dialogue"][:2],
            [
                {
                    "role": "assistant",
                    "content": "What kind of dogs do you have?",
                },
                {
                    "role": "user",
                    "content": "One is a terrier and one is a sheep dog. How long have you been volunteering?",
                },
            ],
        )
        self.assertEqual(
            msc_reverse_record["target"]["assistant_response"],
            "I have been volunteering for last 7 years",
        )


if __name__ == "__main__":
    unittest.main()