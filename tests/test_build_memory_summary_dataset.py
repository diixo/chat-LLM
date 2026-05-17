from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from build_memory_summary_dataset import build_record  # noqa: E402


class BuildMemorySummaryDatasetTest(unittest.TestCase):
    def test_build_record_uses_init_persona_fallback_for_missing_speaker_memory(self) -> None:
        raw_record = {
            "split": "train",
            "episode_id": "train:ordered_1:session_1",
            "example_id": "train:ordered_1:session_1:line_1",
            "metadata": {
                "session_id": 1,
                "session_folder": "session_1",
                "initial_data_id": "train:ordered_1",
                "annotated_dialogue": [
                    {
                        "speaker_id": "bot_0",
                        "speaker_index": 0,
                        "text": "Hi there",
                        "agg_persona_list": [],
                        "persona_text": "",
                        "problem_data": {},
                    },
                    {
                        "speaker_id": "bot_1",
                        "speaker_index": 1,
                        "text": "I work in a library",
                        "agg_persona_list": ["I work in a library."],
                        "persona_text": "I work in a library.",
                        "problem_data": {"persona": "I work in a library."},
                    },
                ],
                "init_personachat": {
                    "init_personas": [
                        [
                            "I have two dogs.",
                            "My favorite hobby is running. I run every morning.",
                        ],
                        ["I work in a library."],
                    ]
                },
            },
        }

        processed_record = build_record(raw_record, target_speaker_index=0, output_index=1)

        self.assertIsNotNone(processed_record)
        self.assertEqual(
            processed_record["target"]["memory"],
            [
                "The user has two dogs.",
                "The user's favorite hobby is running.",
                "I run every morning.",
            ],
        )
        self.assertEqual(processed_record["metadata"]["target_speaker_index"], 0)
        self.assertEqual(processed_record["metadata"]["initial_data_id"], "train:ordered_1")


if __name__ == "__main__":
    unittest.main()