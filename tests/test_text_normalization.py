from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from build_convai2_response_dataset import load_episodes  # noqa: E402
from msc_processed_utils import normalize_reply_text, orient_dialogue_turns, rewrite_fact_to_user_perspective  # noqa: E402


class TextNormalizationTest(unittest.TestCase):
    def test_normalize_reply_text_matches_parlai_style_sentence_casing(self) -> None:
        self.assertEqual(normalize_reply_text("hi , how are you doing ?"), "Hi, how are you doing?")
        self.assertEqual(
            normalize_reply_text("i'm doing great . just relaxing with my two dogs ."),
            "I'm doing great. Just relaxing with my two dogs.",
        )

    def test_load_episodes_normalizes_convai2_persona_and_dialogue(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "mini_convai2.txt"
            input_path.write_text(
                "1 your persona: i like iced tea.\n"
                "2 hi , how are you doing ?\ti'm doing great . just relaxing with my two dogs .\n",
                encoding="utf-8",
            )
            episodes = load_episodes(input_path)

        self.assertEqual(episodes[0]["personality"], ["I like iced tea."])
        self.assertEqual(
            episodes[0]["dialogue_pairs"],
            [("Hi, how are you doing?", "I'm doing great. Just relaxing with my two dogs.")],
        )

    def test_orient_dialogue_turns_normalizes_msc_dialogue(self) -> None:
        turns = [
            {"speaker_index": 0, "text": "hi , how are you doing ?"},
            {"speaker_index": 1, "text": "i'm doing great ."},
        ]

        self.assertEqual(
            orient_dialogue_turns(turns, target_user_speaker_index=0),
            [
                {"role": "user", "content": "Hi, how are you doing?"},
                {"role": "assistant", "content": "I'm doing great."},
            ],
        )

    def test_rewrite_fact_to_user_perspective_rewrites_generic_first_person_facts(self) -> None:
        self.assertEqual(rewrite_fact_to_user_perspective("i wear contacts."), "The user wears contacts.")
        self.assertEqual(rewrite_fact_to_user_perspective("I read books."), "The user reads books.")


if __name__ == "__main__":
    unittest.main()