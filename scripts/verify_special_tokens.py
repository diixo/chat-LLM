from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.special_tokens import (  # noqa: E402
    assert_pipeline_special_tokens,
    format_special_token_report,
    register_pipeline_special_tokens,
    save_pipeline_special_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register and verify the pipeline special tokens on a real Hugging Face tokenizer."
    )
    parser.add_argument("--model-name-or-path", required=True, help="Tokenizer source passed to AutoTokenizer.")
    parser.add_argument(
        "--save-tokenizer-dir",
        type=Path,
        default=None,
        help="Optional directory to save the updated tokenizer.",
    )
    parser.add_argument(
        "--save-token-list",
        type=Path,
        default=None,
        help="Optional JSON file that stores the pipeline special token list.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to AutoTokenizer.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError("transformers is required to verify special tokens.") from error

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    added_token_count = register_pipeline_special_tokens(tokenizer)
    verification = assert_pipeline_special_tokens(tokenizer)

    print(f"Tokenizer: {args.model_name_or_path}")
    print(f"Added pipeline special tokens: {added_token_count}")
    print(format_special_token_report(verification))

    if args.save_tokenizer_dir is not None:
        args.save_tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(args.save_tokenizer_dir)
        print(f"Saved tokenizer to {args.save_tokenizer_dir.as_posix()}")

    if args.save_token_list is not None:
        save_pipeline_special_tokens(args.save_token_list)
        print(f"Saved token list to {args.save_token_list.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())