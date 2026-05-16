from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import (  # noqa: E402
    ProcessedSFTCollator,
    ProcessedSFTDataset,
    assert_pipeline_special_tokens,
    format_special_token_report,
    register_pipeline_special_tokens,
)
from src.data.special_tokens import save_pipeline_special_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on the processed Stage A or Stage B JSONL datasets."
    )
    parser.add_argument("--model-name-or-path", required=True, help="Base model passed to AutoTokenizer/AutoModelForCausalLM.")
    parser.add_argument("--train-file", type=Path, required=True, help="Processed train JSONL file.")
    parser.add_argument("--valid-file", type=Path, default=None, help="Optional processed valid JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Trainer output directory.")
    parser.add_argument("--max-length", type=int, default=2048, help="Tokenization max_length for both prompt and full text.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Trainer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Trainer weight decay.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Trainer num_train_epochs.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Trainer logging_steps.")
    parser.add_argument("--save-steps", type=int, default=500, help="Trainer save_steps.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Trainer eval_steps when valid data is provided.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Trainer save_total_limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--pad-to-multiple-of", type=int, default=8, help="Optional padding multiple for the collator.")
    parser.add_argument("--report-to", default="none", help="Trainer report_to target, or 'none'.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Optional Trainer resume_from_checkpoint value.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Forward trust_remote_code=True to HF loaders.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing on the model.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 in TrainingArguments.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 in TrainingArguments.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars in Trainer.")
    parser.add_argument("--overwrite-output-dir", action="store_true", help="Allow writing into a non-empty output dir.")
    return parser.parse_args()


def main() -> int:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as error:
        raise ImportError("transformers is required to run train_sft.py.") from error

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    added_token_count = register_pipeline_special_tokens(tokenizer)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added_token_count += tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    verification = assert_pipeline_special_tokens(tokenizer)
    print(format_special_token_report(verification))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    if added_token_count > 0 or model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = ProcessedSFTDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
    )
    eval_dataset = None
    if args.valid_file is not None:
        eval_dataset = ProcessedSFTDataset(
            args.valid_file,
            tokenizer,
            max_length=args.max_length,
        )

    collator = ProcessedSFTCollator(
        tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of,
        return_tensors="pt",
        include_metadata=False,
    )

    training_args = build_training_arguments(args, eval_dataset is not None)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    save_pipeline_special_tokens(args.output_dir / "pipeline_special_tokens.json", extra_tokens=[tokenizer.pad_token])
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    print(f"Train examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval examples: {len(eval_dataset)}")
    print(f"Training loss: {train_result.training_loss}")

    return 0


def build_training_arguments(args: argparse.Namespace, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    report_to = [] if args.report_to == "none" else [args.report_to]
    evaluation_strategy = "steps" if has_eval else "no"

    return TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=args.overwrite_output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if has_eval else None,
        save_total_limit=args.save_total_limit,
        evaluation_strategy=evaluation_strategy,
        report_to=report_to,
        bf16=args.bf16,
        fp16=args.fp16,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns=False,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())