import json
import os
import torch
from src.data.training_dataset import ProcessedSFTDataset, ProcessedSFTCollator

tmp_file = "temp_smoke_test.jsonl"
example = {
    "id": "1",
    "task": "sft",
    "source": "manual",
    "split": "train",
    "text": "<|task|> response\n<|assistant|>\nhello<|endoftext|>",
    "target_start_marker": "<|assistant|>\n",
    "target_text": "hello<|endoftext|>",
    "metadata": {}
}
with open(tmp_file, "w") as f:
    f.write(json.dumps(example) + "\n")

class TinyTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 99
    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) % 90 + 1 for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}

try:
    tokenizer = TinyTokenizer()
    dataset = ProcessedSFTDataset(tmp_file, tokenizer, max_length=128)
    item = dataset[0]
    input_ids, labels = item["input_ids"], item["labels"]
    
    prompt_text = "<|task|> response\n<|assistant|>\n"
    prompt_ids = tokenizer(prompt_text)["input_ids"]
    prompt_len = len(prompt_ids)
    
    # Assertions for dataset
    assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} != {len(labels)}"
    for i in range(prompt_len):
        assert labels[i] == -100, f"Index {i} ({input_ids[i]}) should be -100, got {labels[i]}"
    for i in range(prompt_len, len(labels)):
        assert labels[i] == input_ids[i], f"Index {i} should match input_ids"

    # Testing Collator
    collator = ProcessedSFTCollator(tokenizer=tokenizer, return_tensors='pt')
    batch = collator([item, item])
    assert batch["input_ids"].shape == (2, len(input_ids))
    assert torch.all(batch["labels"][:, :prompt_len] == -100)
    
    print("SMOKE TEST SUCCESSFUL")
finally:
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
