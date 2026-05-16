# MSC Dataset Pipeline for Training a Custom Long-Term Memory Chat Model

## Goal

Build a data preparation and training pipeline based on the ParlAI MSC dataset.

The target architecture is a two-stage memory-based chat training pipeline:


1. Export MSC examples into project-local JSONL files.
2. Prepare the exported data into custom training formats.
3. Train a model that converts dialogue history into compact memory summaries.
4. Train a chat model that answers using `memory + current_dialog -> assistant_response`.

This document is intended as implementation guidance for Copilot or another coding assistant inside the project.

---


# 3. Export MSC examples into JSONL

Create script:

```text
scripts/export_msc_from_parlai.py
```

The script should:

1. Accept split argument:
   - `train`
   - `valid`
   - `test`

2. Normalize each example into a project-owned JSONL schema.

3. Write one JSON object per line.

Recommended CLI:

```bash
python scripts/export_msc_from_parlai.py \
  --task msc \
  --split train \
  --out data/raw/msc/train.jsonl
```

```bash
python scripts/export_msc_from_parlai.py \
  --task msc:PersonaSummary \
  --split train \
  --out data/raw/msc/persona_summary_train.jsonl
```

---

# 4. Raw JSONL schema for exported MSC examples

Use a conservative schema that preserves as much original data as possible.

Recommended raw schema:

```json
{
  "source": "parlai_msc",
  "task": "msc",
  "split": "train",
  "episode_id": "optional_episode_id",
  "example_id": "optional_example_id",
  "text": "raw input text from ParlAI",
  "labels": ["target response if available"],
  "eval_labels": ["target response if available"],
  "label_candidates": [],
  "metadata": {
    "session_id": null,
    "speaker": null,
    "previous_persona": null,
    "raw_parlai_example": {}
  }
}
```

Do not throw away fields that are unclear. Put them under `metadata.raw_parlai_example`.

Reason: ParlAI tasks may expose fields such as `text`, `labels`, `eval_labels`, `episode_done`, candidates, persona strings, or previous dialogue history. The export script should preserve these fields even if the first processing version does not use all of them.

---

# 5. Important parsing rule

The exported raw `text` field may contain several parts mixed together:

- current dialogue context
- previous session history
- persona or memory strings
- current user message
- special task formatting

Do not hard-code too much logic into the export script.

Instead, separate the pipeline into two stages:

```text
ParlAI example -> raw JSONL
raw JSONL -> project-specific training JSONL
```

The export script should be simple and stable.

The processing scripts should contain the task-specific parsing and formatting logic.

---

# 6. Final target: third training variant

The selected strategy is the two-stage variant:

```text
Stage A:
previous_dialogue_history -> memory_summary

Stage B:
memory_summary + current_dialogue -> next_assistant_response
```

This is better than training only on raw history because a small GPT-style model may not handle very long multi-session dialogue context well.

Instead of forcing the model to read all previous sessions every time, we train or prepare a compact memory representation.

---

# 7. Stage A dataset: dialogue history to memory summary

Create script:

```text
scripts/build_memory_summary_dataset.py
```

Input:

```text
data/raw/msc/persona_summary_train.jsonl
data/raw/msc/persona_summary_valid.jsonl
```

Output:

```text
data/processed/msc_memory_summary/train.jsonl
data/processed/msc_memory_summary/valid.jsonl
```

The goal is to build examples like:

```json
{
  "task": "memory_summary",
  "source": "msc",
  "input": {
    "previous_dialogue": [
      {"role": "user", "content": "I like hiking and old movies."},
      {"role": "assistant", "content": "That sounds fun. What kind of old movies do you enjoy?"}
    ]
  },
  "target": {
    "memory": [
      "The user likes hiking.",
      "The user likes old movies."
    ]
  }
}
```

For GPT-style SFT, convert this into text format:

```text
<|task|> memory_summary
<|dialog|>
<|user|> I like hiking and old movies.
<|assistant|> That sounds fun. What kind of old movies do you enjoy?

<|memory|>
The user likes hiking.
The user likes old movies.<|endoftext|>
```

Only the memory target should be trained as labels.

The prompt/input part should be masked with `-100`.

---

# 8. Stage A formatting rules

Use a stable text format.

Recommended special sections:

```text
<|task|> memory_summary
<|dialog|>
...
<|memory|>
...
```

Recommended role tokens:

```text
<|user|>
<|assistant|>
<|system|>
<|memory|>
<|dialog|>
<|task|>
```

For your GPT-style model, these can either be added as tokenizer special tokens or represented as normal text markers.

If the tokenizer is custom, register them as special tokens.

The memory summary should be:

- short
- factual
- written in third person
- stable across sessions
- useful for future conversation
- not a transcript
- not a reasoning chain

Good memory examples:

```text
The user likes hiking.
The user enjoys old movies.
The user has a dog.
The user is interested in cooking.
```

Bad memory examples:

```text
The user said many things about hobbies.
The assistant asked a question.
The conversation was friendly.
```

---

# 9. Stage A label masking

For decoder-only GPT training:

Input text:

```text
<|task|> memory_summary
<|dialog|>
<|user|> ...
<|assistant|> ...

<|memory|>
The user likes hiking.
The user likes old movies.<|endoftext|>
```

Labels:

```text
-100 for everything before <|memory|>
valid token labels only for:
The user likes hiking.
The user likes old movies.<|endoftext|>
```

The model should learn to generate only the memory summary.

---

# 10. Stage B dataset: memory + current dialogue to assistant response

Create script:

```text
scripts/build_response_sft_dataset.py
```

Input:

```text
data/raw/msc/train.jsonl
data/raw/msc/valid.jsonl
```

Optional input:

```text
data/processed/msc_memory_summary/train.jsonl
```

Output:

```text
data/processed/msc_response_sft/train.jsonl
data/processed/msc_response_sft/valid.jsonl
```

The goal is to build examples like:

```json
{
  "task": "response_sft",
  "source": "msc",
  "input": {
    "memory": [
      "The user likes hiking.",
      "The user likes old movies."
    ],
    "current_dialogue": [
      {"role": "user", "content": "Guess what I did this weekend?"}
    ]
  },
  "target": {
    "assistant_response": "Did you go hiking again?"
  }
}
```

For GPT-style SFT, convert this into text format:

```text
<|task|> response
<|memory|>
The user likes hiking.
The user likes old movies.

<|dialog|>
<|user|> Guess what I did this weekend?

<|assistant|>
Did you go hiking again?<|endoftext|>
```

Only the final assistant response should be trained as labels.

Everything before the final assistant answer should be masked with `-100`.

---

# 11. Stage B formatting rules

Use a consistent order:

```text
<|task|> response
<|memory|>
...

<|dialog|>
...

<|assistant|>
target response
```

The model should see memory before the current dialogue.

Reason: the memory acts like a compact long-term context.

Avoid placing large raw previous sessions directly into Stage B examples unless you intentionally create a long-context variant.

Recommended Stage B variants:

## Variant B1: memory only

```text
memory + current_dialogue -> response
```

This is the main target format.

## Variant B2: memory + limited recent history

```text
memory + last N turns from current session -> response
```

This is useful for realistic chat behavior.

## Variant B3: memory + retrieved facts

```text
memory + retrieved relevant memory facts + current_dialogue -> response
```

This is useful later if the project adds RAG.

---

# 12. Stage B label masking

For decoder-only GPT training:

Input text:

```text
<|task|> response
<|memory|>
The user likes hiking.

<|dialog|>
<|user|> What did I tell you I like?

<|assistant|>
You told me that you like hiking.<|endoftext|>
```

Labels:

```text
-100 for:
<|task|> response
<|memory|>
The user likes hiking.
<|dialog|>
<|user|> What did I tell you I like?
<|assistant|>

valid labels for:
You told me that you like hiking.<|endoftext|>
```

This matches the user's preferred last-assistant-only SFT style.

---

# 13. Recommended processed JSONL schema

Use one shared processed schema for both Stage A and Stage B.

```json
{
  "id": "msc_train_000001",
  "task": "response_sft",
  "source": "msc",
  "split": "train",
  "text": "<|task|> response\n<|memory|>\n...\n<|assistant|>\n...",
  "target_start_marker": "<|assistant|>\n",
  "target_text": "Did you go hiking again?<|endoftext|>",
  "metadata": {
    "episode_id": "optional",
    "session_id": "optional",
    "num_memory_items": 2,
    "num_dialog_turns": 1
  }
}
```

For memory summary:

```json
{
  "id": "msc_memory_train_000001",
  "task": "memory_summary",
  "source": "msc",
  "split": "train",
  "text": "<|task|> memory_summary\n<|dialog|>\n...\n<|memory|>\n...",
  "target_start_marker": "<|memory|>\n",
  "target_text": "The user likes hiking.\nThe user likes old movies.<|endoftext|>",
  "metadata": {
    "episode_id": "optional",
    "session_id": "optional"
  }
}
```

---

# 14. Tokenization and collate function

The dataset loader should tokenize the full `text`.

Then it should find the token boundary where the target begins.

Recommended approach:

1. Store `target_start_marker`.
2. Store `target_text`.
3. During preprocessing, construct:

```python
prompt_text = text_without_target
full_text = prompt_text + target_text
```

4. Tokenize `prompt_text`.
5. Tokenize `full_text`.
6. Set labels to `-100` for the length of `prompt_text` tokens.
7. Set labels to input ids for target tokens.
8. Set pad labels to `-100`.

Pseudo-code:

```python
prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

assert len(labels) == len(full_ids)
```

Important: with BPE tokenizers, tokenization of `prompt_text + target_text` may differ slightly from tokenizing them separately. Prefer tokenizing `prompt_text` and `full_text`, then use the length of `prompt_ids` as the masking boundary.

---

# 15. Training order

Recommended training sequence:

## Step 1: Train memory summarizer

Train on:

```text
data/processed/msc_memory_summary/train.jsonl
```

Objective:

```text
dialogue history -> compact memory
```

This can be a separate model or the same base model fine-tuned with a task token.

## Step 2: Generate memory for response examples

Use one of two approaches:

### Approach 1: use gold memory

If `msc:PersonaSummary` provides suitable gold summaries, use them as memory.

Pros:

- cleaner
- easier
- stable
- good for first experiments

Cons:

- may not match the memory produced by your own summarizer

### Approach 2: use predicted memory

Run the trained Stage A model over previous sessions and produce memory summaries.

Pros:

- closer to production behavior
- tests real pipeline

Cons:

- errors from the summarizer can affect response training

Recommended start:

```text
First use gold memory.
Then try predicted memory.
Then compare.
```

## Step 3: Train response model

Train on:

```text
data/processed/msc_response_sft/train.jsonl
```

Objective:

```text
memory + current_dialogue -> assistant response
```

---

# 16. Evaluation ideas

Evaluate Stage A and Stage B separately.

## Stage A evaluation

Check whether memory summaries are:

- factual
- compact
- useful
- not hallucinated
- not copied as full dialogue
- stable across sessions

Automatic metrics can include ROUGE/BLEU, but manual inspection is important.

## Stage B evaluation

Check whether responses:

- use memory when relevant
- ignore memory when irrelevant
- do not invent unsupported personal facts
- answer the current user message naturally
- do not overuse memory in every response

Create a small manual evaluation set:

```json
{
  "memory": ["The user likes hiking."],
  "dialogue": [{"role": "user", "content": "What hobby did I mention before?"}],
  "expected_behavior": "Assistant should mention hiking."
}
```

Also include negative examples:

```json
{
  "memory": ["The user likes hiking."],
  "dialogue": [{"role": "user", "content": "Can you explain transformers?"}],
  "expected_behavior": "Assistant should answer about transformers, not force hiking into the answer."
}
```

---

# 17. Production-like runtime pipeline

The final runtime behavior should look like this:

```text
1. User sends a new message.
2. Load stored long-term memory for this user.
3. Load recent dialogue context.
4. Format prompt:

   <|memory|>
   saved user facts

   <|dialog|>
   recent turns

   <|assistant|>

5. Generate assistant response.
6. After the session, run memory summarizer:
   previous memory + new dialogue -> updated memory
7. Store updated memory.
```

For the first implementation, memory can be a simple list of strings.

Later, memory can be stored in a vector database or SQLite table with embeddings.

---

# 18. Recommended memory object schema

Use a structured memory format internally, even if the model sees plain text.

```json
{
  "user_id": "local_user",
  "facts": [
    {
      "id": "fact_001",
      "text": "The user likes hiking.",
      "source": "msc_session_001",
      "confidence": 1.0,
      "created_at": "2026-05-15T00:00:00Z",
      "updated_at": "2026-05-15T00:00:00Z"
    }
  ]
}
```

Render this as model text:

```text
<|memory|>
The user likes hiking.
The user likes old movies.
```

Do not expose implementation metadata to the model unless needed.

---

# 19. Important implementation constraints

## Keep raw and processed data separate

Never overwrite raw exports.

```text
data/raw/msc/*.jsonl
data/processed/msc_*/*.jsonl
```

## Make scripts reproducible

Every processing script should accept:

```bash
--input
--output
--split
--max-examples
--seed
```

## Save formatting version

Add a `format_version` field to processed examples.

```json
{
  "format_version": "msc_memory_v1"
}
```

This helps avoid mixing incompatible data formats.

## Log skipped examples

If an example cannot be parsed, write it to:

```text
data/processed/logs/skipped_examples.jsonl
```

Each skipped record should include:

```json
{
  "reason": "missing_label",
  "raw_example": {}
}
```

---

# 20. Minimal implementation checklist

## Export

- [ ] Create `scripts/export_msc_from_parlai.py`
- [ ] Export `msc train`
- [ ] Export `msc valid`
- [ ] Export `msc test`
- [ ] Export `msc:PersonaSummary train`
- [ ] Export `msc:PersonaSummary valid`

## Processing

- [ ] Create `scripts/build_memory_summary_dataset.py`
- [ ] Create `scripts/build_response_sft_dataset.py`
- [ ] Define shared schemas in `src/data/schemas.py`
- [ ] Define text formatting in `src/data/formatting.py`
- [ ] Add skipped-example logging

## Tokenization

- [ ] Add special tokens if using a custom tokenizer
- [ ] Implement last-target-only masking
- [ ] Mask prompt tokens with `-100`
- [ ] Mask padding tokens with `-100`
- [ ] Verify that target tokens are not accidentally masked

## Training

- [ ] Train Stage A memory summarizer
- [ ] Generate or collect memory summaries
- [ ] Train Stage B response model
- [ ] Compare gold memory vs predicted memory

## Evaluation

- [ ] Manual examples where memory is required
- [ ] Manual examples where memory is irrelevant
- [ ] Check hallucinated memory usage
- [ ] Check response naturalness

---

# 21. Suggested first experiment

Start with a small subset.

```bash
python scripts/export_msc_from_parlai.py \
  --task msc \
  --split train \
  --out data/raw/msc/train.jsonl \
  --max-examples 1000
```

```bash
python scripts/export_msc_from_parlai.py \
  --task msc:PersonaSummary \
  --split train \
  --out data/raw/msc/persona_summary_train.jsonl \
  --max-examples 1000
```

Then build processed data:

```bash
python scripts/build_memory_summary_dataset.py \
  --input data/raw/msc/persona_summary_train.jsonl \
  --output data/processed/msc_memory_summary/train.jsonl
```

```bash
python scripts/build_response_sft_dataset.py \
  --input data/raw/msc/train.jsonl \
  --memory data/processed/msc_memory_summary/train.jsonl \
  --output data/processed/msc_response_sft/train.jsonl
```

Train first on a small model and inspect generated examples before running a full training job.

---

# 22. Expected final model behavior

The final chat model should be able to do this:

```text
<|memory|>
The user likes hiking.
The user likes old movies.

<|dialog|>
<|user|> What did I tell you I like doing outdoors?

<|assistant|>
You told me that you like hiking.
```

It should not require the full old conversation to answer this.

The memory summary acts as compressed long-term context.

---

# 23. Why this design is useful for small models

Small models usually struggle with:

- very long dialogue history
- noisy previous sessions
- remembering facts across turns
- distinguishing important facts from casual text

The two-stage design reduces the task difficulty.

Instead of training one model to do everything from raw long context, the system learns:

```text
Stage A: extract useful long-term memory
Stage B: use compact memory during conversation
```

This is easier to debug and easier to improve.

It also matches a practical production architecture where long-term memory is stored outside the model.
