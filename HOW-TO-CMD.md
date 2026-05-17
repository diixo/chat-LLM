# HOW-TO-CMD

Короткий runbook для `c:\git\chat-LLM`.

## 1. Проверить special tokens

```powershell
python scripts/verify_special_tokens.py --model-name-or-path gpt2
```

## 2. Собрать raw JSONL

### Dialogue

Вход:

- `data/msc_v0.1/msc/msc_dialogue/session_*/train.txt`
- `data/msc_v0.1/msc/msc_dialogue/session_*/valid.txt`

Выход:

- `datasets/raw/msc/train.jsonl`
- `datasets/raw/msc/valid.jsonl`

```powershell
python scripts/export_msc_dialogue.py --split train --output datasets/raw/msc/train.jsonl
python scripts/export_msc_dialogue.py --split valid --output datasets/raw/msc/valid.jsonl
```

### Persona summary

Вход:

- `data/msc_v0.1/msc/msc_personasummary/session_*/train.txt`
- `data/msc_v0.1/msc/msc_personasummary/session_*/valid.txt`

Выход:

- `datasets/raw/msc/persona_summary_train.jsonl`
- `datasets/raw/msc/persona_summary_valid.jsonl`

```powershell
python scripts/export_msc_personasummary.py --split train --output datasets/raw/msc/persona_summary_train.jsonl
python scripts/export_msc_personasummary.py --split valid --output datasets/raw/msc/persona_summary_valid.jsonl
```

## 3. Собрать processed Stage A

Вход:

- `datasets/raw/msc/persona_summary_train.jsonl`
- `datasets/raw/msc/persona_summary_valid.jsonl`

Выход:

- `datasets/processed/msc_memory_summary/train.jsonl`
- `datasets/processed/msc_memory_summary/valid.jsonl`

```powershell
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_train.jsonl --output datasets/processed/msc_memory_summary/train.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_valid.jsonl --output datasets/processed/msc_memory_summary/valid.jsonl
```

## 4. Собрать processed Stage B

Вход:

- `data/convai2/train_self_original_no_cands.txt`
- `data/convai2/valid_self_original_no_cands.txt`
- `datasets/raw/msc/train.jsonl`
- `datasets/raw/msc/valid.jsonl`
- `datasets/processed/msc_memory_summary/train.jsonl`
- `datasets/processed/msc_memory_summary/valid.jsonl`

Выход:

- `datasets/processed/msc_response_sft/train.jsonl`
- `datasets/processed/msc_response_sft/valid.jsonl`

```powershell
python scripts/build_response_sft_dataset.py --session1-input data/convai2/train_self_original_no_cands.txt --input datasets/raw/msc/train.jsonl --memory datasets/processed/msc_memory_summary/train.jsonl --require-memory --output datasets/processed/msc_response_sft/train.jsonl
python scripts/build_response_sft_dataset.py --session1-input data/convai2/valid_self_original_no_cands.txt --input datasets/raw/msc/valid.jsonl --memory datasets/processed/msc_memory_summary/valid.jsonl --require-memory --output datasets/processed/msc_response_sft/valid.jsonl
```

Примечание:

- `--session1-input` закрывает `session_1` из raw ConvAI2 `self_original_no_cands`.
- `--input` даёт `session_2+` из local MSC dialogue.
- `--memory` нужен для gold memory на `session_2+`, который строится из local MSC persona-summary.

## 5. Обучить Stage A

Вход:

- `datasets/processed/msc_memory_summary/train.jsonl`
- `datasets/processed/msc_memory_summary/valid.jsonl`

Выход:

- `runs/stage_a_memory_summary`

```powershell
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_memory_summary/train.jsonl --valid-file datasets/processed/msc_memory_summary/valid.jsonl --output-dir runs/stage_a_memory_summary
```

## 6. Обучить Stage B

Вход:

- `datasets/processed/msc_response_sft/train.jsonl`
- `datasets/processed/msc_response_sft/valid.jsonl`

Выход:

- `runs/stage_b_response`

```powershell
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_response_sft/train.jsonl --valid-file datasets/processed/msc_response_sft/valid.jsonl --output-dir runs/stage_b_response
```

## 7. Полный порядок основных команд подряд

```powershell
python scripts/verify_special_tokens.py --model-name-or-path gpt2

python scripts/export_msc_dialogue.py --split train --output datasets/raw/msc/train.jsonl
python scripts/export_msc_dialogue.py --split valid --output datasets/raw/msc/valid.jsonl

python scripts/export_msc_personasummary.py --split train --output datasets/raw/msc/persona_summary_train.jsonl
python scripts/export_msc_personasummary.py --split valid --output datasets/raw/msc/persona_summary_valid.jsonl

python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_train.jsonl --output datasets/processed/msc_memory_summary/train.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_valid.jsonl --output datasets/processed/msc_memory_summary/valid.jsonl

python scripts/build_response_sft_dataset.py --session1-input data/convai2/train_self_original_no_cands.txt --input datasets/raw/msc/train.jsonl --memory datasets/processed/msc_memory_summary/train.jsonl --require-memory --output datasets/processed/msc_response_sft/train.jsonl
python scripts/build_response_sft_dataset.py --session1-input data/convai2/valid_self_original_no_cands.txt --input datasets/raw/msc/valid.jsonl --memory datasets/processed/msc_memory_summary/valid.jsonl --require-memory --output datasets/processed/msc_response_sft/valid.jsonl

python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_memory_summary/train.jsonl --valid-file datasets/processed/msc_memory_summary/valid.jsonl --output-dir runs/stage_a_memory_summary
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_response_sft/train.jsonl --valid-file datasets/processed/msc_response_sft/valid.jsonl --output-dir runs/stage_b_response
```

## 8. Проверочный unified valid run

```powershell
python scripts/build_response_sft_dataset.py --session1-input data/convai2/valid_self_original_no_cands.txt --input datasets/raw/msc/valid.jsonl --memory datasets/processed/msc_memory_summary/valid.jsonl --require-memory --output datasets/processed/msc_response_sft/valid_convai2_session1_check.jsonl
```