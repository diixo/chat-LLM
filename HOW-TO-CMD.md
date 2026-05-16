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

- `datasets/raw/msc/train.jsonl`
- `datasets/raw/msc/valid.jsonl`
- `datasets/processed/msc_memory_summary/train.jsonl`
- `datasets/processed/msc_memory_summary/valid.jsonl`

Выход:

- `datasets/processed/msc_response_sft/train.jsonl`
- `datasets/processed/msc_response_sft/valid.jsonl`

```powershell
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/train.jsonl --memory datasets/processed/msc_memory_summary/train.jsonl --require-memory --output datasets/processed/msc_response_sft/train.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/valid.jsonl --memory datasets/processed/msc_memory_summary/valid.jsonl --require-memory --output datasets/processed/msc_response_sft/valid.jsonl
```

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

## 7. Полный порядок команд подряд

```powershell
python scripts/verify_special_tokens.py --model-name-or-path gpt2
python scripts/export_msc_dialogue.py --split train --output datasets/raw/msc/train.jsonl
python scripts/export_msc_dialogue.py --split valid --output datasets/raw/msc/valid.jsonl
python scripts/export_msc_personasummary.py --split train --output datasets/raw/msc/persona_summary_train.jsonl
python scripts/export_msc_personasummary.py --split valid --output datasets/raw/msc/persona_summary_valid.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_train.jsonl --output datasets/processed/msc_memory_summary/train.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_valid.jsonl --output datasets/processed/msc_memory_summary/valid.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/train.jsonl --memory datasets/processed/msc_memory_summary/train.jsonl --require-memory --output datasets/processed/msc_response_sft/train.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/valid.jsonl --memory datasets/processed/msc_memory_summary/valid.jsonl --require-memory --output datasets/processed/msc_response_sft/valid.jsonl
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_memory_summary/train.jsonl --valid-file datasets/processed/msc_memory_summary/valid.jsonl --output-dir runs/stage_a_memory_summary
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_response_sft/train.jsonl --valid-file datasets/processed/msc_response_sft/valid.jsonl --output-dir runs/stage_b_response
```

## 8. Узкие тесты на маленьком подмножестве

```powershell
python scripts/export_msc_personasummary.py --split train --session 1 --max-examples 100 --output datasets/raw/msc/persona_summary_session_1_sample.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_session_1_sample.jsonl --output datasets/processed/msc_memory_summary/session_1_sample.jsonl
python scripts/export_msc_dialogue.py --split train --session 2 --max-examples 100 --output datasets/raw/msc/dialogue_session_2_sample.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/dialogue_session_2_sample.jsonl --memory datasets/processed/msc_memory_summary/session_1_sample.jsonl --require-memory --output datasets/processed/msc_response_sft/session_2_sample.jsonl
```