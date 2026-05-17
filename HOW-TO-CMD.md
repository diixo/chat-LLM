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

Примечание:

- если для speaker orientation нет annotated persona items, builder берёт fallback из `metadata.init_personachat.init_personas[target_speaker_index]`;
- memory facts приводятся к единому third-person виду `The user ...`, чтобы не было смеси `The user ...` и `I ...` в processed memory;
- в summary теперь печатаются `Annotated memory hits` и `Init persona fallbacks`.

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
- текст ConvAI2 и MSC в processed JSONL приводится к общему sentence-case стилю по аналогии с ParlAI `normalize_reply`, чтобы не было микса lowercase и case-sensitive текста.
- память в итоговом processed JSONL тоже приводится к единому `The user ...` стилю, например `The user wears contacts.`.

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

## 9. Актуальные результаты сборки

- Stage A train: `20570` examples, `0 skipped`, `20466` annotated hits, `104` init persona fallbacks.
- Stage A valid: `4000` examples, `0 skipped`, `3972` annotated hits, `28` init persona fallbacks.
- Stage B train: `227984` examples, `0 skipped`, `memory hits = 18002`, `memory misses = 0`, `ConvAI2 hits = 131438`, `ConvAI2 misses = 0`.
- Stage B valid: `29456` examples, `0 skipped`, `memory hits = 4000`, `memory misses = 0`, `ConvAI2 hits = 7801`, `ConvAI2 misses = 0`.