# HOW-TO

## Что это за пайплайн

В этом репозитории пайплайн состоит из двух стадий:

- Stage A: `previous_dialogue_history -> memory_summary`
- Stage B: `memory_summary + current_dialogue -> assistant_response`

Правило путей для этого репозитория:

- `data/` = исходные данные, которые можно хранить в git
- `datasets/` = генерируемые raw/processed артефакты пайплайна

Общий поток такой:

```text
локальный MSC dataset
    ->
raw export scripts
    ->
raw JSONL
    ->
processed builders
    ->
processed JSONL
    ->
training_dataset / collator
    ->
train_sft
    ->
trained checkpoints + tokenizer
```

По файлам это выглядит так:

1. Источник данных:
   - `data/msc_v0.1/msc/msc_dialogue`
   - `data/msc_v0.1/msc/msc_personasummary`
2. Экспорт в единый raw-формат:
   - `scripts/export_msc_dialogue.py`
   - `scripts/export_msc_personasummary.py`
3. Построение processed datasets:
   - `scripts/build_memory_summary_dataset.py`
   - `scripts/build_response_sft_dataset.py`
4. Подготовка примера для модели:
   - `src/data/training_dataset.py`
5. Спецтокены пайплайна:
   - `src/data/special_tokens.py`
   - `scripts/verify_special_tokens.py`
6. Обучение:
   - `scripts/train_sft.py`

## Что уже сделано в репозитории

- Добавлены local raw exporters:
  - `scripts/export_msc_dialogue.py`
  - `scripts/export_msc_personasummary.py`
- Добавлены processed dataset builders:
  - `scripts/build_memory_summary_dataset.py`
  - `scripts/build_response_sft_dataset.py`
- Добавлены shared helpers для JSONL parsing, speaker orientation и memory-fact normalization:
  - `scripts/msc_processed_utils.py`
- Добавлен processed dataset loader и collator с masking по `target_start_marker`:
  - `src/data/training_dataset.py`
- Добавлен модуль и скрипт для special tokens:
  - `src/data/special_tokens.py`
  - `scripts/verify_special_tokens.py`
- Добавлен training entrypoint:
  - `scripts/train_sft.py`

## Что делают training_dataset.py и train_sft.py

### `src/data/training_dataset.py`

Это слой подготовки данных для модели. Он:

- читает processed JSONL;
- превращает одну запись в `input_ids`, `attention_mask` и `labels`;
- вычисляет границу `prompt/target`;
- маскирует всё до target значением `-100`;
- паддит батч через `ProcessedSFTCollator`.

То есть он отвечает за то, что именно модель увидит на входе и по каким токенам будет считаться loss.

### `scripts/train_sft.py`

Это слой запуска обучения. Он:

- загружает реальный tokenizer и model из Transformers;
- регистрирует и проверяет спецтокены;
- при необходимости расширяет эмбеддинги;
- создаёт `ProcessedSFTDataset` и `ProcessedSFTCollator`;
- собирает `TrainingArguments`;
- запускает `Trainer.train()`;
- сохраняет модель, tokenizer и state.

То есть он отвечает не за разметку примера, а за сам процесс fine-tuning.

Коротко:

- `src/data/training_dataset.py` = как сформировать обучающий пример.
- `scripts/train_sft.py` = как обучать модель на этих примерах.

## Специальные токены пайплайна

Зафиксированы такие токены:

```text
<|user|>
<|assistant|>
<|system|>
<|memory|>
<|dialog|>
<|task|>
```

Они объявлены в `src/data/special_tokens.py`.

Проверка tokenizer делается так:

```powershell
python scripts/verify_special_tokens.py --model-name-or-path gpt2
```

Что делает эта команда:

- загружает tokenizer;
- регистрирует pipeline special tokens;
- проверяет, что они стабильно токенизируются;
- при необходимости может сохранить tokenizer и список токенов.

Проверка на `gpt2` уже проходила успешно: для всех шести токенов был подтверждён single-token encoding, exact round-trip и стабильность в контексте.

## Пошаговый запуск для этого репозитория

Ниже приведён рекомендуемый порядок запуска именно для текущей структуры `c:\git\chat-LLM`.
Исходный MSC остаётся в `data/`, а все генерируемые JSONL-файлы пишутся в `datasets/`.

### Шаг 1. Экспорт dialogue dataset в raw JSONL

Скрипт:

- `scripts/export_msc_dialogue.py`

Вход:

- `data/msc_v0.1/msc/msc_dialogue/session_*/train.txt`
- `data/msc_v0.1/msc/msc_dialogue/session_*/valid.txt`

Выход:

- `datasets/raw/msc/train.jsonl`
- `datasets/raw/msc/valid.jsonl`

Команды:

```powershell
python scripts/export_msc_dialogue.py --split train --output datasets/raw/msc/train.jsonl
python scripts/export_msc_dialogue.py --split valid --output datasets/raw/msc/valid.jsonl
```

Полезные параметры:

- `--input-root data/msc_v0.1/msc/msc_dialogue`
- `--session 2`
- `--session session_3`
- `--max-examples 1000`

Пример узкого теста:

```powershell
python scripts/export_msc_dialogue.py --split train --session 2 --max-examples 100 --output datasets/raw/msc/dialogue_session_2_sample.jsonl
```

### Шаг 2. Экспорт persona summary dataset в raw JSONL

Скрипт:

- `scripts/export_msc_personasummary.py`

Вход:

- `data/msc_v0.1/msc/msc_personasummary/session_*/train.txt`
- `data/msc_v0.1/msc/msc_personasummary/session_*/valid.txt`

Выход:

- `datasets/raw/msc/persona_summary_train.jsonl`
- `datasets/raw/msc/persona_summary_valid.jsonl`

Команды:

```powershell
python scripts/export_msc_personasummary.py --split train --output datasets/raw/msc/persona_summary_train.jsonl
python scripts/export_msc_personasummary.py --split valid --output datasets/raw/msc/persona_summary_valid.jsonl
```

Полезные параметры:

- `--input-root data/msc_v0.1/msc/msc_personasummary`
- `--session 1`
- `--session session_2`
- `--max-examples 1000`

Пример узкого теста:

```powershell
python scripts/export_msc_personasummary.py --split train --session 1 --max-examples 100 --output datasets/raw/msc/persona_summary_session_1_sample.jsonl
```

### Шаг 3. Построить Stage A dataset: memory summary

Скрипт:

- `scripts/build_memory_summary_dataset.py`

Вход:

- `datasets/raw/msc/persona_summary_train.jsonl`
- `datasets/raw/msc/persona_summary_valid.jsonl`

Выход:

- `datasets/processed/msc_memory_summary/train.jsonl`
- `datasets/processed/msc_memory_summary/valid.jsonl`

Команды:

```powershell
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_train.jsonl --output datasets/processed/msc_memory_summary/train.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_valid.jsonl --output datasets/processed/msc_memory_summary/valid.jsonl
```

Полезные параметры:

- `--max-examples 1000`

Что делает этот шаг:

- берёт raw persona summary JSONL;
- строит Stage A examples вида `dialogue history -> memory`;
- формирует поля `text`, `target_start_marker`, `target_text`;
- пишет processed JSONL в формате, пригодном для training_dataset.

### Шаг 4. Построить Stage B dataset: response SFT

Скрипт:

- `scripts/build_response_sft_dataset.py`

Вход:

- `datasets/raw/msc/train.jsonl`
- `datasets/raw/msc/valid.jsonl`
- `datasets/processed/msc_memory_summary/train.jsonl`
- `datasets/processed/msc_memory_summary/valid.jsonl`

Выход:

- `datasets/processed/msc_response_sft/train.jsonl`
- `datasets/processed/msc_response_sft/valid.jsonl`

Команды:

```powershell
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/train.jsonl --memory datasets/processed/msc_memory_summary/train.jsonl --require-memory --output datasets/processed/msc_response_sft/train.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/valid.jsonl --memory datasets/processed/msc_memory_summary/valid.jsonl --require-memory --output datasets/processed/msc_response_sft/valid.jsonl
```

Полезные параметры:

- `--memory datasets/processed/msc_memory_summary/train.jsonl`
- `--require-memory`
- `--max-examples 1000`

Что делает этот шаг:

- берёт raw dialogue examples;
- подтягивает gold memory из Stage A;
- строит примеры вида `memory + current_dialogue -> assistant_response`;
- формирует поля `text`, `target_start_marker`, `target_text`.

Важно:

- локальная связка gold memory идёт со сдвигом по сессии:
  - `msc_personasummary/session_N` summarises memory для `msc_dialogue/session_{N+1}`
- `--require-memory` отбрасывает примеры, для которых не найдено соответствующее memory.

## Полный build train/valid

Если нужно полностью собрать все train/valid данные, запускай так:

```powershell
python scripts/export_msc_dialogue.py --split train --output datasets/raw/msc/train.jsonl
python scripts/export_msc_dialogue.py --split valid --output datasets/raw/msc/valid.jsonl
python scripts/export_msc_personasummary.py --split train --output datasets/raw/msc/persona_summary_train.jsonl
python scripts/export_msc_personasummary.py --split valid --output datasets/raw/msc/persona_summary_valid.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_train.jsonl --output datasets/processed/msc_memory_summary/train.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_valid.jsonl --output datasets/processed/msc_memory_summary/valid.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/train.jsonl --memory datasets/processed/msc_memory_summary/train.jsonl --require-memory --output datasets/processed/msc_response_sft/train.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/valid.jsonl --memory datasets/processed/msc_memory_summary/valid.jsonl --require-memory --output datasets/processed/msc_response_sft/valid.jsonl
```

## Уже полученные размеры датасетов

На текущий момент при полном прогоне были получены такие числа:

- Raw dialogue export train: 9001 examples
- Raw dialogue export valid: 2000 examples
- Raw persona summary export train: 10285 examples
- Raw persona summary export valid: 2000 examples
- Processed memory summary train: 20466 examples, skipped 104
- Processed memory summary valid: 3972 examples, skipped 28
- Processed response SFT train: 96067 examples, memory hits 17913, memory misses 89
- Processed response SFT valid: 21505 examples, memory hits 3972, memory misses 28

Также было проверено, что в итоговых `response_sft` train/valid файлах нет пустого `input.memory`, если сборка делается с `--require-memory`.

## Как использовать training_dataset.py в обучении

`src/data/training_dataset.py` используется не как отдельная CLI-команда, а как библиотека внутри `scripts/train_sft.py`.

Он делает следующее:

- загружает processed JSONL;
- строит `prompt_text` и `full_text`;
- токенизирует оба текста;
- ставит `-100` на всё до начала target;
- создаёт батчи с padded `input_ids`, `attention_mask`, `labels`.

Именно здесь реализована логика masking, а не в export/build scripts.

## Обучение Stage A

Скрипт:

- `scripts/train_sft.py`

Вход:

- `datasets/processed/msc_memory_summary/train.jsonl`
- `datasets/processed/msc_memory_summary/valid.jsonl`
- имя базовой модели Hugging Face

Выход:

- директория с чекпоинтами, моделью и tokenizer, например:
  - `runs/stage_a_memory_summary`

Пример команды:

```powershell
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_memory_summary/train.jsonl --valid-file datasets/processed/msc_memory_summary/valid.jsonl --output-dir runs/stage_a_memory_summary
```

Часто используемые параметры:

- `--max-length 2048`
- `--learning-rate 2e-5`
- `--num-train-epochs 1`
- `--per-device-train-batch-size 1`
- `--per-device-eval-batch-size 1`
- `--gradient-accumulation-steps 16`
- `--warmup-ratio 0.03`
- `--logging-steps 10`
- `--save-steps 500`
- `--eval-steps 500`
- `--save-total-limit 2`
- `--gradient-checkpointing`
- `--bf16`
- `--fp16`
- `--trust-remote-code`
- `--overwrite-output-dir`

Что делает команда:

- загружает tokenizer и model;
- регистрирует pipeline special tokens;
- проверяет стабильность токенов;
- при необходимости расширяет эмбеддинги модели;
- запускает `Trainer.train()` на Stage A dataset;
- сохраняет модель, tokenizer и `pipeline_special_tokens.json`.

## Обучение Stage B

Скрипт:

- `scripts/train_sft.py`

Вход:

- `datasets/processed/msc_response_sft/train.jsonl`
- `datasets/processed/msc_response_sft/valid.jsonl`
- имя базовой модели Hugging Face

Выход:

- директория с чекпоинтами, моделью и tokenizer, например:
  - `runs/stage_b_response`

Пример команды:

```powershell
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_response_sft/train.jsonl --valid-file datasets/processed/msc_response_sft/valid.jsonl --output-dir runs/stage_b_response
```

## Быстрая схема по стадиям

### Stage A

```text
msc_personasummary txt
-> export_msc_personasummary.py
-> datasets/raw/msc/persona_summary_*.jsonl
-> build_memory_summary_dataset.py
-> datasets/processed/msc_memory_summary/*.jsonl
-> training_dataset.py
-> train_sft.py
-> memory summarizer model
```

### Stage B

```text
msc_dialogue txt
-> export_msc_dialogue.py
-> datasets/raw/msc/*.jsonl
+ gold memory from datasets/processed/msc_memory_summary/*.jsonl
-> build_response_sft_dataset.py
-> datasets/processed/msc_response_sft/*.jsonl
-> training_dataset.py
-> train_sft.py
-> response model
```

## Минимальный рекомендуемый порядок работы

Если запускать всё с нуля, практический порядок такой:

1. Проверить special tokens:

```powershell
python scripts/verify_special_tokens.py --model-name-or-path gpt2
```

2. Собрать raw файлы:

```powershell
python scripts/export_msc_dialogue.py --split train --output datasets/raw/msc/train.jsonl
python scripts/export_msc_dialogue.py --split valid --output datasets/raw/msc/valid.jsonl
python scripts/export_msc_personasummary.py --split train --output datasets/raw/msc/persona_summary_train.jsonl
python scripts/export_msc_personasummary.py --split valid --output datasets/raw/msc/persona_summary_valid.jsonl
```

3. Собрать processed Stage A и Stage B:

```powershell
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_train.jsonl --output datasets/processed/msc_memory_summary/train.jsonl
python scripts/build_memory_summary_dataset.py --input datasets/raw/msc/persona_summary_valid.jsonl --output datasets/processed/msc_memory_summary/valid.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/train.jsonl --memory datasets/processed/msc_memory_summary/train.jsonl --require-memory --output datasets/processed/msc_response_sft/train.jsonl
python scripts/build_response_sft_dataset.py --input datasets/raw/msc/valid.jsonl --memory datasets/processed/msc_memory_summary/valid.jsonl --require-memory --output datasets/processed/msc_response_sft/valid.jsonl
```

4. Обучить Stage A:

```powershell
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_memory_summary/train.jsonl --valid-file datasets/processed/msc_memory_summary/valid.jsonl --output-dir runs/stage_a_memory_summary
```

5. Обучить Stage B:

```powershell
python scripts/train_sft.py --model-name-or-path gpt2 --train-file datasets/processed/msc_response_sft/train.jsonl --valid-file datasets/processed/msc_response_sft/valid.jsonl --output-dir runs/stage_b_response
```