# chat-LLM

## Training Data: Fact Extraction from Dialogue

## Task Description

The goal is to fine-tune a small LLM (e.g. GPT-2) to reliably extract
explicit user facts from a conversation turn and return them as a JSON array.

### Concrete task

Given a single dialogue turn:

```
User: I am ok. I want to sleep.
Assistant: That's great to hear! ...
```

The model must output:

```json
["User is ok", "User wants to sleep"]
```


### Target behaviour

| User message | Expected output |
|---|---|
| `"I am ok. I want to sleep"` | `["User is ok", "User wants to sleep"]` |
| `"My name is Alice and I live in Berlin"` | `["User's name is Alice", "User lives in Berlin"]` |
| `"Can you help me with Python?"` | `[]` |
| `"I have a dog named Rex"` | `["User has a dog named Rex"]` |

---

## Relevant Datasets

| Dataset | Description | Link |
|---|---|---|
| **MSC (Multi-Session Chat)** | Multi-session dialogues by Meta with explicit per-persona memory annotations. Best match for our task. | https://parl.ai/projects/msc/ |
| **ConvAI2** | Single-session persona-grounded dialogues used to recover MSC session_1 by `initial_data_id`. | http://parl.ai/downloads/convai2/convai2_fix_723.tgz |
| **MemoryBank / LONGMEM** | Dialogue datasets focused on long-term user memory and recall. | https://huggingface.co/datasets/Zxnii/MemoryBank-SiliconFriend |
| **ShareGPT** | Real GPT-4 conversations. Used as base for synthetic annotation via teacher model. | https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered |
| **UltraChat** | Large-scale synthetic multi-turn dialogues. Good base for augmentation. | https://huggingface.co/datasets/stingning/ultrachat |

---

## Current Local Pipeline

The repository currently implements a local two-stage MSC/ConvAI2 pipeline rather than only a single-turn JSON extraction toy setup.

- Stage A builds `previous_dialogue_history -> memory_summary` examples from local MSC persona-summary data.
- Stage B builds `memory_summary + current_dialogue -> assistant_response` examples by combining ConvAI2 `session_1` with MSC `session_2+`.
- `session_1` is recovered from ConvAI2 by `initial_data_id`, not by text matching.
- Processed builders normalize ConvAI2 and MSC text into a common sentence-case style similar to ParlAI `normalize_reply`, so the final corpora do not mix raw lowercase ConvAI2 with case-sensitive MSC text.
- Memory facts are normalized into a consistent third-person style such as `The user wears contacts.` instead of leaving a mix like `The user is very athletic.` and `I wear contacts.`.

For the executable runbook and current build counts, see [HOW-TO.md](HOW-TO.md) and [HOW-TO-CMD.md](HOW-TO-CMD.md).

