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
| **PersonaChat** | Single-session persona-grounded dialogues. Persona facts are the ground-truth target for extraction. | https://huggingface.co/datasets/bavard/personachat_truecased or http://parl.ai/downloads/personachat/personachat.tgz |
| **MemoryBank / LONGMEM** | Dialogue datasets focused on long-term user memory and recall. | https://huggingface.co/datasets/Zxnii/MemoryBank-SiliconFriend |
| **ShareGPT** | Real GPT-4 conversations. Used as base for synthetic annotation via teacher model. | https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered |
| **UltraChat** | Large-scale synthetic multi-turn dialogues. Good base for augmentation. | https://huggingface.co/datasets/stingning/ultrachat |

---

