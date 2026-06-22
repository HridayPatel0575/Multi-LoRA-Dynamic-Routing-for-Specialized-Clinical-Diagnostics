# Multi-LoRA Dynamic Routing for Specialized Clinical Diagnostics

This repository contains the architecture and model weights for a highly specialized, locally fine-tuned Medical AI system. It utilizes a **Multi-LoRA Dynamic Routing** approach, built on top of the `Qwen2.5-3B` base model, to act as a highly accurate, multi-disciplinary medical assistant.

Rather than relying on a single generalized model, this system dynamically routes clinical queries to specialized domain experts (Cardiology, Oncology, or Infectious Disease) depending on the patient's context.

## Architecture & 3-Phase Training Pipeline

Each domain expert was trained through a rigorous 3-phase curriculum:

1. **Phase 1: Domain Knowledge (CPT)** 
   - *Continued Pre-Training* on dense, unstructured scientific literature (PubMed, clinical notes) to embed deep domain-specific terminology and biomedical mechanics.
2. **Phase 2: Instruction Tuning (SFT)** 
   - *Supervised Fine-Tuning* on structured medical chat datasets (ChatML format) to translate Phase 1 scientific knowledge into empathetic, conversational bedside manner.
3. **Phase 3: Preference Alignment (DPO)** 
   - *Direct Preference Optimization* using clinical formatting datasets (like clinical-notes-to-fhir) to penalize hallucinations and ensure structurally sound, safe medical outputs.

---

## Hugging Face Model Registry

All 9 specialized LoRA adapters are hosted on Hugging Face. You can explore the model cards and access the weights below:

### Cardiology Experts
| Phase | Training Stage | Hugging Face Repository |
| :--- | :--- | :--- |
| **1** | Domain Knowledge | [Hriday75/qwen2.5-3b-cardiology-lora](https://huggingface.co/Hriday75/qwen2.5-3b-cardiology-lora) |
| **2** | Conversational Chat | [Hriday75/qwen2.5-3b-cardio-chat](https://huggingface.co/Hriday75/qwen2.5-3b-cardio-chat) |
| **3** | DPO Aligned (Final) | [Hriday75/qwen2.5-3b-cardio-dpo-aligned](https://huggingface.co/Hriday75/qwen2.5-3b-cardio-dpo-aligned) |

### Oncology Experts
| Phase | Training Stage | Hugging Face Repository |
| :--- | :--- | :--- |
| **1** | Domain Knowledge | [Hriday75/qwen2.5-3b-oncology-lora](https://huggingface.co/Hriday75/qwen2.5-3b-oncology-lora) |
| **2** | Conversational Chat | [Hriday75/qwen2.5-3b-oncology-chat](https://huggingface.co/Hriday75/qwen2.5-3b-oncology-chat) |
| **3** | DPO Aligned (Final) | [Hriday75/qwen2.5-3b-oncology-dpo-aligned](https://huggingface.co/Hriday75/qwen2.5-3b-oncology-dpo-aligned) |

### Infectious Disease Experts
| Phase | Training Stage | Hugging Face Repository |
| :--- | :--- | :--- |
| **1** | Domain Knowledge | [Hriday75/qwen2.5-3b-infectious-disease-lora](https://huggingface.co/Hriday75/qwen2.5-3b-infectious-disease-lora) |
| **2** | Conversational Chat | [Hriday75/qwen2.5-3b-infectious-disease-chat](https://huggingface.co/Hriday75/qwen2.5-3b-infectious-disease-chat) |
| **3** | DPO Aligned (Final) | [Hriday75/qwen2.5-3b-infectious-disease-dpo-aligned](https://huggingface.co/Hriday75/qwen2.5-3b-infectious-disease-dpo-aligned) |

---

## Quick Start

To use any of these specialized adapters, you need the base model (`unsloth/Qwen2.5-3B-bnb-4bit`) and the `peft` library. 

Here is how to dynamically load a specific expert (e.g., the fully aligned Cardiology model):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the underlying base model
base_model_name = "unsloth/Qwen2.5-3B-bnb-4bit"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 2. Attach the specific domain adapter (Example: Cardiology DPO Phase 3)
adapter_name = "Hriday75/qwen2.5-3b-cardio-dpo-aligned"
expert_model = PeftModel.from_pretrained(base_model, adapter_name)

# 3. You can now run inference with `expert_model.generate(...)`
