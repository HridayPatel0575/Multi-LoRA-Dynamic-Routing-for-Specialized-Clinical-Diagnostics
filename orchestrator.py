import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Qwen2.5 chat template (used when the tokenizer has none set)
_QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

ADAPTER_MAP = {
    "cardiology":          "Hriday75/qwen2.5-3b-cardio-dpo-aligned",
    "oncology":            "Hriday75/qwen2.5-3b-oncology-dpo-aligned",
    "infectious_disease":  "Hriday75/qwen2.5-3b-infectious-disease-dpo-aligned",
}

SYSTEM_PROMPT = (
    "You are a highly specialized, empathetic medical AI expert. "
    "Provide clinically accurate, well-structured responses. "
    "Always recommend consulting a qualified physician for personal medical decisions."
)

ROUTING_PROMPT = """You are a medical query classifier. Classify the query below into exactly one category.

Categories:
- cardiology   (heart, blood pressure, arrhythmia, coronary, cardiac)
- oncology     (cancer, tumor, chemotherapy, carcinoma, malignancy, biopsy)
- infectious_disease (infection, virus, bacteria, fever, antibiotic, sepsis, pathogen)

Query: "{query}"

Respond with only the category name and nothing else.
Category:"""


class MedicalOrchestrator:
    def __init__(self):
        print("Loading base model...")
        self.base_model_name = "unsloth/Qwen2.5-3B-bnb-4bit"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            load_in_4bit=True,
        )

        print("Stacking expert adapters into VRAM...")
        adapters = list(ADAPTER_MAP.items())
        # First adapter bootstraps PeftModel
        self.model = PeftModel.from_pretrained(
            base_model, adapters[0][1], adapter_name=adapters[0][0]
        )
        for name, repo in adapters[1:]:
            self.model.load_adapter(repo, adapter_name=name)

        self.session_memory: dict[str, list[dict]] = {}
        print("Orchestrator ready.\n")



    def get_session(self, session_id: str) -> list[dict]:
        if session_id not in self.session_memory:
            self.session_memory[session_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        return self.session_memory[session_id]

    def reset_session(self, session_id: str) -> None:
        self.session_memory[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]



    def route_query(self, user_query: str) -> str:
        prompt = ROUTING_PROMPT.format(query=user_query)

        with self.model.disable_adapter():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).lower().strip()

        # Priority-ordered keyword match to avoid mis-routing
        if any(k in response for k in ("cardio", "heart")):
            return "cardiology"
        if any(k in response for k in ("onco", "cancer", "tumor")):
            return "oncology"
        if any(k in response for k in ("infect", "disease")):
            return "infectious_disease"

        # Fallback: let the full history guide which expert is most relevant
        return "infectious_disease"


    def _format_prompt(self, history: list[dict]) -> torch.Tensor:
        """Formats the chat history into model input tensors.

        Falls back to the Qwen2.5 <|im_start|> template when the tokenizer
        has no chat_template attribute (e.g. unsloth checkpoints).
        """
        try:
            return self.tokenizer.apply_chat_template(
                history,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        except ValueError:
            # Tokenizer lacks a chat_template — inject Qwen2.5's template and retry
            self.tokenizer.chat_template = _QWEN_CHAT_TEMPLATE
            return self.tokenizer.apply_chat_template(
                history,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)

    def chat(self, session_id: str, user_query: str) -> tuple[str, str]:
        """
        Returns (expert_name, response_text).
        Maintains full session context across turns.
        """
        expert = self.route_query(user_query)
        self.model.set_adapter(expert)

        history = self.get_session(session_id)
        history.append({"role": "user", "content": user_query})

        inputs = self._format_prompt(history)

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()

        history.append({"role": "assistant", "content": response_text})
        return expert, response_text
