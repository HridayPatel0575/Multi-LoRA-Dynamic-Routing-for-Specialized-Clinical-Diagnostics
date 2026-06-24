"""
demonstration.py
================
Runs a single cardiology query through all four training stages
and prints the responses side-by-side for comparison.

Stages
------
  1. Base   — raw Qwen2.5-3B, no adapters
  2. LoRA   — Phase 1: continual pre-training adapter (medical science)
  3. SFT    — Phase 2: chat / bedside manner adapter
  4. DPO    — Phase 3: preference-aligned, safe response adapter

Usage
-----
  python demonstration.py
  python demonstration.py --query "Explain ST-elevation myocardial infarction"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich.table import Table

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

BASE_MODEL_ID = "unsloth/Qwen2.5-3B-bnb-4bit"

CARDIOLOGY_STAGES = [
    {
        "label":  "1 · BASE",
        "tag":    "base",
        "repo":   None,                                  # no adapter
        "color":  "yellow",
        "icon":   "",
    },
    {
        "label":  "2 · LoRA (CLM)",
        "tag":    "lora",
        "repo":   "Hriday75/qwen2.5-3b-cardiology-lora",
        "color":  "cyan",
        "icon":   "",
    },
    {
        "label":  "3 · SFT (Chat)",
        "tag":    "sft",
        "repo":   "Hriday75/qwen2.5-3b-cardio-chat",
        "color":  "green",
        "icon":   "",
    },
    {
        "label":  "4 · DPO (Aligned)",
        "tag":    "dpo",
        "repo":   "Hriday75/qwen2.5-3b-cardio-dpo-aligned",
        "color":  "magenta",
        "icon":   "",
    },
]

SYSTEM_PROMPT = (
    "You are a specialized cardiology AI assistant. "
    "Provide clinically accurate and well-structured responses."
)

# Qwen2.5 fallback template for unsloth checkpoints that lack chat_template
_QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

GENERATION_KWARGS = dict(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.3,
    repetition_penalty=1.1,
)

console = Console(width=120)


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        load_in_4bit=True,
    )
    model.eval()
    return tokenizer, model


def format_prompt(tokenizer, query: str, device: str) -> torch.Tensor:
    history = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": query},
    ]
    try:
        return tokenizer.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    except ValueError:
        tokenizer.chat_template = _QWEN_CHAT_TEMPLATE
        return tokenizer.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)


def generate(model, tokenizer, inputs: torch.Tensor) -> str:
    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            pad_token_id=tokenizer.eos_token_id,
            **GENERATION_KWARGS,
        )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def response_panel(stage: dict, text: str) -> Panel:
    return Panel(
        text,
        title=f"[bold {stage['color']}]{stage['icon']}  {stage['label']}[/]",
        border_style=stage["color"],
        padding=(1, 2),
        expand=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-stage cardiology demo")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="I was just diagnosed with invasive ductal carcinoma of the breast. "
                "Could this affect my heart, and what cardiology monitoring should I expect?",
        help="The cardiology query to run through all four models.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    console.print()
    console.print(Panel(
        Text("  Multi-Stage Cardiology Model Comparison  \n"
             "  Base  →  LoRA (CLM)  →  SFT (Chat)  →  DPO (Aligned)  ",
             justify="center", style="bold white"),
        border_style="bright_white",
        padding=(1, 6),
    ))
    console.print(Panel(
        f"[bold]Query:[/]  {args.query}",
        border_style="dim white",
        padding=(0, 2),
    ))
    console.print()


    with console.status("[cyan]Loading base model into VRAM…[/]", spinner="dots"):
        tokenizer, base_model = load_base_model()
    console.print("[green]✓ Base model loaded.[/]\n")

    responses: list[tuple[dict, str]] = []

    for stage in CARDIOLOGY_STAGES:
        console.print(Rule(
            f"[bold {stage['color']}]{stage['icon']}  Stage: {stage['label']}[/]",
            style=stage["color"],
        ))

        # Wrap base_model with the adapter (or use it bare for the base stage)
        if stage["repo"] is None:
            active_model = base_model
            console.print("[dim]Using raw base model — no adapter.[/]")
        else:
            with console.status(f"[{stage['color']}]Loading adapter: {stage['repo']}…[/]", spinner="dots"):
                active_model = PeftModel.from_pretrained(
                    base_model, stage["repo"], adapter_name=stage["tag"]
                )
                active_model.set_adapter(stage["tag"])
            console.print(f"[{stage['color']}]✓ Adapter loaded: {stage['repo']}[/]")

        with console.status(f"[{stage['color']}]Generating response…[/]", spinner="dots"):
            inputs = format_prompt(tokenizer, args.query, device)
            text = generate(active_model, tokenizer, inputs)

        responses.append((stage, text))
        console.print(response_panel(stage, text))
        console.print()

        # Unload the adapter after each stage to free VRAM before the next load
        if stage["repo"] is not None:
            active_model.delete_adapter(stage["tag"])
            del active_model

    console.print(Rule("[bold white]Response Length Summary[/]", style="white"))
    table = Table(show_header=True, header_style="bold white", border_style="dim white")
    table.add_column("Stage",    style="bold", min_width=20)
    table.add_column("Tokens (approx.)", justify="right")
    table.add_column("Preview", no_wrap=False, max_width=70)

    for stage, text in responses:
        word_count = str(len(text.split()))
        preview = text[:120].replace("\n", " ") + ("…" if len(text) > 120 else "")
        table.add_row(
            f"[{stage['color']}]{stage['icon']}  {stage['label']}[/]",
            word_count,
            preview,
        )

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
