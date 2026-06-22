"""
Base Model Chatbot (no LoRA adapters) — for comparison against the Multi-LoRA expert chatbot.
Run:  python chatbot_base.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich.spinner import Spinner

BASE_MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"

SYSTEM_PROMPT = (
    "You are a helpful medical AI assistant. "
    "Provide informative, accurate responses to medical questions. "
    "Always recommend consulting a qualified physician for personal medical decisions."
)

# Qwen2.5 chat template — injected if the tokenizer doesn't carry one
_QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

console = Console()


def load_model():
    console.print("[cyan]Loading base model (no adapters)…[/]")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
    )
    model.eval()
    return tokenizer, model


def format_prompt(tokenizer, history: list[dict], device: str) -> torch.Tensor:
    try:
        return tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    except ValueError:
        tokenizer.chat_template = _QWEN_CHAT_TEMPLATE
        return tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)


def generate_response(tokenizer, model, history: list[dict], device: str) -> str:
    inputs = format_prompt(tokenizer, history, device)
    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()



def print_banner():
    banner = Text()
    banner.append("  Base Model Chatbot  \n", style="bold white")
    banner.append(f"  {BASE_MODEL_NAME} · No LoRA Adapters  ", style="dim white")
    console.print(Panel(banner, border_style="yellow", padding=(1, 4)))
    console.print(
        "  Commands: [bold yellow]reset[/] — clear history  |  [bold yellow]exit[/] — quit\n",
        style="dim"
    )


def print_response(text: str):
    console.print(
        Panel(text, title="[bold yellow]⚡ BASE MODEL[/]", border_style="yellow", padding=(1, 2))
    )




def main():
    print_banner()

    with console.status("[cyan]Loading model…[/]", spinner="dots"):
        tokenizer, model = load_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print("✅ [green]Base model ready.[/]\n")
    console.print(Rule("[dim]Session started[/]", style="dim yellow"))

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = console.input("\n[bold yellow]You ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("exit", "quit"):
            console.print("[dim]Goodbye.[/]")
            break

        if cmd == "reset":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            console.print("[yellow]✓ Session history cleared.[/]")
            continue

        history.append({"role": "user", "content": user_input})

        response = None
        with Live(Spinner("dots", text="[yellow]Generating…[/]"), refresh_per_second=12, console=console):
            response = generate_response(tokenizer, model, history, device)

        history.append({"role": "assistant", "content": response})
        print_response(response)


if __name__ == "__main__":
    main()
