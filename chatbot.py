"""
Multi-LoRA Clinical Chatbot — Terminal UI
Run:  python chatbot.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.spinner import Spinner
from rich import print as rprint
from rich.live import Live

from orchestrator import MedicalOrchestrator

console = Console()

EXPERT_STYLES = {
    "cardiology":         ("CARDIOLOGY",        "bold red"),
    "oncology":           ("ONCOLOGY",           "bold magenta"),
    "infectious_disease": ("INFECTIOUS DISEASE", "bold green"),
}

COMMANDS = {
    "reset": "Clear current session history",
    "exit":  "Quit the chatbot",
    "quit":  "Quit the chatbot",
}


def print_banner():
    banner = Text()
    banner.append("  Multi-LoRA Clinical Diagnostic Chatbot  \n", style="bold white")
    banner.append("  Powered by Qwen2.5-3B · Dynamic Expert Routing  ", style="dim white")
    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))
    console.print(
        "  Commands: [bold yellow]reset[/] — clear history  |  [bold yellow]exit[/] — quit\n",
        style="dim"
    )


def print_expert_badge(expert: str):
    label, style = EXPERT_STYLES[expert]
    console.print(f"Routing to  [bold]{label}[/]  expert", style=style)


def print_response(expert: str, text: str):
    _, style = EXPERT_STYLES[expert]
    console.print(
        Panel(
            text,
            title=f"[{style}]{EXPERT_STYLES[expert][0]}[/]",
            border_style=style.split()[-1],   # extract color token
            padding=(1, 2),
        )
    )


def main():
    print_banner()

    with console.status("[cyan]Loading models into VRAM…[/]", spinner="dots"):
        orchestrator = MedicalOrchestrator()

    session_id = "cli_session"
    console.print(Rule("[dim]Session started[/]", style="dim cyan"))

    while True:
        try:
            user_input = console.input("\n[bold cyan]You ›[/] ").strip()
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
            orchestrator.reset_session(session_id)
            console.print("[yellow]✓ Session history cleared.[/]")
            continue

        # Routing + generation with a live spinner
        expert = None
        response = None

        with Live(Spinner("dots", text="[cyan]Routing query…[/]"), refresh_per_second=12, console=console):
            expert, response = orchestrator.chat(session_id, user_input)

        print_expert_badge(expert)
        print_response(expert, response)


if __name__ == "__main__":
    main()
