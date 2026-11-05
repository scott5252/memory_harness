"""CLI: a chat loop with an optional trace of the memory machinery.

    uv run harness                       # plain chat, memory works silently
    uv run harness --trace               # show classification, store I/O, consolidation, recall
    uv run harness --session work --db ./work.db --consolidate-after 800

In-chat commands: /topics  /summary [topic_id]  /transcript [topic_id]  /trace on|off  /quit
"""

from __future__ import annotations

import os
from typing import Annotated, Optional

import typer
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from harness.config import Settings
from harness.graph import build_graph, run_config
from harness.memory import store as S
from harness.persistence import open_persistence
from harness.text import text_of
from harness.tracing import drain_buffer

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

# Colour per event family so the trace is scannable.
STYLE = {
    "classify": "bold magenta",
    "topic": "bold yellow",
    "consolidate": "bold cyan",
    "memory": "cyan",
    "store": "dim green",
    "agent": "blue",
    "tool": "bold green",
}


def render_event(e: dict) -> None:
    event = e["event"]
    family = event.split(".")[0]
    style = STYLE.get(family, "white")
    details = "  ".join(f"{k}={v!r}" if not isinstance(v, str) else f"{k}={v}" for k, v in e.items() if k != "event")
    if event == "classify.decision":
        console.print(Panel(f"[bold]{e['decision']}[/] -> {e.get('topic_id') or e.get('label')}\n[dim]{e['reasoning']}[/]",
                            title="classify_topic", style=style, expand=False))
    else:
        console.print(f"  [{style}]{event:<22}[/] {details}")


@app.command()
def main(
    trace: Annotated[bool, typer.Option("--trace/--no-trace", help="Show the memory machinery as it runs.")] = False,
    session: Annotated[str, typer.Option(help="Thread id. Same name resumes the same conversation.")] = "default",
    user: Annotated[str, typer.Option(help="User id. Topics are remembered per user across sessions.")] = "local-user",
    db: Annotated[str, typer.Option(help="SQLite file for checkpoints and memory.")] = "memory.db",
    model: Annotated[Optional[str], typer.Option(help="Agent model id.")] = None,
    helper_model: Annotated[Optional[str], typer.Option(help="Model for classification and summarization.")] = None,
    consolidate_after: Annotated[int, typer.Option(help="Approx tokens of new conversation before consolidation.")] = 1500,
    keep_recent: Annotated[int, typer.Option(help="Approx tokens kept verbatim after consolidation.")] = 400,
    recall_turns: Annotated[int, typer.Option(help="Raw transcript rows reloaded on topic recall.")] = 4,
    fake: Annotated[bool, typer.Option("--fake", help="Use scripted models (no API key, for exploring the mechanics).")] = False,
) -> None:
    settings = Settings(
        thread_id=session,
        user_id=user,
        db_path=db,
        consolidate_after_tokens=consolidate_after,
        keep_recent_tokens=keep_recent,
        recall_turns=recall_turns,
        use_fake_models=fake,
    )
    if model:
        settings.model = model
    if helper_model:
        settings.helper_model = helper_model
    if not fake and not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        console.print("[yellow]No ANTHROPIC_API_KEY set. Set it, or pass --fake to use scripted models.[/]")

    persistence = open_persistence(settings.db_path)
    graph = build_graph(settings, checkpointer=persistence.checkpointer, store=persistence.store)
    cfg = run_config(settings)
    show_trace = trace

    console.print(Panel(
        f"session=[bold]{session}[/]  user=[bold]{user}[/]  db=[bold]{persistence.path}[/]\n"
        f"model={settings.model}  helper={settings.helper_model}  consolidate_after={consolidate_after} tokens\n"
        "[dim]/topics  /summary [id]  /transcript [id]  /trace on|off  /quit[/]",
        title="memory harness", expand=False,
    ))

    def flush_buffer() -> None:
        for e in drain_buffer():
            if show_trace:
                render_event(e)

    try:
        while True:
            try:
                text = console.input("[bold green]you>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text.startswith("/trace"):
                show_trace = text.split()[-1] != "off"
                console.print(f"[dim]trace {'on' if show_trace else 'off'}[/]")
                continue
            if text == "/topics":
                table = Table("id", "label", "one-liner", "turns", "last active")
                for t in S.list_topics(persistence.store, user):
                    table.add_row(t["id"], t["label"], t["one_liner"], str(t["turns"]), t["last_active"])
                console.print(table)
                drain_buffer()
                continue
            if text.startswith("/summary") or text.startswith("/transcript"):
                parts = text.split()
                state = graph.get_state(cfg).values
                topic_id = parts[1] if len(parts) > 1 else state.get("active_topic_id")
                if not topic_id:
                    console.print("[dim]no active topic[/]")
                    continue
                if parts[0] == "/summary":
                    summary = S.get_summary(persistence.store, user, topic_id)
                    console.print(Panel(summary.render() if summary else "(none yet)", title=f"summary: {topic_id}"))
                else:
                    for r in S.get_recent_transcript(persistence.store, user, topic_id, 1000):
                        console.print(f"[dim]{r['seq']:>4}[/] [bold]{r['role']}[/]: {r['content']}")
                drain_buffer()
                continue

            final: AIMessage | None = None
            for chunk in graph.stream({"messages": [HumanMessage(text)]}, cfg, stream_mode=["updates", "custom"], version="v2"):
                if chunk["type"] == "custom":
                    if show_trace:
                        render_event(chunk["data"])
                elif chunk["type"] == "updates":
                    for node, upd in chunk["data"].items():
                        if show_trace:
                            console.print(f"[dim]-- node {node} done[/]")
                        if node == "agent" and upd and upd.get("messages"):
                            last = upd["messages"][-1]
                            if isinstance(last, AIMessage) and not last.tool_calls:
                                final = last
            flush_buffer()
            if final is not None:
                console.print()
                console.print(Markdown(text_of(final)))
                console.print()
    finally:
        persistence.close()


if __name__ == "__main__":
    app()
