# harness — topic-aware memory for a LangGraph agent (POC)

A learning project: a CLI chat agent that keeps its context window small by
consolidating the conversation into **per-topic** memory, notices when you
change subject, and silently recalls an earlier subject when you come back
to it. Built on LangChain 1.x + LangGraph 1.x with SQLite persistence.

Read **[docs/architecture.md](docs/architecture.md)** for the design,
diagrams and a walkthrough of the code, and
**[docs/erd.md](docs/erd.md)** for the `memory.db` tables, fields and relationships.

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
uv run harness --trace                 # chat, and watch the memory machinery
uv run harness                         # chat; memory works silently
uv run harness --fake --trace          # no API key: scripted models, real memory logic
uv run harness --session work --db ./work.db --consolidate-after 800
```

Flags: `--trace`, `--session NAME`, `--user ID`, `--db PATH`,
`--consolidate-after N`, `--keep-recent N`, `--recall-turns N`,
`--model ID`, `--helper-model ID`, `--fake`.

In-chat commands: `/topics`, `/summary [id]`, `/transcript [id]`,
`/trace on|off`, `/quit`.

## Demo script

Use a low threshold so consolidation fires within a few turns:

```bash
uv run harness --trace --consolidate-after 300 --session demo
```

1. Ask three or four questions about tigers. Watch `memory.check` fire and
   `consolidate.done` shrink the window.
2. Ask about the weather. Watch `classify_topic` decide `new`, the tiger
   topic get consolidated on the way out, and the window reset.
3. Ask "back to the tigers — what did we conclude?". Watch
   `switch_existing -> tigers`, `topic.recall`, and an answer that uses the
   tiger facts without you repeating them.
4. `/quit`, relaunch with the same `--session demo`, and continue; or
   relaunch with `--session other` and ask about tigers — the registry is
   per user, so it still routes to the tiger topic.

## Tests

```bash
uv run pytest
```

The tests use scripted models and a throwaway SQLite file; no network.
