"""Thin, named helpers over the LangGraph store.

Namespace layout (all under one user):

    (user_id, "topics")                          key=<topic_id>  -> Topic registry entry
    (user_id, "topics", <topic_id>)              key="summary"   -> TopicSummary dict
    (user_id, "topics", <topic_id>, "transcript") key="000001".. -> {"role","content","ts","seq"}

Every function takes the ``BaseStore`` explicitly so the graph nodes, the CLI
and the tests all share the same code path. Each helper also emits a trace
event so ``--trace`` shows exactly which namespace was read or written.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.store.base import BaseStore

from harness.state import Topic, TopicSummary
from harness.text import text_of
from harness.tracing import trace


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def topics_ns(user_id: str) -> tuple[str, ...]:
    return (user_id, "topics")


def topic_ns(user_id: str, topic_id: str) -> tuple[str, ...]:
    return (user_id, "topics", topic_id)


def transcript_ns(user_id: str, topic_id: str) -> tuple[str, ...]:
    return (user_id, "topics", topic_id, "transcript")


# --- registry -----------------------------------------------------------------

def _search_exact(store: BaseStore, ns: tuple[str, ...], limit: int) -> list:
    """``store.search`` matches the namespace *prefix*, so (u, "topics") would also
    return summaries and transcript rows. Keep only items in exactly ``ns``."""
    return [item for item in store.search(ns, limit=limit) if tuple(item.namespace) == ns]


def list_topics(store: BaseStore, user_id: str) -> list[Topic]:
    items = _search_exact(store, topics_ns(user_id), limit=100)
    topics = [Topic(**item.value) for item in items]
    topics.sort(key=lambda t: t["last_active"], reverse=True)
    trace("store.read", ns=topics_ns(user_id), count=len(topics))
    return topics


def get_topic(store: BaseStore, user_id: str, topic_id: str) -> Topic | None:
    item = store.get(topics_ns(user_id), topic_id)
    return Topic(**item.value) if item else None


def upsert_topic(store: BaseStore, user_id: str, topic: Topic) -> None:
    store.put(topics_ns(user_id), topic["id"], dict(topic))
    trace("store.write", ns=topics_ns(user_id), key=topic["id"], what="registry entry")


def new_topic(store: BaseStore, user_id: str, label: str, one_liner: str) -> Topic:
    slug = "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-")
    existing = {t["id"] for t in list_topics(store, user_id)}
    topic_id, n = slug, 2
    while topic_id in existing:
        topic_id, n = f"{slug}-{n}", n + 1
    topic = Topic(id=topic_id, label=label, one_liner=one_liner, turns=0, last_active=_now())
    upsert_topic(store, user_id, topic)
    return topic


def touch_topic(store: BaseStore, user_id: str, topic_id: str, turns_delta: int = 1) -> None:
    topic = get_topic(store, user_id, topic_id)
    if topic is None:
        return
    topic["turns"] += turns_delta
    topic["last_active"] = _now()
    upsert_topic(store, user_id, topic)


# --- summary ------------------------------------------------------------------

def get_summary(store: BaseStore, user_id: str, topic_id: str) -> TopicSummary | None:
    item = store.get(topic_ns(user_id, topic_id), "summary")
    trace("store.read", ns=topic_ns(user_id, topic_id), key="summary", found=item is not None)
    return TopicSummary(**item.value) if item else None


def put_summary(store: BaseStore, user_id: str, topic_id: str, summary: TopicSummary) -> None:
    store.put(topic_ns(user_id, topic_id), "summary", summary.model_dump())
    trace("store.write", ns=topic_ns(user_id, topic_id), key="summary", facts=len(summary.facts))


# --- transcript ---------------------------------------------------------------

def _transcript_items(store: BaseStore, user_id: str, topic_id: str) -> list[dict]:
    items = _search_exact(store, transcript_ns(user_id, topic_id), limit=10_000)
    rows = [item.value for item in items]
    rows.sort(key=lambda r: r["seq"])
    return rows


def append_transcript(
    store: BaseStore, user_id: str, topic_id: str, messages: list[BaseMessage]
) -> int:
    """Append raw messages, in order, after whatever is already there. Returns rows written."""
    existing = _transcript_items(store, user_id, topic_id)
    seq = existing[-1]["seq"] if existing else 0
    written = 0
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else m.type
        content = text_of(m)
        if not content.strip():
            continue  # e.g. an AIMessage that only carried tool calls or thinking
        seq += 1
        store.put(
            transcript_ns(user_id, topic_id),
            f"{seq:06d}",
            {"seq": seq, "role": role, "content": content, "ts": _now()},
        )
        written += 1
    trace("store.write", ns=transcript_ns(user_id, topic_id), rows=written, total=seq)
    return written


def get_recent_transcript(store: BaseStore, user_id: str, topic_id: str, k: int) -> list[dict]:
    rows = _transcript_items(store, user_id, topic_id)
    recent = rows[-k:] if k > 0 else []
    trace("store.read", ns=transcript_ns(user_id, topic_id), rows=len(recent), total=len(rows))
    return recent


def search_transcript(store: BaseStore, user_id: str, topic_id: str, query: str, limit: int = 6) -> list[dict]:
    """Keyword search over raw rows. Deliberately simple for the POC; a
    SqliteStore(index=...) with embeddings would replace this."""
    terms = [t for t in query.lower().split() if len(t) > 2]
    rows = _transcript_items(store, user_id, topic_id)
    scored = []
    for r in rows:
        text = r["content"].lower()
        score = sum(text.count(t) for t in terms)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda s: (-s[0], s[1]["seq"]))
    hits = [r for _, r in scored[:limit]]
    trace("store.read", ns=transcript_ns(user_id, topic_id), query=query, hits=len(hits))
    return hits


def to_messages(rows: list[dict]) -> list[BaseMessage]:
    """Rebuild LangChain messages from transcript rows. Tagged so consolidation
    never re-appends them to the transcript."""
    out: list[BaseMessage] = []
    for r in rows:
        kwargs = {"additional_kwargs": {"in_transcript": True}}
        if r["role"] == "user":
            out.append(HumanMessage(content=r["content"], **kwargs))
        elif r["role"] == "assistant":
            out.append(AIMessage(content=r["content"], **kwargs))
        else:  # tool results are folded into an assistant note for the POC
            out.append(AIMessage(content=f"[{r['role']} result] {r['content']}", **kwargs))
    return out
