"""tigers -> weather -> tigers with scripted models and a throwaway SQLite file."""

import pytest
from langchain_core.messages import HumanMessage

from harness.config import Settings
from harness.fake_models import ScriptedModels
from harness.graph import build_graph, run_config
from harness.memory import store as S
from harness.persistence import open_persistence
from harness.text import text_of


@pytest.fixture
def harness():
    settings = Settings(use_fake_models=True, consolidate_after_tokens=150, keep_recent_tokens=120, db_path=":memory:")
    p = open_persistence(settings.db_path)
    models = ScriptedModels()
    graph = build_graph(settings, checkpointer=p.checkpointer, store=p.store, models=models)
    yield settings, p, models, graph
    p.close()


UPDATES: list[tuple[str, dict]] = []   # (node, partial update) for the most recent turn


def run_turn(graph, cfg, text):
    events = []
    UPDATES.clear()
    for chunk in graph.stream({"messages": [HumanMessage(text)]}, cfg, stream_mode=["updates", "custom"], version="v2"):
        if chunk["type"] == "custom":
            events.append(chunk["data"])
        elif chunk["type"] == "updates":
            UPDATES.extend(chunk["data"].items())
    return events, graph.get_state(cfg).values


def names(events, event):
    return [e for e in events if e["event"] == event]


def test_tigers_weather_tigers(harness):
    settings, p, models, graph = harness
    cfg = run_config(settings)
    uid = settings.user_id

    ev, st = run_turn(graph, cfg, "tell me about tigers")
    assert names(ev, "classify.decision")[0]["decision"] == "new"
    assert st["active_topic_id"] == "tigers"
    assert names(ev, "memory.check")[0]["fire"] is False

    ev, st = run_turn(graph, cfg, "how big do tigers get?")
    # threshold consolidation: raw rows written, summary updated, window shrunk
    assert names(ev, "memory.check")[0]["fire"] is True
    assert names(ev, "consolidate.done")[0]["reason"] == "threshold"
    rows = S.get_recent_transcript(p.store, uid, "tigers", 100)
    assert len(rows) == 4
    assert rows[1]["content"].startswith("[fake reply")        # list-of-blocks content stored as text
    assert S.get_summary(p.store, uid, "tigers") is not None
    # recency buffer keeps the last exchange (human AND ai), both tagged as already stored,
    # and the kept assistant message has been reduced to plain text (no thinking block)
    assert [m.type for m in st["messages"]] == ["human", "ai"]
    assert all(m.additional_kwargs["in_transcript"] for m in st["messages"])
    assert isinstance(st["messages"][-1].content, str)

    ev, st = run_turn(graph, cfg, "do tigers swim?")
    assert "summary facts visible: 3" in text_of(st["messages"][-1])  # summary was injected into the prompt

    ev, st = run_turn(graph, cfg, "what is the weather like today?")
    assert names(ev, "classify.decision")[0]["decision"] == "new"
    assert names(ev, "consolidate.done")[0]["reason"] == "switch-out"
    assert st["active_topic_id"] == "weather"
    tiger_rows = S.get_recent_transcript(p.store, uid, "tigers", 100)
    assert len(tiger_rows) == 6
    assert not any("weather" in r["content"] for r in tiger_rows)  # incoming question excluded from outgoing topic
    assert [m.type for m in st["messages"]] == ["human", "ai"]       # window replaced

    ev, st = run_turn(graph, cfg, "will it rain tomorrow?")
    assert names(ev, "consolidate.done")[0]["reason"] == "threshold"

    ev, st = run_turn(graph, cfg, "back to tigers - what did we say about swimming?")
    d = names(ev, "classify.decision")[0]
    assert d["decision"] == "switch_existing" and d["topic_id"] == "tigers"
    recall = names(ev, "topic.recall")[0]
    assert recall["summary_loaded"] and recall["raw_turns_loaded"] == settings.recall_turns
    assert isinstance(st["messages"][0], HumanMessage)  # rebuilt window starts on a user turn
    assert st["active_topic_id"] == "tigers"
    assert "Tigers" in st["topic_summary"]
    # 3 facts + 1 entity from the recalled summary + 1 "other topic" bullet (weather)
    assert "summary facts visible: 5" in text_of(st["messages"][-1])
    # reloaded turns are tagged; only the new exchange is pending
    assert [m.additional_kwargs.get("in_transcript", False) for m in st["messages"]] == [True, True, True, True, False, False]

    # the agent digs into the raw transcript with the recall_detail tool
    ev, st = run_turn(graph, cfg, "quote exactly what you said about tigers swimming")
    tool_ev = names(ev, "tool.recall_detail")
    assert tool_ev and tool_ev[0]["topic_id"] == "tigers"
    assert [node for node, _ in UPDATES] == ["classify_topic", "agent", "tools", "agent", "maybe_consolidate"]
    tool_msg = dict(UPDATES)["tools"]["messages"][-1]
    assert tool_msg.type == "tool" and "swim" in tool_msg.content.lower()   # transcript hits came back
    final = [u for n, u in UPDATES if n == "agent"][-1]["messages"][-1]
    assert "fake answer from transcript" in text_of(final)
    # that exchange tipped the threshold: the tool result is now in the raw transcript too
    assert names(ev, "consolidate.done")[0]["reason"] == "threshold"
    roles = [r["role"] for r in S.get_recent_transcript(p.store, uid, "tigers", 100)]
    assert "tool" in roles

    # a later consolidation must not duplicate reloaded rows in the transcript
    consolidations = 0
    for i in range(3):
        ev, _ = run_turn(graph, cfg, f"tiger follow-up number {i}")
        assert names(ev, "classify.decision")[0]["decision"] == "continue"
        consolidations += len(names(ev, "consolidate.done"))
    assert consolidations >= 1  # at least one threshold consolidation happened across the follow-ups
    rows = S.get_recent_transcript(p.store, uid, "tigers", 100)
    assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1))
    contents = [r["content"] for r in rows]
    assert len(contents) == len(set(contents)), "reloaded rows were appended to the transcript again"
    assert contents[:6] == [r["content"] for r in tiger_rows]  # original rows untouched, new ones appended after

    assert {t["id"] for t in S.list_topics(p.store, uid)} == {"tigers", "weather"}


def test_registry_is_per_user_not_per_thread(harness):
    settings, p, models, graph = harness
    run_turn(graph, run_config(settings), "tell me about tigers")

    other = Settings(**{**settings.__dict__, "thread_id": "second-session"})
    ev, st = run_turn(graph, run_config(other), "are tigers endangered?")
    d = [e for e in ev if e["event"] == "classify.decision"][0]
    assert d["decision"] == "switch_existing" and d["topic_id"] == "tigers"
    assert st["active_topic_id"] == "tigers"


def test_resume_same_thread_after_restart(harness):
    settings, p, models, graph = harness
    cfg = run_config(settings)
    run_turn(graph, cfg, "tell me about tigers")
    run_turn(graph, cfg, "how big do tigers get?")

    # "restart": a fresh graph object over the same checkpointer + store
    graph2 = build_graph(settings, checkpointer=p.checkpointer, store=p.store, models=ScriptedModels())
    ev, st = run_turn(graph2, cfg, "do tigers swim?")
    assert [e for e in ev if e["event"] == "classify.decision"][0]["decision"] == "continue"
    assert st["active_topic_id"] == "tigers"
    assert "Tigers" in st["topic_summary"]
