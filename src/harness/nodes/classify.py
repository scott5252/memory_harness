"""Node 1: decide which topic the new user message belongs to."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from harness.deps import Deps
from harness.memory import store as S
from harness.models import ClassifyInputs
from harness.state import MemoryState, TopicDecision
from harness.tracing import trace


def user_id_of(config: RunnableConfig) -> str:
    return config["configurable"]["user_id"]


def make_classify_node(deps: Deps):
    def classify_topic(state: MemoryState, config: RunnableConfig) -> dict:
        user_id = user_id_of(config)
        messages = state["messages"]
        new_msg = messages[-1]
        assert isinstance(new_msg, HumanMessage), "graph input must end with the user's message"

        active_id = state.get("active_topic_id")
        active = S.get_topic(deps.store, user_id, active_id) if active_id else None
        others = [t for t in S.list_topics(deps.store, user_id) if t["id"] != active_id]
        recent = [m.content for m in messages[:-1] if isinstance(m, HumanMessage)][-2:]

        decision = deps.models.classify(
            ClassifyInputs(active_topic=active, other_topics=others, recent_human_turns=recent, new_message=new_msg.content)
        )
        decision = _sanitize(decision, active_id, {t["id"] for t in others})
        trace(
            "classify.decision",
            decision=decision.decision,
            topic_id=decision.topic_id or (active_id if decision.decision == "continue" else None),
            label=decision.label,
            reasoning=decision.reasoning,
            known_topics=len(others) + (1 if active else 0),
        )
        return {"last_decision": decision.model_dump()}

    return classify_topic


def _sanitize(d: TopicDecision, active_id: str | None, other_ids: set[str]) -> TopicDecision:
    """Guard against impossible decisions so routing never dead-ends."""
    if d.decision == "continue" and active_id is None:
        return d.model_copy(update={"decision": "new", "label": d.label or "General", "one_liner": d.one_liner or "General conversation"})
    if d.decision == "switch_existing" and d.topic_id not in other_ids:
        if d.topic_id == active_id:
            return d.model_copy(update={"decision": "continue"})
        return d.model_copy(update={"decision": "new", "label": d.label or (d.topic_id or "General"), "one_liner": d.one_liner or ""})
    if d.decision == "new" and not d.label:
        return d.model_copy(update={"label": "General", "one_liner": d.one_liner or "General conversation"})
    return d


def route_after_classify(state: MemoryState) -> str:
    return state["last_decision"]["decision"]  # "continue" | "switch_existing" | "new"
