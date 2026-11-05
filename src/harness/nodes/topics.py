"""Nodes 2a/2b: start a brand-new topic, or recall an existing one.

Both begin by consolidating the *outgoing* topic so nothing is lost, then
replace the working window.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig

from harness.deps import Deps
from harness.memory import store as S
from harness.memory.consolidate import consolidate
from harness.nodes.classify import user_id_of
from harness.state import MemoryState
from harness.tracing import trace


def _consolidate_outgoing(deps: Deps, state: MemoryState, user_id: str) -> dict:
    outgoing = state.get("active_topic_id")
    if not outgoing:
        return {}
    return consolidate(
        store=deps.store,
        models=deps.models,
        settings=deps.settings,
        user_id=user_id,
        topic_id=outgoing,
        messages=state["messages"],
        reason="switch-out",
    )


def make_start_topic_node(deps: Deps):
    def start_topic(state: MemoryState, config: RunnableConfig) -> dict:
        user_id = user_id_of(config)
        decision = state["last_decision"]
        update = _consolidate_outgoing(deps, state, user_id)

        topic = S.new_topic(deps.store, user_id, decision["label"], decision.get("one_liner") or "")
        trace("topic.new", topic_id=topic["id"], label=topic["label"], previous=state.get("active_topic_id"))
        new_msg = state["messages"][-1]
        return {
            **update,
            "active_topic_id": topic["id"],
            "topic_summary": "",
            "unconsolidated_tokens": count_tokens_approximately([new_msg]),
        }

    return start_topic


def make_recall_topic_node(deps: Deps):
    def recall_topic(state: MemoryState, config: RunnableConfig) -> dict:
        user_id = user_id_of(config)
        target = state["last_decision"]["topic_id"]
        update = _consolidate_outgoing(deps, state, user_id)

        summary = S.get_summary(deps.store, user_id, target)
        rows = S.get_recent_transcript(deps.store, user_id, target, deps.settings.recall_turns)
        reloaded = S.to_messages(rows)
        # The window must start on a user turn (the API rejects a leading assistant
        # message), so drop any assistant/tool rows that precede the first user row.
        while reloaded and not isinstance(reloaded[0], HumanMessage):
            reloaded.pop(0)
        S.touch_topic(deps.store, user_id, target, turns_delta=0)

        # Rebuild the window: drop everything, then [reloaded raw turns..., the new question].
        # The new HumanMessage is re-created so it lands *after* the reloaded turns.
        new_msg = state["messages"][-1]
        removals = [RemoveMessage(id=m.id) for m in state["messages"]]
        fresh = HumanMessage(content=new_msg.content)
        trace(
            "topic.recall",
            topic_id=target,
            previous=state.get("active_topic_id"),
            summary_loaded=summary is not None,
            raw_turns_loaded=len(reloaded),
        )
        return {
            **update,
            "messages": removals + reloaded + [fresh],
            "active_topic_id": target,
            "topic_summary": summary.render() if summary else "",
            "unconsolidated_tokens": count_tokens_approximately([fresh]),
        }

    return recall_topic
