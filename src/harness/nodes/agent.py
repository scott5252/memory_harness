"""Node 3: answer the user, with the topic summary in the system prompt and one
tool that can dig into raw history when the summary is not enough.

Node 4: ``maybe_consolidate`` - after the answer, check the token threshold.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_config

from harness.deps import Deps
from harness.memory import store as S
from harness.memory.consolidate import consolidate
from harness.nodes.classify import user_id_of
from harness.state import MemoryState
from harness.tracing import trace

PERSONA = """You are a helpful, concise assistant having an ongoing conversation with one user.
The conversation may span many topics. Below is what you remember about the CURRENT topic;
treat it as things the two of you already discussed - do not re-explain them, build on them.
If the summary mentions something but lacks the detail you need, call the recall_detail tool
to search the raw transcript of that topic. Never mention summaries, memory or tools to the user."""


def make_recall_tool(deps: Deps):
    @tool
    def recall_detail(topic_id: str, query: str) -> str:
        """Search the raw transcript of a previously discussed topic for specific details.
        Use when the topic summary mentions something you need to quote or elaborate on.
        topic_id must be one of the known topic ids listed in your instructions."""
        user_id = get_config()["configurable"]["user_id"]
        trace("tool.recall_detail", topic_id=topic_id, query=query)
        hits = S.search_transcript(deps.store, user_id, topic_id, query)
        if not hits:
            return "No matching transcript lines."
        return "\n".join(f"[{h['role']} #{h['seq']}] {h['content']}" for h in hits)

    return recall_detail


def build_system_prompt(deps: Deps, state: MemoryState, user_id: str) -> str:
    active_id = state.get("active_topic_id")
    topics = S.list_topics(deps.store, user_id)
    active = next((t for t in topics if t["id"] == active_id), None)
    others = [t for t in topics if t["id"] != active_id]

    parts = [PERSONA]
    if active:
        parts.append(f"## Current topic: {active['label']} (id={active['id']})")
        parts.append(state.get("topic_summary") or "(nothing consolidated yet - this topic just started)")
    if others:
        parts.append("## Other topics you have discussed with this user")
        parts.append("\n".join(f"- id={t['id']}: {t['label']} - {t['one_liner']}" for t in others))
    return "\n\n".join(parts)


def make_agent_node(deps: Deps, tools: list):
    def agent(state: MemoryState, config: RunnableConfig) -> dict:
        user_id = user_id_of(config)
        system = SystemMessage(build_system_prompt(deps, state, user_id))
        window = state["messages"]
        trace(
            "agent.call",
            window_messages=len(window),
            window_tokens=count_tokens_approximately(window),
            summary_chars=len(state.get("topic_summary") or ""),
        )
        reply: AIMessage = deps.models.respond([system, *window], tools)
        if reply.tool_calls:
            trace("agent.tool_calls", calls=[c["name"] for c in reply.tool_calls])
        return {"messages": [reply]}

    return agent


def route_after_agent(state: MemoryState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "maybe_consolidate"


def make_maybe_consolidate_node(deps: Deps):
    def maybe_consolidate(state: MemoryState, config: RunnableConfig) -> dict:
        user_id = user_id_of(config)
        pending = [m for m in state["messages"] if not m.additional_kwargs.get("in_transcript")]
        tokens = count_tokens_approximately(pending)
        threshold = deps.settings.consolidate_after_tokens
        trace("memory.check", unconsolidated_tokens=tokens, threshold=threshold, fire=tokens >= threshold)
        if tokens < threshold or not state.get("active_topic_id"):
            return {"unconsolidated_tokens": tokens}
        return consolidate(
            store=deps.store,
            models=deps.models,
            settings=deps.settings,
            user_id=user_id,
            topic_id=state["active_topic_id"],
            messages=state["messages"],
            reason="threshold",
        )

    return maybe_consolidate
