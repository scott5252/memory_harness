"""Assemble the LangGraph.

    START -> classify_topic -+-> (continue)        -> agent
                             +-> (switch_existing) -> recall_topic -> agent
                             +-> (new)             -> start_topic  -> agent
    agent -+-> (tool calls) -> tools -> agent
           +-> maybe_consolidate -> END
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from harness.config import Settings
from harness.deps import Deps
from harness.models import Models, make_models
from harness.nodes.agent import make_agent_node, make_maybe_consolidate_node, make_recall_tool, route_after_agent
from harness.nodes.classify import make_classify_node, route_after_classify
from harness.nodes.topics import make_recall_topic_node, make_start_topic_node
from harness.state import MemoryState


def build_graph(
    settings: Settings,
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    models: Models | None = None,
):
    deps = Deps(settings=settings, models=models or make_models(settings), store=store)
    tools = [make_recall_tool(deps)]

    g = StateGraph(MemoryState)
    g.add_node("classify_topic", make_classify_node(deps))
    g.add_node("start_topic", make_start_topic_node(deps))
    g.add_node("recall_topic", make_recall_topic_node(deps))
    g.add_node("agent", make_agent_node(deps, tools))
    g.add_node("tools", ToolNode(tools))
    g.add_node("maybe_consolidate", make_maybe_consolidate_node(deps))

    g.add_edge(START, "classify_topic")
    g.add_conditional_edges(
        "classify_topic",
        route_after_classify,
        {"continue": "agent", "switch_existing": "recall_topic", "new": "start_topic"},
    )
    g.add_edge("start_topic", "agent")
    g.add_edge("recall_topic", "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "maybe_consolidate": "maybe_consolidate"})
    g.add_edge("tools", "agent")
    g.add_edge("maybe_consolidate", END)

    return g.compile(checkpointer=checkpointer, store=store)


def run_config(settings: Settings) -> dict:
    """The per-invoke config: thread_id selects the checkpoint, user_id the store namespace."""
    return {"configurable": {"thread_id": settings.thread_id, "user_id": settings.user_id}}
