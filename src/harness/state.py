"""Graph state and the structured schemas exchanged with the models.

Three things live here:

* ``MemoryState``   - what LangGraph checkpoints every turn (the working window).
* ``TopicDecision`` - what the topic classifier returns.
* ``TopicSummary``  - what the summarizer returns and what we store per topic.

The topic *registry* is intentionally NOT in state. It lives in the LangGraph
store, namespaced by user, so a brand-new thread still knows every topic the
user has ever discussed.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class Topic(TypedDict):
    """Registry entry stored at (user_id, "topics") / key=topic_id."""

    id: str
    label: str
    one_liner: str
    turns: int
    last_active: str


class MemoryState(MessagesState):
    """Per-thread working state. ``messages`` is inherited from MessagesState."""

    active_topic_id: str | None
    topic_summary: str              # rendered summary of the active topic, "" when none yet
    unconsolidated_tokens: int      # approx tokens in the window not yet written to memory
    last_decision: dict | None      # the classifier's TopicDecision for the current turn (drives routing)


class TopicDecision(BaseModel):
    """Classifier output: does the new message continue, switch, or start a topic?"""

    decision: Literal["continue", "switch_existing", "new"] = Field(
        description="continue = same topic as the active one; switch_existing = one of the "
        "other known topics; new = none of the known topics fit."
    )
    topic_id: str | None = Field(
        default=None, description="Required when decision is switch_existing: the id of the known topic."
    )
    label: str | None = Field(
        default=None, description="Required when decision is new: a 2-4 word label for the topic."
    )
    one_liner: str | None = Field(
        default=None, description="Required when decision is new: one sentence describing the topic."
    )
    reasoning: str = Field(description="One or two sentences explaining the decision.")


class TopicSummary(BaseModel):
    """Structured, incrementally-updated summary of one topic."""

    facts: list[str] = Field(default_factory=list, description="Concrete facts established so far.")
    user_preferences: list[str] = Field(
        default_factory=list, description="Preferences, decisions or constraints the user expressed."
    )
    open_questions: list[str] = Field(default_factory=list, description="Unresolved questions.")
    entities: list[str] = Field(default_factory=list, description="Named things worth remembering.")
    narrative: str = Field(default="", description="A short paragraph telling the story of the topic so far.")

    def render(self) -> str:
        """Render as a compact block for the system prompt."""
        parts: list[str] = []
        if self.narrative:
            parts.append(self.narrative)
        for title, items in (
            ("Facts", self.facts),
            ("User preferences / decisions", self.user_preferences),
            ("Open questions", self.open_questions),
            ("Entities", self.entities),
        ):
            if items:
                parts.append(f"{title}:\n" + "\n".join(f"- {i}" for i in items))
        return "\n\n".join(parts)
