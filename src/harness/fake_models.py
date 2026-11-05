"""Deterministic stand-ins for the three model calls. No network.

* classify  - keyword based: a message mentioning a known topic's label routes
              to it; otherwise the keyword becomes a new topic.
* summarize - appends each assistant reply as a "fact".
* respond   - mirrors the real model's shape: content is a LIST of blocks
              (thinking + text), and a question containing "quote" or "exactly"
              first calls the recall_detail tool, then answers from its result.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from harness.models import ClassifyInputs, SummarizeInputs
from harness.state import TopicDecision, TopicSummary
from harness.text import text_of

KEYWORDS = {
    "tigers": ["tiger", "tigers", "stripes", "bengal", "siberian"],
    "weather": ["weather", "rain", "forecast", "temperature", "sunny"],
}
FILLER = "Tigers are the largest living cat species and are apex predators. "


def _topic_for(text: str) -> str | None:
    t = text.lower()
    for key, words in KEYWORDS.items():
        if any(w in t for w in words):
            return key
    return None


def _blocks(text: str) -> list[dict]:
    """The shape ChatAnthropic returns with thinking enabled."""
    return [{"type": "thinking", "thinking": "fake reasoning"}, {"type": "text", "text": text}]


class ScriptedModels:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, inputs: ClassifyInputs) -> TopicDecision:
        self.calls.append("classify")
        hit = _topic_for(inputs.new_message)
        active = inputs.active_topic
        if active is None and hit is None:
            return TopicDecision(decision="new", label="General", one_liner="General chat", reasoning="fake: nothing known")
        if hit is None or (active and hit == active["id"]):
            return TopicDecision(decision="continue", reasoning="fake: no keyword for another topic")
        for t in inputs.other_topics:
            if t["id"] == hit:
                return TopicDecision(decision="switch_existing", topic_id=t["id"], reasoning=f"fake: keyword {hit}")
        return TopicDecision(
            decision="new", label=hit.title(), one_liner=f"Conversation about {hit}", reasoning=f"fake: new keyword {hit}"
        )

    def summarize(self, inputs: SummarizeInputs) -> TopicSummary:
        self.calls.append("summarize")
        prior = inputs.prior or TopicSummary()
        facts = list(prior.facts)
        for m in inputs.messages:
            if isinstance(m, AIMessage) and text_of(m).strip():
                facts.append(text_of(m).strip()[:80])
        return TopicSummary(
            facts=facts,
            user_preferences=prior.user_preferences,
            open_questions=[],
            entities=sorted(set(prior.entities) | {inputs.topic["label"]}),
            narrative=f"We discussed {inputs.topic['label']} over {len(facts)} points.",
        )

    def respond(self, messages: list[BaseMessage], tools: list) -> AIMessage:
        self.calls.append("respond")
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        last = messages[-1]
        question = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        text = text_of(question) if question else ""
        n_facts = len(re.findall(r"^- ", str(system), flags=re.M))

        # Tool round-trip: answer from the tool result.
        if isinstance(last, ToolMessage):
            return AIMessage(content=_blocks(f"[fake answer from transcript | {text_of(last)[:120]}]"))

        # Ask for raw detail first when the user wants exact wording.
        if tools and any(w in text.lower() for w in ("quote", "exactly")):
            m = re.search(r"\(id=([^)]+)\)", str(system))
            topic_id = m.group(1) if m else "unknown"
            return AIMessage(
                content=[{"type": "thinking", "thinking": "fake: need the transcript"}],
                tool_calls=[{"name": "recall_detail", "args": {"topic_id": topic_id, "query": text}, "id": "call_fake_1"}],
            )

        # A long, distinctive reply so the token threshold is reached in a few turns.
        return AIMessage(content=_blocks(f"[fake reply to '{text}' | summary facts visible: {n_facts}] " + FILLER * 3))
