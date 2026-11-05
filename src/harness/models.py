"""The only place that knows about a concrete LLM provider.

The graph calls three functions - ``classify``, ``summarize``, ``respond`` - and
never touches ``ChatAnthropic`` directly. That keeps the memory logic testable
(``fake_models.ScriptedModels``) and makes a provider swap a one-file change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from harness.config import Settings
from harness.state import Topic, TopicDecision, TopicSummary
from harness.text import text_of


@dataclass
class ClassifyInputs:
    active_topic: Topic | None
    other_topics: list[Topic]
    recent_human_turns: list[str]   # last couple of user messages in the window, oldest first
    new_message: str


@dataclass
class SummarizeInputs:
    topic: Topic
    prior: TopicSummary | None
    messages: list[BaseMessage]      # the raw messages being consolidated


class Models(Protocol):
    def classify(self, inputs: ClassifyInputs) -> TopicDecision: ...
    def summarize(self, inputs: SummarizeInputs) -> TopicSummary: ...
    def respond(self, messages: list[BaseMessage], tools: list) -> AIMessage: ...


# --- prompts ------------------------------------------------------------------

CLASSIFY_SYSTEM = """You route a user's chat message to a conversation topic.

You are given the ACTIVE topic (what the user was just talking about), a list of
OTHER known topics, the last few user messages, and the NEW message.

Decide:
- "continue": the new message is about the active topic, a follow-up, a
  clarification, a short reaction, or anything ambiguous. This is the default.
  Only leave the active topic when the new message clearly introduces a
  different subject.
- "switch_existing": the new message clearly returns to one of the OTHER topics.
  Set topic_id to that topic's id.
- "new": the new message clearly starts a subject that matches neither the active
  topic nor any other topic. Set label (2-4 words) and one_liner.

If there is no active topic, the answer is "switch_existing" when an other topic
clearly fits, otherwise "new". Always fill in reasoning (1-2 sentences)."""

SUMMARIZE_SYSTEM = """You maintain a running summary of one conversation topic.

You are given the PRIOR summary (may be empty) and NEW messages from the same
topic. Produce an UPDATED summary that merges both. Preserve every fact,
preference, decision and open question from the prior summary unless the new
messages explicitly change it. Be concrete: keep numbers, names and choices.
Mark questions that were answered as resolved by removing them from
open_questions. The narrative is a short paragraph, written in past tense,
describing what was discussed and concluded."""


def _fmt_topic(t: Topic | None) -> str:
    if t is None:
        return "(none)"
    return f"id={t['id']} | {t['label']} - {t['one_liner']}"


def render_classify_prompt(inputs: ClassifyInputs) -> str:
    others = "\n".join(f"- {_fmt_topic(t)}" for t in inputs.other_topics) or "- (none)"
    recent = "\n".join(f"- {h}" for h in inputs.recent_human_turns) or "- (none)"
    return (
        f"ACTIVE topic: {_fmt_topic(inputs.active_topic)}\n\n"
        f"OTHER known topics:\n{others}\n\n"
        f"Recent user messages:\n{recent}\n\n"
        f"NEW message: {inputs.new_message}"
    )


def render_summarize_prompt(inputs: SummarizeInputs) -> str:
    prior = inputs.prior.model_dump_json(indent=2) if inputs.prior else "(empty)"
    lines = []
    for m in inputs.messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = text_of(m)
        if content.strip():
            lines.append(f"{role}: {content}")
    return (
        f"Topic: {inputs.topic['label']} - {inputs.topic['one_liner']}\n\n"
        f"PRIOR summary:\n{prior}\n\n"
        f"NEW messages:\n" + "\n".join(lines)
    )


# --- real implementation --------------------------------------------------------

class AnthropicModels:
    def __init__(self, settings: Settings):
        from langchain_anthropic import ChatAnthropic

        # Claude 5.5 models reject forced tool_choice, so structured output must
        # use the json_schema method rather than the default function_calling.
        # Thinking is on by default for these models and its tokens count against
        # max_tokens, so leave generous room.
        helper = ChatAnthropic(model=settings.helper_model, max_tokens=8000)
        self._classifier = helper.with_structured_output(TopicDecision, method="json_schema")
        self._summarizer = helper.with_structured_output(TopicSummary, method="json_schema")
        self._agent = ChatAnthropic(model=settings.model, max_tokens=16000)

    def classify(self, inputs: ClassifyInputs) -> TopicDecision:
        return self._classifier.invoke(
            [SystemMessage(CLASSIFY_SYSTEM), HumanMessage(render_classify_prompt(inputs))]
        )

    def summarize(self, inputs: SummarizeInputs) -> TopicSummary:
        return self._summarizer.invoke(
            [SystemMessage(SUMMARIZE_SYSTEM), HumanMessage(render_summarize_prompt(inputs))]
        )

    def respond(self, messages: list[BaseMessage], tools: list) -> AIMessage:
        model = self._agent.bind_tools(tools) if tools else self._agent
        return model.invoke(messages)


def make_models(settings: Settings) -> Models:
    if settings.use_fake_models:
        from harness.fake_models import ScriptedModels

        return ScriptedModels()
    return AnthropicModels(settings)
