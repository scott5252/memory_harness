"""Consolidation: move the working window into long-term memory.

One function, called from three places:

* ``maybe_consolidate`` node  - the token threshold was reached (``reason="threshold"``)
* ``start_topic`` node         - user started a new topic (``reason="switch-out"``)
* ``recall_topic`` node        - user went back to an old topic (``reason="switch-out"``)

Steps, in this order (so a crash never loses raw history):
  0. choose which window messages to consolidate
  1. append them to the topic's raw transcript in the store
  2. fold them into the topic's structured summary (prior + new -> updated)
  3. return the state update: RemoveMessage for everything outside the recency buffer
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.store.base import BaseStore

from harness.config import Settings
from harness.memory import store as S
from harness.models import Models, SummarizeInputs
from harness.state import TopicSummary
from harness.text import text_of
from harness.tracing import trace


def select_for_consolidation(messages: list[BaseMessage], *, exclude_trailing_human: bool) -> list[BaseMessage]:
    """Which window messages become memory.

    * Messages reloaded from the transcript during recall are skipped: they are
      already in the store and must not be appended twice.
    * On a topic switch the trailing HumanMessage belongs to the *incoming*
      topic, so it is excluded from the outgoing topic's memory.
    """
    chosen = list(messages)
    if exclude_trailing_human and chosen and isinstance(chosen[-1], HumanMessage):
        chosen = chosen[:-1]
    return [m for m in chosen if not m.additional_kwargs.get("in_transcript")]


def recency_buffer(messages: list[BaseMessage], keep_tokens: int) -> list[BaseMessage]:
    """The tail of the window we keep verbatim. ``start_on="human"`` guarantees the
    kept window starts on a user turn, so it can never begin with an assistant
    message or with a ToolMessage whose tool-call AIMessage was dropped (the
    Anthropic API rejects both). Note: do NOT pass ``end_on`` here - it discards
    everything *after* the last message of that type, i.e. the latest reply."""
    if keep_tokens <= 0:
        return []
    return trim_messages(
        messages,
        strategy="last",
        max_tokens=keep_tokens,
        token_counter=count_tokens_approximately,
        start_on="human",
        include_system=False,
    )


def consolidate(
    *,
    store: BaseStore,
    models: Models,
    settings: Settings,
    user_id: str,
    topic_id: str,
    messages: list[BaseMessage],
    reason: str,
) -> dict:
    """Returns a partial state update. Caller merges it into its own update."""
    switching = reason == "switch-out"
    to_store = select_for_consolidation(messages, exclude_trailing_human=switching)
    before_tokens = count_tokens_approximately(messages)
    trace(
        "consolidate.start",
        reason=reason,
        topic_id=topic_id,
        window_messages=len(messages),
        to_store=len(to_store),
        window_tokens=before_tokens,
    )

    if not to_store:
        trace("consolidate.skip", reason="nothing new to store")
        summary = S.get_summary(store, user_id, topic_id)
    else:
        topic = S.get_topic(store, user_id, topic_id)
        assert topic is not None, f"unknown topic {topic_id}"

        # 1. raw transcript first - the lossless record
        S.append_transcript(store, user_id, topic_id, to_store)

        # 2. incremental summary
        prior = S.get_summary(store, user_id, topic_id)
        summary = models.summarize(SummarizeInputs(topic=topic, prior=prior, messages=to_store))
        S.put_summary(store, user_id, topic_id, summary)
        S.touch_topic(store, user_id, topic_id, turns_delta=sum(isinstance(m, HumanMessage) for m in to_store))
        trace("consolidate.summary", topic_id=topic_id, facts=len(summary.facts), open_questions=len(summary.open_questions))

    # 3. shrink the window.
    #    - on a switch keep only the incoming human message (it belongs to the next topic)
    #    - otherwise keep a short recency buffer so the model still sees the last exchange
    if switching:
        keep = [messages[-1]] if messages and isinstance(messages[-1], HumanMessage) else []
    else:
        keep = recency_buffer(messages, settings.keep_recent_tokens)
    keep_ids = {m.id for m in keep}
    removals = [RemoveMessage(id=m.id) for m in messages if m.id not in keep_ids]

    # Kept messages that were just written to the transcript get re-emitted with the
    # same id (add_messages replaces by id) and a tag, so the next consolidation
    # does not append them a second time. Kept assistant messages are also reduced
    # to plain text: their thinking blocks referred to turns we just deleted, and
    # replaying thinking blocks against an edited history can be rejected.
    stored_ids = {m.id for m in to_store}
    retagged = []
    for m in keep:
        if m.id not in stored_ids:
            continue
        update = {"additional_kwargs": {**m.additional_kwargs, "in_transcript": True}}
        if isinstance(m, AIMessage):
            update["content"] = text_of(m)
        retagged.append(m.model_copy(update=update))
    pending = [m for m in keep if m.id not in stored_ids and not m.additional_kwargs.get("in_transcript")]
    trace(
        "consolidate.done",
        reason=reason,
        removed=len(removals),
        kept=len(keep),
        window_tokens_after=count_tokens_approximately(keep) if keep else 0,
    )
    return {
        "messages": removals + retagged,
        "topic_summary": summary.render() if summary else "",
        "unconsolidated_tokens": count_tokens_approximately(pending) if pending else 0,
    }


__all__ = ["consolidate", "select_for_consolidation", "recency_buffer", "TopicSummary"]
