from langchain_core.messages import AIMessage, HumanMessage

from harness.memory import store as S
from harness.persistence import open_persistence
from harness.state import TopicSummary


def test_registry_summary_and_transcript_roundtrip():
    p = open_persistence(":memory:")
    try:
        t = S.new_topic(p.store, "u", "Tigers", "Facts about tigers")
        assert t["id"] == "tigers"
        # a second topic with the same label gets a distinct id
        assert S.new_topic(p.store, "u", "Tigers", "again")["id"] == "tigers-2"

        S.append_transcript(p.store, "u", "tigers", [HumanMessage("tell me about tigers"), AIMessage("Largest cats.")])
        S.append_transcript(p.store, "u", "tigers", [HumanMessage("how big?")])
        rows = S.get_recent_transcript(p.store, "u", "tigers", 10)
        assert [r["seq"] for r in rows] == [1, 2, 3]
        assert [r["role"] for r in rows] == ["user", "assistant", "user"]

        S.put_summary(p.store, "u", "tigers", TopicSummary(facts=["largest cat"]))
        assert S.get_summary(p.store, "u", "tigers").facts == ["largest cat"]
        assert S.get_summary(p.store, "u", "tigers-2") is None

        # registry listing must not leak summary / transcript rows (prefix search)
        ids = {t["id"] for t in S.list_topics(p.store, "u")}
        assert ids == {"tigers", "tigers-2"}

        hits = S.search_transcript(p.store, "u", "tigers", "largest cats")
        assert [h["seq"] for h in hits] == [2]

        rebuilt = S.to_messages(rows)
        assert all(m.additional_kwargs["in_transcript"] for m in rebuilt)
    finally:
        p.close()
