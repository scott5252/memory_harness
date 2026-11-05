"""Runtime settings for the memory harness.

Everything tunable lives here so the CLI, the tests and the docs all point at
one place. Thresholds are deliberately low so the memory mechanisms fire during
a short demo session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Models. The agent answers the user; the helper does classification and
    # summarization, which are cheaper, structured tasks.
    model: str = field(default_factory=lambda: os.getenv("HARNESS_MODEL", "claude-opus-5-5"))
    helper_model: str = field(
        default_factory=lambda: os.getenv("HARNESS_HELPER_MODEL", "claude-haiku-5-5")
    )

    # Memory thresholds (approximate tokens, counted with count_tokens_approximately).
    consolidate_after_tokens: int = 1500   # run consolidation once this much new text accumulates
    keep_recent_tokens: int = 400          # recency buffer kept in the window after consolidation
    recall_turns: int = 4                  # raw transcript rows reloaded on topic recall

    # Persistence.
    db_path: str = "memory.db"
    user_id: str = "local-user"
    thread_id: str = "default"

    # When True the graph uses scripted models (no network). Tests use this.
    use_fake_models: bool = False
