"""Everything a graph node needs besides state: settings, models, the store."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.store.base import BaseStore

from harness.config import Settings
from harness.models import Models


@dataclass
class Deps:
    settings: Settings
    models: Models
    store: BaseStore
