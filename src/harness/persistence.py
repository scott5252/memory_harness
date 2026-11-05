"""Open the SQLite file and build the two LangGraph persistence objects.

* ``SqliteSaver``  - the *checkpointer*. Saves ``MemoryState`` after every node so a
  thread can be resumed. Keyed by ``thread_id``.
* ``SqliteStore``  - the *store*. A key/value namespace tree that outlives any one
  thread. We keep topic registry, summaries and raw transcripts here. Keyed by
  namespace tuples such as ``(user_id, "topics", topic_id)``.

Both live in the same file but need different connection settings (the store
runs in autocommit mode and issues its own BEGIN; the checkpointer manages
transactions itself), so each gets its own connection. SQLite's in-memory
shared-cache mode does not honor busy timeouts across connections, so
":memory:" is backed by a throwaway temp file instead; WAL mode lets the two
connections interleave reads and writes without "database is locked" errors.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


@dataclass
class Persistence:
    checkpointer: SqliteSaver
    store: SqliteStore
    path: str
    _conns: tuple[sqlite3.Connection, ...]
    _tmpdir: str | None = None

    def close(self) -> None:
        for c in self._conns:
            c.close()
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)


def open_persistence(db_path: str = "memory.db") -> Persistence:
    """``db_path`` may be ":memory:" for tests (a temp file that ``close()`` deletes)."""
    tmpdir = None
    if db_path == ":memory:":
        tmpdir = tempfile.mkdtemp(prefix="harness-")
        db_path = str(Path(tmpdir) / "memory.db")

    saver_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    store_conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=30)
    for c in (saver_conn, store_conn):
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=30000")

    checkpointer = SqliteSaver(saver_conn)
    checkpointer.setup()
    store = SqliteStore(store_conn)
    store.setup()
    return Persistence(checkpointer=checkpointer, store=store, path=db_path, _conns=(saver_conn, store_conn), _tmpdir=tmpdir)
