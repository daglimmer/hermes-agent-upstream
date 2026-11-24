"""Human-authority holds on the board — cards parked waiting for a person.

A worker that needs an answer files ``kanban_block(kind="needs_input")``. The
card then waits for a person: an automatic sweep must not promote it back
into the dispatch pool, or the card looks busy on the board and does no work.

Two board facts make this check subtle, and both are why it lives on the
EVENT trail rather than on the task row:

* ``tasks.block_kind`` deliberately survives ``kanban unblock`` (clearing it
  was the amnesia that let a cron-unblock / re-block loop run unbounded), so
  the row still says ``needs_input`` after a human released the card.
* The park itself is written as ``blocked`` — or, once the unblock-loop
  breaker escalates it, as ``block_loop_detected``, which routes the card to
  ``triage`` for a human decision — and released again by ``unblocked``.

So: the NEWEST park-or-release event decides. A park event whose kind is
``needs_input`` means a person still owes an answer; a later ``unblocked``
means they gave it, and the card may move again.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Iterable
from typing import Optional

# The park-or-release trail, newest event last. ``dependency_wait`` is a park
# too, but never a human one (it waits on a parent, not on a person).
_PARK_EVENTS = ("blocked", "block_loop_detected", "dependency_wait")

# The release event: the operator let the card go.
_RELEASE_EVENT = "unblocked"

# Block kinds only a person can clear.
_HUMAN_BLOCK_KINDS = ("needs_input",)


def _payload_kind(payload: Optional[str]) -> Optional[str]:
    """The ``kind`` a park event carries, or ``None`` when it carries none."""
    if not payload:
        return None
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    kind = decoded.get("kind")
    return kind if isinstance(kind, str) else None


def _newest_park_event(
    conn: sqlite3.Connection, task_id: str
) -> Optional[sqlite3.Row]:
    """The card's newest park-or-release event, or ``None``."""
    return conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'block_loop_detected', "
        "'dependency_wait', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()


def is_parked_on_human(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the card waits for a person to answer before it may move.

    The newest park-or-release event decides: a ``needs_input`` park means a
    person still owes an answer, and an ``unblocked`` after it means they
    gave it.
    """
    row = _newest_park_event(conn, task_id)
    if row is None or row["kind"] == _RELEASE_EVENT:
        return False
    return _payload_kind(row["payload"]) in _HUMAN_BLOCK_KINDS


def parked_on_human_task_ids(
    conn: sqlite3.Connection, task_ids: Iterable[str]
) -> set[str]:
    """The subset of *task_ids* that waits for a person to answer.

    One query for the whole batch: the newest park-or-release event per card,
    then the cards whose newest one parks them on a human.
    """
    ids = list(dict.fromkeys(task_ids))
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT e.task_id AS task_id, e.kind AS kind, e.payload AS payload "
        "FROM task_events e JOIN ("
        "  SELECT task_id, MAX(id) AS newest FROM task_events "
        f"  WHERE task_id IN ({placeholders}) "
        "    AND kind IN ('blocked', 'block_loop_detected', 'dependency_wait', "
        "                 'unblocked') "
        "  GROUP BY task_id"
        ") last ON last.newest = e.id",
        tuple(ids),
    ).fetchall()
    parked: set[str] = set()
    for row in rows:
        if row["kind"] == _RELEASE_EVENT:
            continue
        if _payload_kind(row["payload"]) in _HUMAN_BLOCK_KINDS:
            parked.add(row["task_id"])
    return parked
