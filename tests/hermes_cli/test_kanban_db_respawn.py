"""Tests for the respawn-guard failure-text classification.

The guard reads ``tasks.last_failure_error`` and asks whether the card sits
behind a wall that retrying will not clear (``blocker_auth``). It classifies
TEXT, so the dispatcher's own bookkeeping must not read as a diagnosis: its
rate-limit requeue note names a quota wall while reporting a run it released
on purpose, without counting a failure.

That note parked cards for good: the cooldown branch only covers the window
while the LATEST run is still ``rate_limited``, so one later run (a block, a
crash) turned the stale stamp into a permanent deferral — 61 cards fleet-wide
and 164 ``respawn_guarded`` events on one card with 0 spawns (#118620).

Genuine quota/auth failures must still park the card.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_respawn as respawn


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# The dispatcher's own note, verbatim from ``_classify_dead_worker_exit`` —
# including the owner's attempt-note suffix the board appends to it.
REQUEUE_TEXT = (
    "pid 1163475 exited rate-limited (quota wall) — requeued without counting a failure"
)
REQUEUE_TEXT_WITH_NOTE = (
    "pid 1163475 exited rate-limited (quota wall) — requeued without counting a failure "
    "[owner admission settled failed]"
)


def test_requeue_text_classifies_as_none():
    """The dispatcher's own requeue note is not a blocker — it names no wall.

    The note reports a run the dispatcher released on purpose, WITHOUT counting
    a failure; the card stays retryable. Classifying it as ``blocker_auth``
    parked the card for good once a later run superseded the rate-limited one.
    """
    assert respawn.classify_respawn_blocker(REQUEUE_TEXT) is None
    assert respawn.classify_respawn_blocker(REQUEUE_TEXT_WITH_NOTE) is None


def test_requeue_text_classifies_as_none_through_the_guard(kanban_home, monkeypatch):
    """The guard does not park a card on the dispatcher's own bookkeeping.

    This is the #118620 chain end to end: the card's ``last_failure_error``
    holds the dispatcher's requeue note (with the owner's note appended), the
    latest run is NOT ``rate_limited`` any more, so the cooldown branch no
    longer covers it — and the guard used to return ``blocker_auth`` on the
    stale stamp, deferring the card forever.
    """
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="requeued card", assignee="errol")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            (REQUEUE_TEXT_WITH_NOTE, tid),
        )
        conn.commit()

    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize(
    "error_text",
    [
        "429 Too Many Requests",
        "quota exceeded for this organization",
        "rate limit exceeded, retry later",
        "403 Forbidden",
        "401 auth failed",
        "authentication error from provider",
        "invalid api key",
        "billing required for this model",
    ],
)
def test_genuine_quota_and_auth_failures_still_park_the_card(error_text):
    """A real wall still parks the card — the fix does not weaken the guard."""
    assert respawn.classify_respawn_blocker(error_text) == "blocker_auth"


@pytest.mark.parametrize(
    "error_text",
    [
        "",
        None,
        "worker exited with code 1",
        "the worker could not open the file",
    ],
)
def test_text_naming_no_wall_is_not_a_blocker(error_text):
    """Text that names no wall classifies as None.

    Note the boundary: a bare ``auth`` DOES match the curated pattern (the
    guard keeps it for a real auth failure), so a worker's captured stdout is
    handled by the guard's crashed-run exemption, not by the text — see
    :func:`test_requeue_note_still_parks_when_the_card_is_crashed`.
    """
    assert respawn.classify_respawn_blocker(error_text) is None


def test_requeue_note_still_parks_when_the_card_is_crashed(kanban_home, monkeypatch):
    """A crashed run's captured output is context, not a diagnosis (#117097).

    The guard exempts a crashed run, so a worker's stdout mentioning a quota
    must not park the card either.
    """
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="crashed card", assignee="errol")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("quota exceeded", tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'failed', 'crashed', ?, ?)",
            (tid, 1, 2),
        )
        conn.commit()

    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, tid) is None


def test_genuine_auth_failure_still_guards_the_card(kanban_home, monkeypatch):
    """The guard keeps its teeth for a real auth failure."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auth card", assignee="errol")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("401 auth failed", tid),
        )
        conn.commit()

    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"
