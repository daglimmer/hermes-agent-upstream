"""Tests for the needs_input park gate — a card waiting on a person stays put.

An automatic sweep (the gateway's auto-decompose tick, ``hermes kanban specify
--all``, ``hermes kanban decompose --all``) must not promote a card that a
worker parked with ``kind="needs_input"``: the card looks busy on the board
and does no work, and a card parked on Ray's approval can start a firewall
write that still waits for Ray's go (#118620).

These tests are LLM-free: they exercise the sweep's own enumeration and the
DB-layer promotion, not the decomposer.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_park as park
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _park_on_human(conn, tid, reason="waiting for Ray's go"):
    """Park the card the way a worker files ``kanban_block(kind="needs_input")``.

    The card is first made claimable, then blocked with the human kind, so the
    board holds a real ``blocked`` event carrying ``kind="needs_input"``.
    """
    kb.create_task(conn, title="park me", assignee="errol", idempotency_key=tid)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    conn.commit()
    assert kb.block_task(conn, tid, reason=reason, kind="needs_input") is True


def _triage_card(conn, title="rough idea"):
    return kb.create_task(conn, title=title, body="body", triage=True)


def test_needs_input_parked_card_is_not_promoted(kanban_home):
    """The auto-decompose sweep must not promote a card parked on a human.

    This is the exact failure of #118620: the card was parked with a real
    ``needs_input`` block, the auto-decomposer re-specified it (event
    ``specified``, changed_fields title+body) and promoted it straight out of
    the parked state, and the dispatcher then re-armed it every 60s while the
    guard refused each time — 164 ``respawn_guarded`` events, 0 spawns.
    """
    with kbc.connect() as conn:
        parked = _triage_card(conn, title="verify fw01 and fw03")
        plain = _triage_card(conn, title="an ordinary triage card")
        # Park it: a triage card has no live run, so it is parked via the
        # event trail the board itself uses.
        kb._append_event(conn, parked, "blocked", {"kind": "needs_input", "reason": "waiting for Ray"})
        conn.commit()

    # The park is visible on the event trail.
    with kbc.connect() as conn:
        assert park.is_parked_on_human(conn, parked) is True
        assert park.is_parked_on_human(conn, plain) is False

    # The auto-decompose sweep does not see the parked card.
    assert decomp.list_triage_ids() == [plain]
    assert spec.list_triage_ids() == [plain]

    # And so it is never promoted: the sweep never reaches it, so the card
    # stays in triage with no ``specified``/``promoted`` event.
    with kbc.connect() as conn:
        assert kb.get_task(conn, parked).status == "triage"
        kinds = [
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (parked,)
            )
        ]
    assert "specified" not in kinds
    assert "promoted" not in kinds


def test_unblocked_card_may_be_promoted_again(kanban_home):
    """A human released the card (``unblocked``): the sweep may take it again.

    ``tasks.block_kind`` deliberately survives an unblock, so a check that read
    the task row would keep the card parked forever — the newest park-or-
    release EVENT is the truth.
    """
    with kbc.connect() as conn:
        tid = _triage_card(conn, title="released card")
        kb._append_event(conn, tid, "blocked", {"kind": "needs_input", "reason": "waiting"})
        conn.commit()
    with kbc.connect() as conn:
        kb._append_event(conn, tid, "unblocked", None)
        conn.commit()

    with kbc.connect() as conn:
        assert park.is_parked_on_human(conn, tid) is False
        # The row still carries the old kind — the row is not the truth.
        assert kb.get_task(conn, tid).block_kind is None
    assert decomp.list_triage_ids() == [tid]
    assert spec.list_triage_ids() == [tid]


def test_block_loop_detected_park_keeps_the_card_parked(kanban_home):
    """A card escalated to triage by the unblock-loop breaker stays parked.

    The breaker routes the card to ``triage`` for a human and writes a
    ``block_loop_detected`` event carrying the kind. That is the same hold as
    an ordinary ``blocked`` event, so the sweep must skip it too.
    """
    with kbc.connect() as conn:
        tid = _triage_card(conn, title="breaker-escalated card")
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"kind": "needs_input", "reason": "recurrence limit", "recurrences": 2},
        )
        conn.commit()

    with kbc.connect() as conn:
        assert park.is_parked_on_human(conn, tid) is True
    assert decomp.list_triage_ids() == []
    assert spec.list_triage_ids() == []


def test_non_human_park_does_not_hold_the_card(kanban_home):
    """A ``dependency`` park waits on a parent, not on a person: the sweep may run."""
    with kbc.connect() as conn:
        tid = _triage_card(conn, title="dependency-waiting card")
        kb._append_event(conn, tid, "dependency_wait", {"kind": "dependency"})
        conn.commit()

    with kbc.connect() as conn:
        assert park.is_parked_on_human(conn, tid) is False
    assert decomp.list_triage_ids() == [tid]


def test_parked_on_human_task_ids_batches_a_mixed_set(kanban_home):
    """The batch helper reports exactly the human-parked subset."""
    with kbc.connect() as conn:
        parked = _triage_card(conn, title="parked")
        plain = _triage_card(conn, title="plain")
        kb._append_event(conn, parked, "blocked", {"kind": "needs_input"})
        conn.commit()

    with kbc.connect() as conn:
        assert park.parked_on_human_task_ids(conn, [parked, plain]) == {parked}
        assert park.parked_on_human_task_ids(conn, []) == set()


def test_park_payload_must_be_readable_json(kanban_home):
    """A corrupt/foreign payload is not a human hold — the sweep must not deadlock."""
    with kbc.connect() as conn:
        tid = _triage_card(conn, title="odd payload")
        kb._append_event(conn, tid, "blocked", {"kind": "needs_input"})
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE task_id = ? AND kind = 'blocked'",
            ("not json at all", tid),
        )
        conn.commit()

    with kbc.connect() as conn:
        assert park.is_parked_on_human(conn, tid) is False
    assert decomp.list_triage_ids() == [tid]


def test_park_detection_ignores_archived_and_other_cards(kanban_home):
    """Only the given ids are considered; the helper never widens the set."""
    with kbc.connect() as conn:
        tid = _triage_card(conn, title="card")
        kb._append_event(conn, tid, "blocked", {"kind": "needs_input"})
        conn.commit()

    with kbc.connect() as conn:
        assert park.parked_on_human_task_ids(conn, ["t_nonexistent"]) == set()
        assert park.parked_on_human_task_ids(conn, [tid]) == {tid}
        # The event payload is a JSON object, not a bare string.
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked'",
            (tid,),
        ).fetchone()
    assert jsonlib.loads(row["payload"])["kind"] == "needs_input"
