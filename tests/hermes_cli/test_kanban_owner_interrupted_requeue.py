"""A gateway shutdown must not charge the card's failure budget.

Regression for the 2026-11-23 bounce. At 09:58:42Z the hermes-v2 gateway took a
SIGUSR1 reload: it drained its active turns and exited. Every card run it owned
died with exit 130 in the same second, and the dispatcher booked each of them as
``crashed`` — four cards (two P1, two P2) each spent a failure, and every one of
them hit its limit and was given up on. The work was lost and the scopes with it.

The exit code alone cannot tell an owner shutdown from a worker's own crash, and
it cannot tell an owner shutdown from a user's Stop. The owner's ledger can: it
settles the admission ``interrupted`` and records WHY in the turn's own result
(``interrupt_message``). A gateway shutdown or reload stops the turn with the
owner's own reason; a user's Stop carries a different one.

So the booking decision is the dispatcher's, and it reads the owner's ledger:

- the owner ended the turn with ITS OWN shutdown or restart, and the worker
  exited 130 -> an interrupted run that is requeued, with ``consecutive_failures``
  left alone;
- anything else -> an ordinary crash, exactly as before.

These tests pin both sides of that line, and the bound on it: a card whose owner
keeps bouncing it must not be requeued forever, so the free requeues are capped
and the next owner-ended turn books as a crash the breaker can see.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.turn_exit import INTERRUPTED_EXIT_CODE
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session

NOW = int(time.time())
HOST = kb._claimer_id().split(":", 1)[0]
FINGERPRINT = "6a232414-2dbf-40ab-a534-d9b71db05658:116|4162240"

# The owner's own reasons for ending a turn (gateway/run.py). A gateway shutdown
# or reload carries one of these; a user's Stop carries "Stop requested".
SHUTDOWN_CAUSE = "Gateway shutting down"
RESTART_CAUSE = "Gateway restarting"
USER_STOP_CAUSE = "Stop requested"


def _absent_pid(base: int = 3_500_000) -> int:
    """A pid that provably does not exist on this host, so a fixture can never hit a process."""
    pid = base
    while psutil.pid_exists(pid):
        pid += 1
    return pid


class _Board:
    """A scratch board plus the scratch admission ledger an owner receipt points at."""

    def __init__(self, root: Path):
        self.path = root / "kanban.db"
        kbc.init_db(self.path)
        self.conn = kbc.connect(self.path)
        self.ledger = root / "state.db"
        self.store = open_persistence(profile_scope(home=root), eager=True)
        seed_session(self.store, "sess-1", "cli")

    def close(self) -> None:
        self.conn.close()
        self.store.close()

    # --- seeding --------------------------------------------------------------------------

    def bounced_attempt(
        self, tid: str = "t_bounced", *, cause: str | None = SHUTDOWN_CAUSE,
        worker_pid=None, bound_pid=None, owner_pid=None, exit_code=INTERRUPTED_EXIT_CODE,
        worker_result: bool = True, claim_lock: str | None = None, run_no: int = 1,
    ) -> int:
        """A ``running`` card whose worker died the way the 10:01:46Z bounce killed four.

        The shape is the incident's own: a claim, a spawned submitter, an owner receipt
        naming the owner's ledger, a bound interpreter, and the interpreter's
        ``worker_result`` recording exit 130 under that claim.
        """
        worker_pid = _absent_pid() if worker_pid is None else worker_pid
        bound_pid = _absent_pid(base=3_700_000) if bound_pid is None else bound_pid
        owner_pid = _absent_pid(base=3_600_000) if owner_pid is None else owner_pid
        lock = claim_lock or f"{HOST}:{owner_pid}"
        started = NOW - 3600
        run_id = self.conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'default', 'running', ?, ?, ?, ?)",
            (tid, lock, NOW + 900, worker_pid, started),
        ).lastrowid
        assert run_id is not None
        run = int(run_id)
        # Re-arm the card if it is already on the board: a second bounce of the SAME card
        # is a new run of it, not a second card.
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, claim_lock, claim_expires, "
            "worker_pid, worker_started_at, last_heartbeat_at, current_run_id, started_at, priority) "
            "VALUES (?, ?, 'default', 'running', ?, ?, ?, ?, ?, ?, ?, ?, 5) "
            "ON CONFLICT(id) DO UPDATE SET status='running', claim_lock=excluded.claim_lock, "
            "claim_expires=excluded.claim_expires, worker_pid=excluded.worker_pid, "
            "worker_started_at=excluded.worker_started_at, current_run_id=excluded.current_run_id, "
            "started_at=excluded.started_at",
            (tid, tid, NOW - 86400, lock, NOW + 900, worker_pid, FINGERPRINT, None, run, started),
        )
        self.event(tid, run, "claimed", {"source_status": "ready"}, started)
        self.event(tid, run, "spawned", {"pid": worker_pid, "started_at": FINGERPRINT}, started)
        request_id = f'kanban:["default","{tid}",{run_no}]'
        self.event(tid, run, "owner_admitted",
                   {"db": str(self.ledger), "session_id": "sess-1", "pid": owner_pid,
                    "request_id": request_id}, started + 2)
        self.event(tid, run, "worker_bound",
                   {"pid": bound_pid, "claim_lock": lock, "started_at": FINGERPRINT}, started + 4)
        if worker_result:
            self.event(tid, run, "worker_result",
                       {"pid": bound_pid, "claim_lock": lock, "exit_code": exit_code,
                        "last_output": "Operation interrupted: waiting for model response."},
                       started + 60)
        self._seed_admission(request_id, cause)
        self.conn.commit()
        return run

    def _seed_admission(self, request_id: str, cause: str | None) -> None:
        """Settle the owner's admission ``interrupted``, recording the cause it stopped for."""
        epoch = self.store.runtime.begin_epoch(instance_id=f"seed:{request_id}")
        record, _created = self.store.admissions.admit(
            epoch=epoch, principal_id="seed", session_id="sess-1", request_id=request_id,
            payload={"text": request_id})
        claimed = self.store.admissions.claim(epoch=epoch, session_id="sess-1")
        assert claimed is not None
        result = {"result": {"interrupted": True, "final_response": ""}}
        if cause:
            result["result"]["interrupt_message"] = cause
        self.store.admissions.settle(
            epoch=claimed.owner_epoch, admission_id=claimed.admission_id,
            generation=claimed.generation, outcome="interrupted", result=result)

    def event(self, tid: str, run: int, kind: str, payload: dict, ts: int) -> None:
        self.conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (tid, run, kind, json.dumps(payload), ts))

    # --- reading --------------------------------------------------------------------------

    def state(self, tid: str) -> dict:
        return dict(self.conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())

    def events(self, tid: str) -> list:
        return self.conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)).fetchall()

    def kinds(self, tid: str) -> list[str]:
        return [e["kind"] for e in self.events(tid)]

    def run(self, tid: str) -> dict:
        return dict(self.conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)).fetchone())


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    scratch = _Board(tmp_path)
    yield scratch
    scratch.close()


# --- the incident: an owner shutdown requeues the card without charging it ---------------

def test_a_gateway_shutdown_requeues_the_card_without_counting_a_failure(board):
    """The card's scope survives a gateway bounce; only the run died."""
    board.bounced_attempt("t_bounced")

    assert kbd.detect_crashed_workers(board.conn) == []      # not a crash
    assert kbd.detect_crashed_workers._last_owner_interrupted == ["t_bounced"]

    row = board.state("t_bounced")
    assert row["status"] == "ready"                          # requeued to its source phase
    assert row["consecutive_failures"] == 0                 # NOT charged against the card
    assert row["claim_lock"] is None and row["worker_pid"] is None

    run = board.run("t_bounced")
    assert run["outcome"] == "owner_interrupted"             # an interrupted run, not a crash
    assert run["status"] == "owner_interrupted"

    kinds = board.kinds("t_bounced")
    assert "owner_interrupted" in kinds
    assert "crashed" not in kinds
    assert "gave_up" not in kinds

    # The reason is on the card, in the owner's own words.
    assert "owner admission settled interrupted" in row["last_failure_error"]
    assert SHUTDOWN_CAUSE in row["last_failure_error"]


def test_a_gateway_restart_is_the_same_owner_shutdown(board):
    """A reload restarts the gateway; that is the same host-caused bounce."""
    board.bounced_attempt("t_bounced", cause=RESTART_CAUSE)
    assert kbd.detect_crashed_workers(board.conn) == []
    row = board.state("t_bounced")
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 0
    assert board.run("t_bounced")["outcome"] == "owner_interrupted"


# --- the other side of the line: everything else is still a crash ------------------------

@pytest.mark.parametrize("cause", [None, USER_STOP_CAUSE], ids=["no-cause", "user-stop"])
def test_an_interruption_that_is_not_an_owner_shutdown_is_still_a_crash(board, cause):
    """A user's Stop is a decision about the run, not a host bounce: it stays a crash."""
    board.bounced_attempt("t_stopped", cause=cause)
    crashed = kbd.detect_crashed_workers(board.conn)
    assert crashed == ["t_stopped"]
    row = board.state("t_stopped")
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 1                  # charged, as before
    assert board.run("t_stopped")["outcome"] == "crashed"
    assert "crashed" in board.kinds("t_stopped")


def test_a_plain_worker_crash_is_still_a_crash(board):
    """The owner did not end this turn: the worker died under it. That is a real failure."""
    board.bounced_attempt("t_crash", exit_code=1)
    assert kbd.detect_crashed_workers(board.conn) == ["t_crash"]
    assert board.state("t_crash")["consecutive_failures"] == 1
    assert board.run("t_crash")["outcome"] == "crashed"


def test_a_lost_managed_worker_is_still_a_crash(board):
    """The owner LOST the worker (it never finished): the turn's death is the card's fault."""
    board.bounced_attempt("t_lost")
    board.event("t_lost", board.run("t_lost")["id"], "worker_lost", {"reason": "worker died"}, NOW)
    board.conn.commit()
    assert kbd.detect_crashed_workers(board.conn) == ["t_lost"]
    assert board.state("t_lost")["consecutive_failures"] == 1


def test_a_worker_without_an_owner_receipt_is_still_a_crash(board):
    """No owner receipt means no owner ledger: nothing says the owner ended this turn."""
    board.bounced_attempt("t_plain", cause=None)
    # Drop the receipt so the run never reached an owner.
    board.conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'owner_admitted'",
                       ("t_plain",))
    board.conn.commit()
    assert kbd.detect_crashed_workers(board.conn) == ["t_plain"]
    assert board.state("t_plain")["consecutive_failures"] == 1


# --- the bound: a flapping gateway must not requeue a card forever -----------------------

def test_the_free_requeues_are_bounded_so_the_breaker_can_still_see_the_card(board):
    """A gateway that keeps bouncing one card would requeue it forever without the breaker
    ever seeing it. After the card's own retry budget of free requeues, the next
    owner-ended turn books as an ordinary crash."""
    limit = kbd._OWNER_INTERRUPTED_FAILURE_LIMIT
    for i in range(limit):
        board.bounced_attempt("t_flap", run_no=i + 1)
        assert kbd.detect_crashed_workers(board.conn) == []
        assert board.state("t_flap")["consecutive_failures"] == 0

    # One bounce too many: the card is charged, so the breaker can stop it.
    board.bounced_attempt("t_flap", run_no=limit + 1)
    assert kbd.detect_crashed_workers(board.conn) == ["t_flap"]
    assert board.state("t_flap")["consecutive_failures"] == 1
    assert board.run("t_flap")["outcome"] == "crashed"
