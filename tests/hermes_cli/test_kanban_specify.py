"""Tests for the specifier module + `hermes kanban specify` CLI surface.

The auxiliary LLM client is mocked — these tests don't hit any network or
real provider. They exercise the prompt plumbing, response parsing, DB
writes, and CLI flag surface.
"""

from __future__ import annotations

import argparse
import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    """Build a minimal object shaped like an OpenAI chat.completions result.

    The specifier only reads ``resp.choices[0].message.content``, so we
    avoid importing the openai SDK and build the tree with MagicMock.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    """Patch call_llm at its source module — specify_task now routes through
    it (#35566) instead of building a raw client. Returns (patcher, mock) so
    callers can still assert on the call.
    """
    mock_fn = MagicMock(return_value=_fake_aux_response(content))
    return patch("agent.auxiliary_client.call_llm", mock_fn), mock_fn


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# specify_task (module-level entry point)
# ---------------------------------------------------------------------------

def test_specify_task_happy_path(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nA concrete goal.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    assert outcome.task_id == tid
    assert outcome.new_title == "Refined rough"

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # Parent-free → recompute_ready promotes to ready.
    assert task.status == "ready"
    assert task.title == "Refined rough"
    assert "**Goal**" in (task.body or "")






# ---------------------------------------------------------------------------
# CLI wiring — argparse + _cmd_specify
# ---------------------------------------------------------------------------

def _run_cli(*argv: str) -> int:
    """Invoke the `hermes kanban …` argparse surface directly."""
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", *argv])
    return kanban_cli.kanban_command(ns)




def test_cli_specify_tenant_filter(kanban_home, capsys):
    with kbc.connect() as conn:
        outside = kb.create_task(conn, title="outside", triage=True)
        inside = kb.create_task(
            conn, title="inside", triage=True, tenant="proj-a",
        )

    content = jsonlib.dumps({"title": "spec", "body": "body"})
    p, _ = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", "--all", "--tenant", "proj-a", "--json")
    assert rc == 0
    lines = [
        jsonlib.loads(l)
        for l in capsys.readouterr().out.strip().splitlines()
        if l
    ]
    ids = {row["task_id"] for row in lines}
    assert ids == {inside}

    # The outside task stays in triage.
    with kbc.connect() as conn:
        assert kb.get_task(conn, outside).status == "triage"
        # The inside task was promoted.
        assert kb.get_task(conn, inside).status in {"todo", "ready"}


# ---------------------------------------------------------------------------
# Guard-aware body handling (t_97a9517c): the promotion guard
# (trg_planless_promotion_guard) greps the stored body for a literal
# "## CHANGE PLAN" heading; specify regenerates the body wholesale and used
# to erase the marker, bouncing the card triage -> ready -> triage forever.
# ---------------------------------------------------------------------------


def test_system_prompt_requires_change_plan_heading():
    """The generator must be told to emit the heading the guard greps for."""
    assert "## CHANGE PLAN" in spec._SYSTEM_PROMPT


def test_specify_task_preserves_coordinator_change_plan(kanban_home):
    plan = (
        "## CHANGE PLAN\n\n"
        "**Build** — COORD-MARKER-97a9517c patch\n"
        "**Verify** — tests\n"
        "**Rollback** — restore backup\n"
        "**Window** — any time"
    )
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="rough", triage=True,
            body=f"old coordinator body\n\n{plan}\n",
        )

    # The regenerated body carries no heading — the old failure mode.
    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nregenerated without a plan.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    body = task.body or ""
    assert "COORD-MARKER-97a9517c patch" in body            # section survived verbatim
    assert body.count("## CHANGE PLAN") == 1                # exactly one heading
    assert body.rstrip().endswith("**Window** — any time")  # plan section is last
    assert task.status == "ready"


def test_specify_task_does_not_duplicate_own_change_plan(kanban_home):
    """Generator emitted its own heading → the preserved coordinator section
    is not appended a second time (the guard's LIKE is already satisfied)."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="rough", triage=True,
            body="## CHANGE PLAN\n**Build** — coordinator plan",
        )

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal** x\n\n## CHANGE PLAN\n**Build** — generator plan",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert body.count("## CHANGE PLAN") == 1
    assert "generator plan" in body
    assert "coordinator plan" not in body


def test_specify_task_scaffolds_a_plan_when_none_exists_anywhere(kanban_home):
    """No heading in the old body AND none in the reply → a deterministic
    scaffold is appended so the promotion guard can accept the card.

    The aux model can ignore the prompt instruction; without this the card
    bounces ``triage -> todo -> ready -> triage`` forever.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True, body="plain old body")

    content = jsonlib.dumps({"title": "Refined rough", "body": "**Goal**\nnew."})
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert body.count("## CHANGE PLAN") == 1
    assert "## CHANGE PLAN" in body
    # The specifier's own body is preserved above the scaffold.
    assert body.startswith("**Goal**\nnew.")
    # All four sub-parts are present so the card is executable by construction.
    for part in ("**Build**", "**Verify**", "**Rollback**", "**Window**"):
        assert part in body, f"missing {part}"
    # The scaffold says a worker still has to fill it in — it makes the card
    # promotable, not complete.
    assert spec._SCAFFOLD_PLACEHOLDER in body


def test_scaffold_build_steps_come_from_the_approach_bullets():
    """The scaffold's **Build** steps are the specifier's own Approach
    bullets, so the card keeps the substance the specifier wrote."""
    body = (
        "**Goal** — do the thing.\n\n"
        "**Approach**\n"
        "- patch the parser\n"
        "- add a regression test\n\n"
        "**Acceptance criteria**\n"
        "- [ ] it works\n"
    )
    out = spec._build_change_plan_scaffold(body)
    assert "- patch the parser" in out
    assert "- add a regression test" in out
    assert "**Build**" in out
    assert out.count("## CHANGE PLAN") == 1


def test_scaffold_without_approach_section_is_still_a_valid_plan():
    """A body with no **Approach** section still yields a usable plan."""
    out = spec._build_change_plan_scaffold("**Goal** — do the thing.")
    assert "## CHANGE PLAN" in out
    for part in ("**Build**", "**Verify**", "**Rollback**", "**Window**"):
        assert part in out


def test_ensure_change_plan_resolution_order():
    """1) new body has one → untouched. 2) old body has one → re-appended
    verbatim. 3) neither → scaffold."""
    plan = "## CHANGE PLAN\n\n**Build** — coordinator\n"

    # 1. The regenerated body already carries one: left alone.
    new = "**Goal** x\n\n" + plan
    assert spec._ensure_change_plan("old", new) == new

    # 2. The old body carries one: re-appended verbatim.
    out = spec._ensure_change_plan(plan, "**Goal** x")
    assert out.count("## CHANGE PLAN") == 1
    assert "**Build** — coordinator" in out

    # 3. Neither: a scaffold is generated.
    out = spec._ensure_change_plan("plain old body", "**Goal** x")
    assert out.count("## CHANGE PLAN") == 1
    assert "## CHANGE PLAN" in out


def test_extract_change_plan_edge_cases():
    # No heading → None
    assert spec._extract_change_plan("just text") is None
    assert spec._extract_change_plan(None) is None
    assert spec._extract_change_plan("") is None
    # Heading with trailing text (the guard's LIKE still matches) → captured
    out = spec._extract_change_plan("x\n## CHANGE PLAN — locked\n**Build** a")
    assert out.startswith("## CHANGE PLAN — locked")
    assert out.endswith("**Build** a")
    # Indented heading still matches
    assert spec._extract_change_plan("  ## CHANGE PLAN\nb") is not None
