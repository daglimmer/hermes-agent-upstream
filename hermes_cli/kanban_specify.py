"""Kanban triage specifier — flesh out a one-liner into a real spec.

``hermes kanban specify [task_id | --all]`` asks the auxiliary LLM for a
tightened title + concrete body for a Triage task, then flips it
``triage -> todo`` via ``kanban_db.specify_triage_task``.

Mirrors ``hermes_cli/goals.py``: same aux-client pattern, same "empty config
=> skip, don't crash" tolerance. One shot, no retry loop. JSON mode is not
requested (works on providers without it); the parse is lenient and falls
back to "whole reply is the body" so a malformed reply never strands a task.

Two guard-aware behaviours (see ``_ensure_change_plan``): the system prompt
demands the regenerated body end with a literal ``## CHANGE PLAN`` section,
and the written body is then guaranteed to carry that heading — a
coordinator-authored section is re-appended verbatim after regeneration, and
when neither the regenerated nor the old body has one, a deterministic
scaffold is built from the specifier's own ``**Approach**`` bullets. The
board's ``trg_planless_promotion_guard`` greps the stored body for that
heading and rolls the card back to triage without it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

from utils import env_int

HERMES_KANBAN_SPECIFY_MAX_TOKENS = max(1500, env_int("HERMES_KANBAN_SPECIFY_MAX_TOKENS", 6000))

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban triage specifier for the Hermes Agent board.
A user dropped a rough idea into the Triage column. Your job is to turn it
into a concrete, actionable task spec that an autonomous worker can pick up
and execute without further clarification.

Output a single JSON object with exactly two keys:

  {
    "title": "<tightened task title, <= 80 chars, imperative voice>",
    "body":  "<multi-line spec, see structure below>"
  }

The body MUST include these sections, each prefixed with a bold markdown
heading, in this order:

  **Goal** — one sentence, user-facing outcome.
  **Approach** — 2-5 bullets on how a worker should tackle it.
  **Acceptance criteria** — checklist of concrete, verifiable conditions.
  **Out of scope** — short list of things NOT to touch (omit if nothing
      obvious; never invent scope creep).

The body MUST end with a change-plan section that begins with this literal
markdown heading on its own line — byte-exact (two hashes, single spaces,
nothing else on the line):

  ## CHANGE PLAN

followed by four bold sub-parts in this order:

  **Build** — what gets changed: files, modules, commands, config keys.
  **Verify** — concrete commands/checks that prove the change works.
  **Rollback** — how to restore the pre-change state exactly.
  **Window** — when the change may land (e.g. "any time", "off-hours").

This heading is not decoration: the board's promotion guard greps the stored
body for the exact string `## CHANGE PLAN` and refuses to promote a card that
lacks it, bouncing it back to triage.

Rules:
  - Keep the tightened title close in meaning to the original idea — do
    NOT invent a different project.
  - If the original idea is already detailed, preserve its substance and
    just reformat into the sections above.
  - Never add invented requirements the user didn't hint at.
  - The `## CHANGE PLAN` heading must appear literally in the body exactly
    once, and the body must end with that section.
  - If the original body already carries a `## CHANGE PLAN` section, keep
    its Build/Verify/Rollback/Window content intact (reformatting allowed,
    dropping or re-scoping it is not).
  - No preamble, no closing remarks, no code fences around the JSON.
  - Output only the JSON object and nothing else.
"""


_USER_TEMPLATE = """Task id: {task_id}
Current title: {title}
Current body:
{body}
"""


@dataclass
class SpecifyOutcome:
    """Result of specifying a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    new_title: Optional[str] = None


def _truncate(text: str, limit: int) -> str:
    # Plain length clamp for LLM prompt fields; these never reach a terminal, so no escape stripping here.
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_blob(raw: str, fence_re: re.Pattern = _FENCE_RE) -> Optional[dict]:
    """Lenient JSON object extraction: strip code fences, take the first ``{``
    to the last ``}``. None if nothing parses to a dict."""
    if not raw:
        return None
    stripped = fence_re.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        val = json.loads(stripped[first : last + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return val if isinstance(val, dict) else None


def _nonblank(v) -> Optional[str]:
    return v if isinstance(v, str) and v.strip() else None


def _title_body(parsed: dict) -> tuple[Optional[str], Optional[str]]:
    """``(title, body)`` from an LLM reply: title stripped, body verbatim,
    either None when missing/blank."""
    title = _nonblank(parsed.get("title"))
    return (title.strip() if title else None), _nonblank(parsed.get("body"))


# The board's promotion guard (trg_planless_promotion_guard on the live DB)
# refuses todo|triage -> ready unless body LIKE '%## CHANGE PLAN%'. The body
# is regenerated wholesale below, so without _ensure_change_plan the
# heading a coordinator added via `hermes kanban edit` is erased by the very
# command that needs it and the card bounces triage -> ready -> triage.
_CHANGE_PLAN_HEADING_RE = re.compile(r"^[ \t]*##[ \t]+CHANGE PLAN\b", re.MULTILINE)


def _extract_change_plan(body: Optional[str]) -> Optional[str]:
    """The ``## CHANGE PLAN`` section of *body* — literal heading line to the
    end of the body, stripped — or None when there is no such heading."""
    if not body:
        return None
    m = _CHANGE_PLAN_HEADING_RE.search(body)
    if not m:
        return None
    section = body[m.start():].strip()
    return section or None


# The specifier's own section set (see _SYSTEM_PROMPT). The scaffold below is
# built from the specifier's own **Approach** bullets so the card keeps the
# substance the specifier wrote instead of a generic stub.
_APPROACH_SECTION_RE = re.compile(
    r"^\*\*Approach\*\*\s*$", re.IGNORECASE | re.MULTILINE
)

_SCAFFOLD_PLACEHOLDER = "worker must fill in before promote"


def _approach_bullets(body: str) -> list[str]:
    """The specifier's ``**Approach**`` bullets, in order.

    Used as the ``**Build**`` steps of the generated scaffold so the card
    keeps the substance the specifier wrote instead of a generic stub.
    """
    m = _APPROACH_SECTION_RE.search(body or "")
    if not m:
        return []
    # The Approach section runs until the next bold heading.
    tail = body[m.end():]
    nxt = re.search(r"^\*\*[A-Za-z][^*]*\*\*\s*$", tail, re.MULTILINE)
    section = tail[: nxt.start()] if nxt else tail
    bullets = []
    for line in section.splitlines():
        s = line.strip()
        if s.startswith(("- ", "* ")):
            bullets.append(s[2:].strip())
        elif s and not s.startswith("#"):
            # A non-bullet paragraph under Approach is still a step.
            bullets.append(s)
    return [b for b in bullets if b]


def _build_change_plan_scaffold(body: str) -> str:
    """A deterministic ``## CHANGE PLAN`` section built from *body*.

    Used when neither the regenerated body nor the old body carries a plan:
    the aux model can ignore the prompt, and without a scaffold the card
    bounces ``triage -> todo -> ready -> triage`` forever. The scaffold is
    deliberately explicit that a worker still has to fill it in — it makes the
    card *promotable*, not complete.
    """
    steps = _approach_bullets(body)
    if steps:
        build_lines = "\n".join(f"- {s}" for s in steps)
    else:
        build_lines = f"- {_SCAFFOLD_PLACEHOLDER}"
    return (
        "## CHANGE PLAN\n\n"
        f"**Build**\n{build_lines}\n\n"
        f"**Verify** — {_SCAFFOLD_PLACEHOLDER}\n\n"
        f"**Rollback** — {_SCAFFOLD_PLACEHOLDER}\n\n"
        f"**Window** — {_SCAFFOLD_PLACEHOLDER}\n"
    )


def _ensure_change_plan(old_body: Optional[str], new_body: str) -> str:
    """Return *new_body* guaranteed to carry a literal ``## CHANGE PLAN``.

    Resolution order:
      1. *new_body* already has one — leave it alone.
      2. *old_body* has one — re-append it verbatim (coordinator's plan).
      3. Neither — append a deterministic scaffold built from the
         specifier's own **Approach** bullets.
    """
    if _CHANGE_PLAN_HEADING_RE.search(new_body):
        return new_body
    section = _extract_change_plan(old_body)
    if section is None:
        section = _build_change_plan_scaffold(new_body)
    base = new_body.strip()
    if not base:
        return section + "\n"
    return base + "\n\n" + section + "\n"


def _profile_author(default: str = "specifier") -> str:
    """Same identity contract as ``hermes_cli.kanban._profile_author``; ``$USER`` as the last
    resort for a human running the CLI outside any profile."""
    from hermes_cli.profiles import current_profile_name
    return current_profile_name() or os.environ.get("USER") or default


def _load_triage_task(task_id: str) -> tuple[Optional[kb.Task], str]:
    """``(task, "")`` when the task exists and is in triage, else ``(None, reason)``."""
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return None, "unknown task id"
    if task.status != "triage":
        return None, f"task is not in triage (status={task.status!r})"
    return task, ""


def _task_prompt_fields(task: kb.Task) -> dict[str, str]:
    """Bounded ``task_id``/``title``/``body`` for the user prompt templates."""
    return {
        "task_id": task.id,
        "title": _truncate(task.title or "", 400),
        "body": _truncate(task.body or "(no body)", 4000),
    }


def _call_aux(verb: str, task_id: str, *, aux_task: str, system: str, user: str,
              max_tokens: int, timeout: int, log: logging.Logger = logger) -> tuple[Optional[str], str]:
    """One auxiliary LLM call; ``(reply_text, "")`` or ``(None, reason)``.

    ``call_llm`` applies all ``auxiliary.<aux_task>.*`` config (provider/model/
    base_url, extra_body, reasoning_effort, retries). Imported lazily so a
    missing aux client degrades to a skip instead of an import-time crash.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:  # pragma: no cover — import smoke test
        log.debug("%s: auxiliary client import failed: %s", verb, exc)
        return None, "auxiliary client unavailable"
    # Specify/decompose run outside any agent turn (CLI, dashboard route, gateway watcher), so no
    # conversation affinity scope is bound and the relay-affinity headers (x-opencode-session, the
    # OpenRouter/Portal sticky key) are omitted — the OpenCode Go relay rejects that with 400
    # MissingSessionID (#112043). Declare a per-task scope, but only when none is already bound so an
    # in-turn caller keeps its conversation's key.
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
    affinity_token = None if get_affinity_scope() else set_affinity_scope(f"kanban:{task_id}")
    try:
        # Route through call_llm so auxiliary.triage_specifier.* config (provider/model/base_url,
        # extra_body, reasoning_effort, retries) all apply — the direct-create path dropped extra_body
        # (#35566).
        resp = call_llm(
            task=aux_task,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as exc:
        suffix = " — skipping" if verb == "specify" else ""
        log.info("%s: API call failed for %s (%s)%s", verb, task_id, exc, suffix)
        return None, f"LLM error: {type(exc).__name__}"
    finally:
        if affinity_token is not None:
            reset_affinity_scope(affinity_token)
    try:
        return resp.choices[0].message.content or "", ""
    except Exception:
        return "", ""


def specify_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> SpecifyOutcome:
    """Specify one triage task and promote it to ``todo``. Expected failures
    (not in triage, no aux client, API error, malformed reply) surface as
    ``ok=False`` so an ``--all`` sweep continues."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return SpecifyOutcome(task_id, False, reason)

    raw, reason = _call_aux(
        "specify", task_id, aux_task="triage_specifier", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(**_task_prompt_fields(task)),
        max_tokens=HERMES_KANBAN_SPECIFY_MAX_TOKENS, timeout=timeout or 120,
    )
    if raw is None:
        return SpecifyOutcome(task_id, False, reason)
    raw = raw.strip()

    parsed = _extract_json_blob(raw)
    if parsed is None:
        # Whole reply becomes the body; the user can edit afterward.
        if not raw:
            return SpecifyOutcome(task_id, False, "LLM returned an empty response")
        new_title, new_body = None, raw
    else:
        new_title, new_body = _title_body(parsed)
        if new_body is None and new_title is None:
            return SpecifyOutcome(task_id, False, "LLM response missing title and body")

    # Preserve a coordinator-authored ## CHANGE PLAN across regeneration (the
    # promotion guard reads the body we are about to write), and guarantee the
    # guard's invariant even when the aux model ignored the prompt. body=None
    # means "keep the old body", which already carries the plan — nothing to do.
    if new_body is not None:
        new_body = _ensure_change_plan(task.body, new_body)

    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn,
            task_id,
            title=new_title,
            body=new_body,
            author=author or _profile_author(),
        )
    if not ok:
        # Race: promoted/archived between our read and the write.
        return SpecifyOutcome(task_id, False, "task moved out of triage before promotion")
    return SpecifyOutcome(task_id, True, "specified", new_title=new_title)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Task ids in the triage column; ``tenant`` narrows the sweep."""
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(conn, status="triage", tenant=tenant, include_archived=False)
    return [t.id for t in tasks]
