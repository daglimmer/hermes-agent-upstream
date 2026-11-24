"""Respawn-guard failure-text classification — what a persisted error means.

The respawn guard reads ``tasks.last_failure_error`` and asks whether the card
sits behind a wall that retrying will not clear (``blocker_auth``). The text
it reads is not always a diagnosis: the dispatcher writes its own rate-limit
requeue note into that column, and that note names a quota wall while
reporting a run it released ON PURPOSE, without counting a failure.

Classifying text instead of provenance made that note read as a blocker: the
guard searches for ``quota`` / ``rate limit`` in it, so the sentence announcing
that no failure was counted was the thing counted. Once a later run (a block, a
crash) superseded the rate-limited one, the cooldown branch no longer applied
and the stale stamp parked the card for good — a card that was neither
quota-blocked nor awaiting a human, yet deferred forever.

``classify_respawn_blocker`` is the one place that decides what a persisted
failure text implies, so the dispatcher's own bookkeeping can be recognised
for what it is.
"""

from __future__ import annotations

import re
from typing import Optional

# Patterns in a persisted failure text that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
# The auth family is a curated list, not an open `auth\w*` stem: that stem
# also matched ordinary English words like "author"/"authored"/"authoring"/
# "authoritative" in worker progress prose, parking a healthy card forever
# (#117009).
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|"
    r"auth|authenticat(?:e|es|ed|ing|ion)|authoriz(?:e|es|ed|ing|ation)|"
    r"authoris(?:e|es|ed|ing|ion)|authz|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_\-]denied|permission[\s_\-]denied|"
    r"invalid[\s_\-]api[\s_\-]key)\b",
    re.IGNORECASE,
)

# The dispatcher's own rate-limit requeue note (see
# ``kanban_db_dispatch._classify_dead_worker_exit``). It names a quota wall,
# but it reports a run the dispatcher released on purpose, WITHOUT counting a
# failure — the card stays retryable. The guard classifies text, not
# provenance, so this sentence used to read as a quota blocker and park the
# card forever: the cooldown branch only covers the window while the LATEST
# run is still ``rate_limited``, so one later run (a block, a crash) turned
# the stale stamp into a permanent deferral.
#
# The note is written by the dispatcher alone, which is what makes it
# recognisable, and it is stored with the owner's attempt note appended
# (``... without counting a failure [owner admission settled failed]``), so
# the pattern matches the leading sentence and nothing after it.
_RATE_LIMIT_REQUEUE_ERROR_RE = re.compile(
    r"^pid \d+ exited rate-limited \(quota wall\)\s*[—–-]\s*"
    r"requeued without counting a failure",
    re.IGNORECASE,
)


def classify_respawn_blocker(error_text: Optional[str]) -> Optional[str]:
    """The respawn-guard reason a persisted failure text implies, or ``None``.

    ``"blocker_auth"`` for a quota / auth wall that retrying will not clear.
    ``None`` when the text names no wall — including the dispatcher's own
    rate-limit requeue note, which reports a run released WITHOUT counting a
    failure rather than a card stuck behind a wall.
    """
    text = (error_text or "").strip()
    if not text:
        return None
    if _RATE_LIMIT_REQUEUE_ERROR_RE.match(text):
        return None
    if _RESPAWN_BLOCKER_RE.search(text):
        return "blocker_auth"
    return None
