"""The reasons a turn's owner can end it with, as durable values.

Kept in their own light module so both sides can read them without importing
``gateway.run`` (which boots the whole runtime on import): the gateway records one of
them on the turn it stopped, and the Kanban board reads the same value back out of the
owner's ledger to tell an owner-caused interruption from a user's Stop.

``INTERRUPT_REASON_GATEWAY_SHUTDOWN`` and ``INTERRUPT_REASON_GATEWAY_RESTART`` are the
owner's OWN reasons: the gateway is draining its active turns (a shutdown) or restarting
(a reload). Every other reason — a user's Stop, a session reset, an inactivity timeout —
is a decision about that turn, not a host bounce.
"""

# The gateway is shutting down: it drains the active turns and exits.
INTERRUPT_REASON_GATEWAY_SHUTDOWN = "Gateway shutting down"

# The gateway is restarting (a reload): the same drain, then the supervisor relaunches.
INTERRUPT_REASON_GATEWAY_RESTART = "Gateway restarting"

# The owner's own reasons: a turn ended by one of these died with the host, not with the
# card's work.
OWNER_SHUTDOWN_REASONS = frozenset({INTERRUPT_REASON_GATEWAY_SHUTDOWN, INTERRUPT_REASON_GATEWAY_RESTART})


def is_owner_shutdown(cause) -> bool:
    """True when the recorded reason names the gateway's own shutdown or restart."""
    return bool(cause) and cause in OWNER_SHUTDOWN_REASONS
