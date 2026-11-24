"""`agentgate approvals`: review and decide queued approvals."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from agentgate.approvals import Approval, ApprovalError, ApprovalStatus
from agentgate.cli._shared import (
    AsOption,
    AuditDbOption,
    DbOption,
    FixturesOption,
    PoliciesOption,
    fail,
    open_cli_world,
    shown_time,
)
from agentgate.domain import Role, User, UserStatus
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    OpenRecord,
    World,
)

approvals_app = typer.Typer(
    help="Review and decide queued approvals. --as names the acting approver.",
    no_args_is_help=True,
)


NoteOption = Annotated[str | None, typer.Option("--note", help="Note stored with the decision.")]


def _approver(world: World, user_id: str) -> User:
    """The acting user, if active and holding an approver role in their customer's policy."""
    user = world.store.get_user(user_id)
    if user is None:
        fail(f"unknown user {user_id!r}")
    if user.status is not UserStatus.ACTIVE:
        fail(f"user {user_id!r} is not active")
    engine = world.engines.get(user.customer_id)
    if engine is None:
        fail(f"customer {user.customer_id!r} has no policy")
    settings = engine.bundle.policy.approvals
    if settings is None:
        fail(f"customer {user.customer_id!r} has no approvals in its policy")
    if user.role not in settings.approver_roles:
        fail(f"role {user.role.value!r} may not review approvals")
    return user


def _json(value: object) -> str:
    # ASCII-only escapes show look-alike letters, control characters and bidi overrides as
    # \u escapes, so the approver reads exactly which characters will run.
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)


@approvals_app.command("list")
def approvals_list(
    as_user: AsOption,
    status: Annotated[
        ApprovalStatus | None, typer.Option("--status", help="Only approvals with this status.")
    ] = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """List the acting approver's customer's approvals, oldest first.

    Read-only: an approval past its expiry is shown as "expired (pending sweep)" and counts
    as expired for --status, but only `approvals expire` marks it."""
    with open_cli_world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = _approver(world, as_user)
        now = world.clock.now()
        approvals = [
            approval
            for approval in world.gate.approvals.list(user.customer_id)
            if status is None or _effective_status(approval, now) is status
        ]
    for approval in approvals:
        typer.echo(
            f"{approval.id}  {_shown_status(approval, now)}  {approval.tool}"
            f"  {approval.requested_by}  risk {approval.risk_score}"
            f"  created {shown_time(approval.created_at)}"
            f"  expires {shown_time(approval.expires_at)}"
        )
    typer.echo(f"{user.customer_id}  {len(approvals)} approvals")


@approvals_app.command("show")
def approvals_show(
    approval_id: Annotated[str, typer.Argument(metavar="ID", help="Approval id.")],
    as_user: AsOption,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Show every field of one approval, including the exact args that would run.

    Read-only, like list: an overdue approval shows as "expired (pending sweep)"."""
    with open_cli_world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = _approver(world, as_user)
        now = world.clock.now()
        approval = world.gate.approvals.get(approval_id)
    # Another customer's approval reads exactly like a missing one.
    if approval is None or approval.customer_id != user.customer_id:
        fail("approval not found")
    for line in _describe(approval, now):
        typer.echo(line)


# Statuses an approval leaves when its expiry passes (ApprovalQueue.expire_due).
_EXPIRING = frozenset({ApprovalStatus.PENDING, ApprovalStatus.APPROVED})


PENDING_SWEEP = "expired (pending sweep)"


def _effective_status(approval: Approval, now: datetime) -> ApprovalStatus:
    """The status the approval has in effect: expired once past its expiry, swept or not.
    The queue refuses to decide or run an overdue approval either way."""
    if approval.status in _EXPIRING and now >= approval.expires_at:
        return ApprovalStatus.EXPIRED
    return approval.status


def _shown_status(approval: Approval, now: datetime) -> str:
    effective = _effective_status(approval, now)
    if effective is not approval.status:
        return PENDING_SWEEP
    return approval.status.value


def _describe(approval: Approval, now: datetime) -> list[str]:
    fields = [
        ("id", approval.id),
        ("customer", approval.customer_id),
        ("status", _shown_status(approval, now)),
        ("tool", approval.tool),
        ("action", approval.action.value),
        ("requested_by", approval.requested_by),
        ("run_id", approval.run_id),
        ("risk", str(approval.risk_score)),
        ("rule_ids", ", ".join(approval.rule_ids) or "-"),
        ("policy_version", approval.policy_version),
        ("args_hash", approval.args_hash),
        ("created_at", shown_time(approval.created_at)),
        ("expires_at", shown_time(approval.expires_at)),
        ("decided_by", approval.decided_by or "-"),
        ("decided_at", shown_time(approval.decided_at)),
        # Free text from an approver, so quoted and escaped like the args.
        ("note", "-" if approval.note is None else _json(approval.note)),
        ("executed_at", shown_time(approval.executed_at)),
    ]
    width = max(len(name) for name, _ in fields)
    lines = [f"{name.ljust(width)}  {value}" for name, value in fields]
    return [*lines, "args", _json(approval.args)]


@approvals_app.command("approve")
def approvals_approve(
    approval_id: Annotated[str, typer.Argument(metavar="ID", help="Approval id.")],
    as_user: AsOption,
    note: NoteOption = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Approve a pending approval. The requester's run may then execute it once."""
    _decide(approval_id, as_user, note, db, audit_db, policies, fixtures, approve=True)


@approvals_app.command("reject")
def approvals_reject(
    approval_id: Annotated[str, typer.Argument(metavar="ID", help="Approval id.")],
    as_user: AsOption,
    note: NoteOption = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Reject a pending approval. It can never run."""
    _decide(approval_id, as_user, note, db, audit_db, policies, fixtures, approve=False)


def _decide(
    approval_id: str,
    as_user: str,
    note: str | None,
    db: Path,
    audit_db: Path,
    policies: Path,
    fixtures: Path,
    *,
    approve: bool,
) -> None:
    with open_cli_world(db, audit_db, policies, fixtures) as world:
        # The gate checks the approver itself and records the decision or the refusal.
        try:
            if approve:
                decided = world.gate.approve(approval_id, as_user, note)
            else:
                decided = world.gate.reject(approval_id, as_user, note)
        except ApprovalError as err:
            fail(f"{err.reason.value}: {err}")
    typer.echo(f"{'approved' if approve else 'rejected'} {decided.id}")


@approvals_app.command("expire")
def approvals_expire(
    as_user: AsOption,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Mark every approval past its expiry as expired, for all customers.

    An operator command: the acting user must be an active owner of some customer. Only the
    count is printed, since the expired approvals may belong to other customers.
    """
    with open_cli_world(db, audit_db, policies, fixtures) as world:
        user = _approver(world, as_user)
        if user.role is not Role.OWNER:
            fail(f"role {user.role.value!r} may not expire approvals; owners only")
        expired = world.gate.expire_due()
    typer.echo(f"expired {len(expired)} approvals")
