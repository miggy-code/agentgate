"""`agentgate spend`: show the spend ledger and release stale reservations."""

from datetime import timedelta
from typing import Annotated

import typer

from agentgate.cli._shared import (
    AsOption,
    AuditDbOption,
    DbOption,
    FixturesOption,
    PoliciesOption,
    active_user,
    fail,
    may_manage,
    open_cli_world,
)
from agentgate.domain import Role, User, UserStatus
from agentgate.model.tokens import (
    TokenStore,
)
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    DEFAULT_RESERVATION_LEASE,
    OpenRecord,
    World,
)

spend_app = typer.Typer(help="Operate the spend ledger.", no_args_is_help=True)


def _owner(world: World, user_id: str) -> User:
    """The acting user, if active and an owner of their customer."""
    user = world.store.get_user(user_id)
    if user is None:
        fail(f"unknown user {user_id!r}")
    if user.status is not UserStatus.ACTIVE:
        fail(f"user {user_id!r} is not active")
    if user.role is not Role.OWNER:
        fail(f"role {user.role.value!r} may not do this; owners only")
    return user


@spend_app.command("show")
def spend_show(
    as_user: AsOption,
    run: Annotated[str, typer.Option("--run", metavar="RUN_ID", help="The run to show.")],
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Show what a run has spent: USD (settled plus still reserved), calls counted against
    its cap, and its reservations by status. For a run of the acting user, or of their
    customer if they own it; another customer's run reads like a missing one."""
    with open_cli_world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = active_user(world, as_user)
        found = TokenStore(world.store, world.clock).run(run)
        if not may_manage(user, found):
            fail(f"run {run!r} not found")
        ledger = world.gate.ledger
        with world.store.locked():
            rows = world.store.connection.execute(
                "SELECT status, COUNT(*) AS n FROM spend_reservations WHERE run_id = ?"
                " GROUP BY status",
                (run,),
            ).fetchall()
        counts = {str(row["status"]): int(row["n"]) for row in rows}
        spent, calls = ledger.run_spent(run), ledger.run_calls(run)
    typer.echo(f"run {run}")
    typer.echo(f"spent_usd {spent}")
    typer.echo(f"calls {calls}")
    for status in ("open", "settled", "released"):
        typer.echo(f"{status} {counts.get(status, 0)}")


@spend_app.command("release-stale")
def spend_release_stale(
    as_user: AsOption,
    older_than: Annotated[
        int,
        typer.Option(
            "--older-than",
            metavar="MINUTES",
            min=1,
            help="Release reservations still open this many minutes after they were made.",
        ),
    ] = int(DEFAULT_RESERVATION_LEASE.total_seconds() // 60),
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Release spend reservations left open by hosts that died mid-call, for all customers.

    An operator command: the acting user must be an active owner of some customer. Only the
    count is printed, since the reservations may belong to other customers. Every open
    already releases reservations older than 15 minutes; this one takes another age. A host
    that was only slow and settles later still has its actual cost recorded. The audit log
    of each customer whose reservations were released records how many, their total and
    their runs; the acting owner is named only in their own customer's log."""
    # The automatic sweep is off for this open, so the count covers everything released.
    with open_cli_world(db, audit_db, policies, fixtures, reservation_lease=None) as world:
        owner = _owner(world, as_user)
        released = world.gate.release_stale(timedelta(minutes=older_than), actor=owner)
    typer.echo(f"released {released} stale reservations older than {older_than} minutes")
