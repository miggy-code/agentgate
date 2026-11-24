"""`agentgate users`: list, suspend and reinstate users."""

from typing import Annotated

import typer

from agentgate.cli._shared import (
    AsOption,
    AuditDbOption,
    DbOption,
    FixturesOption,
    PoliciesOption,
    fail,
    open_cli_world,
)
from agentgate.domain import UserStatus
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    OpenRecord,
    OperatorReason,
    UserAdminError,
    reinstate_user,
    suspend_user,
)

users_app = typer.Typer(
    help="List, suspend and reinstate users. --as names the acting user.", no_args_is_help=True
)


@users_app.command("list")
def users_list(
    as_user: AsOption,
    customer: Annotated[
        str | None,
        typer.Option(
            "--customer", metavar="ID", help="Must be the acting user's own customer, if given."
        ),
    ] = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """List the acting user's customer's users: id, role and status."""
    with open_cli_world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = world.store.get_user(as_user)
        if user is None:
            fail(f"unknown user {as_user!r}")
        if user.status is not UserStatus.ACTIVE:
            fail(f"user {as_user!r} is not active")
        # Another customer reads exactly like a missing one.
        if customer is not None and customer != user.customer_id:
            fail(f"customer {customer!r} not found")
        users = world.store.list_users(user.customer_id)
        by_operator = {u.id for u in users if world.store.operator_suspended(u.id)}
    for listed in users:
        status = listed.status.value
        if listed.id in by_operator:
            status += " (by operator)"
        typer.echo(f"{listed.id}  {listed.role.value}  {status}")
    typer.echo(f"{user.customer_id}  {len(users)} users")


UserIdArgument = Annotated[str, typer.Argument(metavar="USER_ID", help="The user to change.")]


ReasonOption = Annotated[
    OperatorReason,
    typer.Option("--reason", help="Why, as a code: it is recorded in the audit log."),
]


@users_app.command("suspend")
def users_suspend(
    user_id: UserIdArgument,
    as_user: AsOption,
    reason: ReasonOption = OperatorReason.UNSPECIFIED,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Suspend a user of the acting owner's customer, from their next gate call on.

    Live runs are refused on their next call. The suspension is sticky: fixture sync does
    not reactivate the user, whatever the fixtures say, until `users reinstate`. Nothing is
    written to the fixtures. The audit log records who suspended whom, and the reason code."""
    with open_cli_world(db, audit_db, policies, fixtures) as world:
        try:
            suspended = suspend_user(world, user_id, acting_user_id=as_user, reason=reason)
        except UserAdminError as exc:
            fail(str(exc))
    typer.echo(f"suspended {suspended.id}")


@users_app.command("reinstate")
def users_reinstate(
    user_id: UserIdArgument,
    as_user: AsOption,
    reason: ReasonOption = OperatorReason.UNSPECIFIED,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Lift a suspension made with `users suspend`. The user's status is then what the
    fixtures say, so a user the fixtures suspend stays suspended. Recorded like a
    suspension."""
    with open_cli_world(db, audit_db, policies, fixtures) as world:
        try:
            reinstated = reinstate_user(world, user_id, acting_user_id=as_user, reason=reason)
        except UserAdminError as exc:
            fail(str(exc))
    typer.echo(f"reinstated {reinstated.id}  status {reinstated.status.value}")
