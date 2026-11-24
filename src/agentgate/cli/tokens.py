"""`agentgate token`: issue, list and revoke model-proxy run tokens."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from agentgate.audit.base import (
    OperatorAction,
)
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
    shown_time,
)
from agentgate.domain import User
from agentgate.model.tokens import (
    DEFAULT_TTL,
    MAX_TTL,
    TokenError,
    TokenRecord,
    TokenStore,
    write_token_file,
)
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    OpenRecord,
    World,
)

token_app = typer.Typer(
    help="Issue, list and revoke run tokens for the model gateway. --as names the acting user.",
    no_args_is_help=True,
)


def _record_token(world: World, user: User, action: OperatorAction, token: TokenRecord) -> None:
    world.gate.record_token_event(action, token, customer_id=user.customer_id, actor=user)


def _token_line(record: TokenRecord, now: datetime) -> str:
    if record.revoked_at is not None:
        status = f"revoked ({record.revoked_reason})"
    elif now >= record.expires_at:
        status = "expired"
    else:
        status = "live"
    return (
        f"{record.token_id}  run {record.run_id}  {status}"
        f"  issued {shown_time(record.issued_at)}  expires {shown_time(record.expires_at)}"
    )


@token_app.command("issue")
def token_issue(
    as_user: AsOption,
    run: Annotated[str, typer.Option("--run", metavar="RUN_ID", help="The run the token is for.")],
    ttl_minutes: Annotated[
        int,
        typer.Option(
            "--ttl-minutes",
            min=1,
            max=int(MAX_TTL.total_seconds() // 60),
            help="How long the token lives, at most. It also dies with its run.",
        ),
    ] = int(DEFAULT_TTL.total_seconds() // 60),
    token_file: Annotated[
        Path | None,
        typer.Option(
            "--token-file",
            help="Write the token to this new file (mode 0600) instead of printing it.",
        ),
    ] = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Issue a token for a run of the acting user, or of their customer if they own it.

    The token is shown once, on stdout or in --token-file; only its hash is stored. The issue
    is recorded in the audit log, and a token whose issue cannot be recorded does not exist."""
    with open_cli_world(db, audit_db, policies, fixtures) as world:
        user = active_user(world, as_user)
        tokens = TokenStore(world.store, world.clock)
        found = tokens.run(run)
        if not may_manage(user, found):
            fail(f"run {run!r} not found")

        try:
            issued = tokens.issue(
                run,
                timedelta(minutes=ttl_minutes),
                record=lambda made: _record_token(world, user, OperatorAction.TOKEN_ISSUED, made),
            )
        except TokenError as exc:
            fail(str(exc))
        if token_file is not None:
            try:
                write_token_file(token_file, issued.token)
            except OSError as exc:
                # Nothing holds the token, so it must not stay live.
                tokens.revoke(
                    issued.record.token_id,
                    record=lambda gone: _record_token(
                        world, user, OperatorAction.TOKEN_REVOKED, gone
                    ),
                )
                fail(f"cannot write {token_file}: {exc.strerror}")
    typer.echo(
        f"issued {issued.record.token_id} for run {run},"
        f" expires {shown_time(issued.record.expires_at)}",
        err=token_file is None,
    )
    if token_file is None:
        typer.echo(issued.token)
    else:
        typer.echo(f"token written to {token_file}")


@token_app.command("revoke")
def token_revoke(
    token_id: Annotated[str, typer.Argument(metavar="TOKEN_ID", help="The token to revoke.")],
    as_user: AsOption,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Revoke a live token of a run of the acting user, or of their customer if they own it."""
    with open_cli_world(db, audit_db, policies, fixtures) as world:
        user = active_user(world, as_user)
        tokens = TokenStore(world.store, world.clock)
        record = tokens.get(token_id)
        if record is None or not may_manage(user, tokens.run(record.run_id)):
            fail(f"token {token_id!r} not found")

        revoked = tokens.revoke(
            token_id,
            record=lambda gone: _record_token(world, user, OperatorAction.TOKEN_REVOKED, gone),
        )
        if not revoked:
            fail(f"token {token_id!r} is already revoked")
    typer.echo(f"revoked {token_id}")


@token_app.command("list")
def token_list(
    as_user: AsOption,
    run: Annotated[
        str | None, typer.Option("--run", metavar="RUN_ID", help="Only this run's tokens.")
    ] = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """List the tokens of the acting user's runs, or of every run of their customer if they
    own it: id, run, status and times. Never the token itself."""
    with open_cli_world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = active_user(world, as_user)
        tokens = TokenStore(world.store, world.clock)
        now = world.clock.now()
        shown: dict[str, bool] = {}
        lines: list[str] = []
        for record in tokens.list_tokens(run):
            if record.run_id not in shown:
                shown[record.run_id] = may_manage(user, tokens.run(record.run_id))
            if shown[record.run_id]:
                lines.append(_token_line(record, now))
    for line in lines:
        typer.echo(line)
    typer.echo(f"{len(lines)} tokens")
