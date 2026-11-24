"""What the command modules share: common options, opening a world, and how a command fails."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from agentgate.context import RunContext
from agentgate.domain import Role, User, UserStatus
from agentgate.world import (
    DEFAULT_RESERVATION_LEASE,
    OpenRecord,
    World,
    open_world,
)

PoliciesOption = Annotated[
    Path, typer.Option("--policies", help="Directory holding one folder per customer policy.")
]


FixturesOption = Annotated[
    Path, typer.Option("--fixtures", help="Directory holding the fixture customers.")
]


def fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


DbOption = Annotated[
    Path, typer.Option("--db", help="agentgate database (created and seeded if missing).")
]


AuditDbOption = Annotated[Path, typer.Option("--audit-db", help="Panoptes audit database.")]


AsOption = Annotated[
    str,
    typer.Option(
        "--as",
        metavar="USER_ID",
        help="Acting user id. A demo identity: agentgate does not authenticate it.",
    ),
]


DEMO_IDENTITY_NOTE = "note: --as is a demo identity; agentgate does not authenticate users"


def database_error(exc: sqlite3.Error) -> str:
    # sqlite3 messages name the problem ("database is locked", "attempt to write a readonly
    # database") but not the data, so they are safe to print.
    return f"database error: {exc}"


@contextmanager
def open_cli_world(
    db: Path,
    audit_db: Path,
    policies: Path,
    fixtures: Path,
    *,
    reservation_lease: timedelta | None = DEFAULT_RESERVATION_LEASE,
    record_open: OpenRecord = OpenRecord.ALL,
    identity_note: str = DEMO_IDENTITY_NOTE,
) -> Iterator[World]:
    # Printed before anything else, so even a failing command says it once.
    typer.echo(identity_note, err=True)
    try:
        world = open_world(
            db_path=db,
            audit_path=audit_db,
            policies=policies,
            fixtures=fixtures,
            reservation_lease=reservation_lease,
            record_open=record_open,
        )
    except sqlite3.Error as exc:
        fail(database_error(exc))
    except (OSError, ValueError, RuntimeError) as exc:
        fail(str(exc))
    try:
        report_open(world, db)
        yield world
    except sqlite3.Error as exc:
        fail(database_error(exc))
    finally:
        world.close()


def report_open(world: World, db: Path) -> None:
    """Say on stderr what opening the world changed, if anything."""
    before = world.store.tightened_from
    if before is not None:
        typer.echo(
            f"warning: {db} was open to other users (mode {before:04o});"
            f" tightened to {before & 0o600:04o}",
            err=True,
        )
    synced = world.fixture_sync
    if synced is not None and synced.changed:
        for line in synced.lines():
            typer.echo(f"fixtures: {line}", err=True)
    if world.stale_released:
        typer.echo(
            f"note: released {world.stale_released} stale spend reservations",
            err=True,
        )


def shown_time(value: datetime | None) -> str:
    return "-" if value is None else value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def active_user(world: World, user_id: str) -> User:
    user = world.store.get_user(user_id)
    if user is None:
        fail(f"unknown user {user_id!r}")
    if user.status is not UserStatus.ACTIVE:
        fail(f"user {user_id!r} is not active")
    return user


def may_manage(user: User, run: RunContext | None) -> bool:
    """Whether the acting user may manage tokens of this run: their own run, or any run of
    their customer if they own it. Another customer's run reads like a missing one."""
    if run is None or run.customer_id != user.customer_id:
        return False
    return run.user_id == user.id or user.role is Role.OWNER
