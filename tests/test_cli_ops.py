"""Operator commands and path safety: `agentgate users`, `agentgate spend release-stale`, and
how every world-opening command handles a wrong path, a read-only, locked or failing
database, a world-readable store and fixtures that changed under an existing database.
"""

import shutil
import sqlite3
import stat
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

import agentgate.gateway
from agentgate.audit.base import MemorySink
from agentgate.audit.panoptes import PanoptesSink
from agentgate.cli import DEMO_IDENTITY_NOTE, app
from agentgate.clock import FixedClock
from agentgate.gate import Denied, Executed, Gate
from agentgate.models import Action, UserStatus
from agentgate.spend import Reservation
from agentgate.store import Store
from agentgate.world import World, open_world
from conftest import FIXTURES, JR, KC, POLICIES

runner = CliRunner()
READ = {"document_id": "kc-service-menu"}


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def _db(tmp: Path) -> Path:
    return tmp / "data" / "agentgate.db"


def _audit(tmp: Path) -> Path:
    return tmp / "data" / "audit.db"


def _paths(
    tmp: Path, *, db: Path | None = None, audit: Path | None = None, fixtures: Path = FIXTURES
) -> list[str]:
    return [
        "--db",
        str(_db(tmp) if db is None else db),
        "--audit-db",
        str(_audit(tmp) if audit is None else audit),
        "--policies",
        str(POLICIES),
        "--fixtures",
        str(fixtures),
    ]


def _cli(tmp: Path, *args: str, **paths: Path) -> Result:
    result = runner.invoke(app, [*args, *_paths(tmp, **paths)])
    assert result.stderr.startswith(DEMO_IDENTITY_NOTE + "\n"), result.stderr
    assert "Traceback" not in result.output
    return result


def _ok(result: Result) -> Result:
    assert result.exit_code == 0, result.output
    return result


def _error(result: Result) -> str:
    """The one error line on stderr after the demo-identity note; nothing on stdout."""
    assert result.exit_code == 1, result.output
    # Exited through typer, not an exception the runner caught.
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.stdout == ""
    rest = result.stderr.removeprefix(DEMO_IDENTITY_NOTE + "\n")
    assert rest.count("\n") == 1 and rest.startswith("error: "), rest
    return rest


def _world(tmp: Path, *, clock: FixedClock | None = None, fixtures: Path = FIXTURES) -> World:
    return open_world(
        db_path=_db(tmp),
        audit_path=_audit(tmp),
        policies=POLICIES,
        fixtures=fixtures,
        clock=clock,
        sink=MemorySink(),
    )


# users list


def test_users_list_shows_the_acting_users_customer(tmp: Path) -> None:
    result = _ok(_cli(tmp, "users", "list", "--as", "kc-staff-nadine"))
    lines = result.stdout.splitlines()
    assert lines[0] == "kc-contractor-hollis  contractor  active"
    assert "kc-staff-gavin  staff  suspended" in lines
    assert lines[-1] == f"{KC}  8 users"
    assert not any(line.startswith("jr-") for line in lines)
    assert result.stderr == DEMO_IDENTITY_NOTE + "\n"

    mine = _ok(_cli(tmp, "users", "list", "--as", "kc-staff-nadine", "--customer", KC))
    assert mine.stdout == result.stdout


@pytest.mark.parametrize(
    ("user", "extra", "message"),
    [
        ("kc-staff-nadine", ["--customer", JR], f"error: customer {JR!r} not found\n"),
        ("kc-staff-nadine", ["--customer", "nope"], "error: customer 'nope' not found\n"),
        ("kc-staff-gavin", [], "error: user 'kc-staff-gavin' is not active\n"),
        ("nobody", [], "error: unknown user 'nobody'\n"),
    ],
)
def test_users_list_refusals(tmp: Path, user: str, extra: list[str], message: str) -> None:
    assert _error(_cli(tmp, "users", "list", "--as", user, *extra)) == message


# users suspend and reinstate


@pytest.mark.parametrize(
    ("target", "acting", "message"),
    [
        ("kc-staff-colby", "kc-mgr-marisol", "role 'manager' may not change users; owners only"),
        ("kc-staff-colby", "jr-owner-marisol", "unknown user 'kc-staff-colby'"),
        ("nobody", "kc-owner-dale", "unknown user 'nobody'"),
        ("kc-owner-dale", "kc-owner-dale", "an owner cannot suspend themselves"),
        ("kc-staff-colby", "kc-staff-gavin", "user 'kc-staff-gavin' is not active"),
    ],
)
def test_users_suspend_refusals(tmp: Path, target: str, acting: str, message: str) -> None:
    assert _error(_cli(tmp, "users", "suspend", target, "--as", acting)) == f"error: {message}\n"
    with Store(_db(tmp)) as store:
        assert not store.operator_suspended(target)


def test_a_cli_suspension_stops_a_live_run_on_its_next_call(tmp: Path) -> None:
    with _world(tmp) as world:
        live = world.gate.start_run(KC, "kc-staff-nadine")
        assert isinstance(world.gate.call_tool(live, "docs.read", Action.READ, READ), Executed)

        result = _ok(_cli(tmp, "users", "suspend", "kc-staff-nadine", "--as", "kc-owner-dale"))
        assert result.stdout == "suspended kc-staff-nadine\n"

        # The live run's own connection sees it on the next call.
        denied = world.gate.call_tool(live, "docs.read", Action.READ, READ)
        assert denied == Denied(reason="suspended_user")

    listed = _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale")).stdout.splitlines()
    assert "kc-staff-nadine  staff  suspended (by operator)" in listed
    # Sticky: every open syncs the fixtures, which say active, and she stays suspended.
    with _world(tmp) as world:
        user = world.store.get_user("kc-staff-nadine")
        assert user is not None and user.status is UserStatus.SUSPENDED

    back = _ok(_cli(tmp, "users", "reinstate", "kc-staff-nadine", "--as", "kc-owner-dale"))
    assert back.stdout == "reinstated kc-staff-nadine  status active\n"
    again = _cli(tmp, "users", "reinstate", "kc-staff-nadine", "--as", "kc-owner-dale")
    assert _error(again) == ("error: user 'kc-staff-nadine' was not suspended by an operator\n")


def test_reinstate_leaves_a_fixture_suspension_in_place(tmp: Path) -> None:
    fixtures = tmp / "fixtures"
    shutil.copytree(FIXTURES, fixtures)
    _ok(_cli(tmp, "users", "suspend", "kc-staff-colby", "--as", "kc-owner-dale", fixtures=fixtures))
    users = fixtures / KC / "users.yaml"
    users.write_text(
        users.read_text().replace(
            "colby@kettlecreek.example\n  role: staff\n  status: active",
            "colby@kettlecreek.example\n  role: staff\n  status: suspended",
        )
    )
    back = _ok(
        _cli(
            tmp, "users", "reinstate", "kc-staff-colby", "--as", "kc-owner-dale", fixtures=fixtures
        )
    )
    assert back.stdout == "reinstated kc-staff-colby  status suspended\n"


# fixture sync on the CLI


def test_fixture_changes_are_applied_and_reported_on_stderr(tmp: Path) -> None:
    fixtures = tmp / "fixtures"
    shutil.copytree(FIXTURES, fixtures)
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale", fixtures=fixtures))
    users = fixtures / KC / "users.yaml"
    users.write_text(
        users.read_text().replace(
            "nadine@kettlecreek.example\n  role: staff\n  status: active",
            "nadine@kettlecreek.example\n  role: manager\n  status: suspended",
        )
    )
    result = _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale", fixtures=fixtures))
    assert "kc-staff-nadine  manager  suspended" in result.stdout.splitlines()
    assert result.stderr == (
        f"{DEMO_IDENTITY_NOTE}\nfixtures: users updated: kc-staff-nadine (role, status)\n"
    )
    # Applied once: the next open has nothing to say.
    quiet = _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale", fixtures=fixtures))
    assert quiet.stderr == DEMO_IDENTITY_NOTE + "\n"


# spend release-stale


def _leak(tmp: Path, count: int, minutes_ago: int) -> None:
    """count runs that reserved a model call minutes_ago and never settled it."""
    clock = FixedClock(datetime.now(UTC) - timedelta(minutes=minutes_ago))
    with _world(tmp, clock=clock) as world:
        for _ in range(count):
            ctx = world.gate.start_run(KC, "kc-staff-nadine")
            assert isinstance(world.gate.charge_model(ctx, Decimal("0.50")), Reservation)


def _open_reservations(tmp: Path) -> int:
    conn = sqlite3.connect(_db(tmp))
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM spend_reservations WHERE status = 'open'"
        ).fetchone()
    finally:
        conn.close()
    return int(count)


def test_spend_release_stale_is_for_owners(tmp: Path) -> None:
    _leak(tmp, 2, minutes_ago=30)
    refused = _cli(tmp, "spend", "release-stale", "--as", "kc-mgr-marisol")
    assert _error(refused) == "error: role 'manager' may not do this; owners only\n"
    assert _open_reservations(tmp) == 2


def test_spend_release_stale_takes_an_age(tmp: Path) -> None:
    _leak(tmp, 3, minutes_ago=30)
    kept = _ok(
        _cli(tmp, "spend", "release-stale", "--as", "jr-owner-marisol", "--older-than", "60")
    )
    assert kept.stdout == "released 0 stale reservations older than 60 minutes\n"
    assert _open_reservations(tmp) == 3
    # An owner of any customer may run it; only the count is printed.
    swept = _ok(
        _cli(tmp, "spend", "release-stale", "--as", "jr-owner-marisol", "--older-than", "20")
    )
    assert swept.stdout == "released 3 stale reservations older than 20 minutes\n"
    assert swept.stderr == DEMO_IDENTITY_NOTE + "\n"
    assert _open_reservations(tmp) == 0


def test_spend_release_stale_defaults_to_the_lease(tmp: Path) -> None:
    _leak(tmp, 1, minutes_ago=16)
    result = _ok(_cli(tmp, "spend", "release-stale", "--as", "kc-owner-dale"))
    assert result.stdout == "released 1 stale reservations older than 15 minutes\n"


def test_spend_release_stale_rejects_a_zero_age(tmp: Path) -> None:
    result = runner.invoke(
        app, ["spend", "release-stale", "--as", "kc-owner-dale", "--older-than", "0", *_paths(tmp)]
    )
    assert result.exit_code == 2


def test_any_open_reclaims_past_the_lease_and_says_so(tmp: Path) -> None:
    _leak(tmp, 2, minutes_ago=16)
    result = _ok(_cli(tmp, "users", "list", "--as", "kc-staff-nadine"))
    assert result.stderr == (f"{DEMO_IDENTITY_NOTE}\nnote: released 2 stale spend reservations\n")
    assert _open_reservations(tmp) == 0


# Store file permissions


def test_a_world_readable_store_is_tightened_with_one_warning(tmp: Path) -> None:
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    assert stat.S_IMODE(_db(tmp).stat().st_mode) == 0o600
    _db(tmp).chmod(0o644)
    result = _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    assert result.stderr == (
        f"{DEMO_IDENTITY_NOTE}\nwarning: {_db(tmp)} was open to other users (mode 0644);"
        " tightened to 0600\n"
    )
    assert stat.S_IMODE(_db(tmp).stat().st_mode) == 0o600
    again = _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    assert again.stderr == DEMO_IDENTITY_NOTE + "\n"


# Path safety, for every command that opens a world on disk

WORLD_COMMANDS = [
    ["approvals", "list", "--as", "kc-owner-dale"],
    ["approvals", "show", "apr_x", "--as", "kc-owner-dale"],
    ["approvals", "approve", "apr_x", "--as", "kc-owner-dale"],
    ["approvals", "expire", "--as", "kc-owner-dale"],
    ["users", "list", "--as", "kc-owner-dale"],
    ["users", "suspend", "kc-staff-colby", "--as", "kc-owner-dale"],
    ["spend", "release-stale", "--as", "kc-owner-dale"],
    ["gateway", "serve", "--as", "kc-staff-nadine"],
]


@pytest.mark.parametrize("command", WORLD_COMMANDS, ids=lambda c: " ".join(c[:2]))
def test_db_and_audit_db_must_differ(tmp: Path, command: list[str]) -> None:
    same = tmp / "data" / "same.db"
    result = _cli(tmp, *command, db=same, audit=same)
    assert _error(result) == (
        f"error: {same}: the store and the audit log must be different files\n"
    )
    assert not same.exists()


@pytest.mark.parametrize("command", WORLD_COMMANDS, ids=lambda c: " ".join(c[:2]))
def test_a_db_that_is_the_audit_log_is_refused(tmp: Path, command: list[str]) -> None:
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    audit = _audit(tmp)
    before = audit.read_bytes()
    result = _cli(tmp, *command, db=audit, audit=tmp / "other.db")
    assert _error(result) == (
        f"error: {audit}: is a Panoptes audit database, not an agentgate store\n"
    )
    assert audit.read_bytes() == before
    assert not (tmp / "other.db").exists()
    verify = runner.invoke(app, ["audit", "verify", "--db", str(audit)])
    assert verify.exit_code == 0 and verify.stdout.endswith("chain intact\n")


@pytest.mark.parametrize("command", WORLD_COMMANDS, ids=lambda c: " ".join(c[:2]))
def test_an_audit_db_that_is_the_store_is_refused(tmp: Path, command: list[str]) -> None:
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    store = _db(tmp)
    result = _cli(tmp, *command, db=tmp / "new.db", audit=store)
    assert _error(result) == (
        f"error: {store}: is an agentgate store, not a Panoptes audit database\n"
    )
    assert not (tmp / "new.db").exists()


def test_audit_verify_refuses_a_store(tmp: Path) -> None:
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    result = runner.invoke(app, ["audit", "verify", "--db", str(_db(tmp))])
    assert result.exit_code == 1
    assert result.stderr == f"error: not a Panoptes audit database: {_db(tmp)}\n"


@pytest.fixture
def read_only_store(tmp: Path) -> Iterator[Path]:
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    _db(tmp).chmod(0o400)
    try:
        yield tmp
    finally:
        _db(tmp).chmod(0o600)


def test_a_read_only_store_serves_reads(read_only_store: Path) -> None:
    # Reading needs no write, so list and show work on a read-only copy.
    result = _ok(_cli(read_only_store, "approvals", "list", "--as", "kc-owner-dale"))
    assert result.stdout == f"{KC}  0 approvals\n"
    _ok(_cli(read_only_store, "users", "list", "--as", "kc-owner-dale"))


@pytest.mark.parametrize(
    "command",
    [
        ["users", "suspend", "kc-staff-colby", "--as", "kc-owner-dale"],
        ["gateway", "serve", "--as", "kc-staff-nadine"],
    ],
    ids=["users suspend", "gateway serve"],
)
def test_a_write_to_a_read_only_store_is_one_error_line(
    read_only_store: Path, command: list[str]
) -> None:
    result = _cli(read_only_store, *command)
    assert _error(result) == "error: database error: attempt to write a readonly database\n"


def test_a_read_only_store_with_stale_spend_is_one_error_line(tmp: Path) -> None:
    _leak(tmp, 1, minutes_ago=16)
    _db(tmp).chmod(0o400)
    try:
        result = _cli(tmp, "approvals", "list", "--as", "kc-owner-dale")
    finally:
        _db(tmp).chmod(0o600)
    assert _error(result) == "error: database error: attempt to write a readonly database\n"


def test_a_directory_as_db_is_one_error_line(tmp: Path) -> None:
    (tmp / "dir.db").mkdir()
    result = _cli(tmp, "users", "list", "--as", "kc-owner-dale", db=tmp / "dir.db")
    assert _error(result).startswith("error: database error: ")


def test_a_corrupt_store_is_one_error_line(tmp: Path) -> None:
    corrupt = tmp / "corrupt.db"
    corrupt.write_bytes(b"\x00not sqlite\x00" * 200)
    result = _cli(tmp, "approvals", "list", "--as", "kc-owner-dale", db=corrupt)
    assert _error(result) == "error: database error: file is not a database\n"


def test_a_locked_store_is_one_error_line(tmp: Path) -> None:
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    holder = sqlite3.connect(_db(tmp), isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        result = _cli(tmp, "approvals", "list", "--as", "kc-owner-dale")
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert _error(result) == "error: database error: database is locked\n"


# A database error while the command runs, after the world opened


def _raise_locked(*_args: object, **_kwargs: object) -> None:
    raise sqlite3.OperationalError("database is locked")


def test_a_database_error_mid_command_is_one_error_line(
    tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    monkeypatch.setattr(Gate, "approve", _raise_locked)
    result = _cli(tmp, "approvals", "approve", "apr_x", "--as", "kc-mgr-marisol")
    assert _error(result) == "error: database error: database is locked\n"
    monkeypatch.setattr(Store, "list_users", _raise_locked)
    result = _cli(tmp, "users", "list", "--as", "kc-owner-dale")
    assert _error(result) == "error: database error: database is locked\n"


def test_a_database_error_while_serving_is_one_error_line(
    tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agentgate.gateway, "serve_stdio", _raise_locked)
    result = _cli(tmp, "gateway", "serve", "--as", "kc-staff-nadine")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.endswith("error: database error: database is locked\n")
    assert "Traceback" not in result.stderr


def test_a_database_error_in_agent_run_is_one_error_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentgate.cli.run_agent", _raise_locked)
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--as",
            "kc-staff-nadine",
            "--task",
            "Hi",
            "--policies",
            str(POLICIES),
            "--fixtures",
            str(FIXTURES),
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stderr.endswith("error: database error: database is locked\n")


def test_a_database_error_in_audit_verify_is_one_error_line(
    tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ok(_cli(tmp, "users", "list", "--as", "kc-owner-dale"))
    monkeypatch.setattr(PanoptesSink, "verify", _raise_locked)
    result = runner.invoke(app, ["audit", "verify", "--db", str(_audit(tmp))])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stdout == ""
    assert result.stderr == "error: database error: database is locked\n"


def test_new_command_groups_are_listed() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "users" in result.stdout and "spend" in result.stdout
    users = runner.invoke(app, ["users", "--help"])
    for name in ("list", "suspend", "reinstate"):
        assert name in users.stdout
