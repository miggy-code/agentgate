"""`agentgate token issue`, `list` and `revoke`: who may manage whose tokens, that the token
is shown once and never stored or listed, and that every issue and revocation is recorded."""

import stat
from pathlib import Path

import pytest

from agentgate.audit.base import EventKind
from agentgate.audit.panoptes import read_events
from agentgate.cli import app
from agentgate.model.tokens import TokenReason, TokenRejected, TokenStore
from test_cli_ops import _audit, _cli, _error, _ok, _paths, _world, runner

STAFF = "kc-staff-nadine"
OWNER = "kc-owner-dale"
JR_OWNER = "jr-owner-marisol"


def _run(tmp_path: Path, user: str) -> str:
    with _world(tmp_path) as world:
        found = world.store.get_user(user)
        assert found is not None
        return world.gate.start_run(found.customer_id, user).run_id


def _issue(tmp_path: Path, user: str, run: str, *extra: str) -> str:
    return _ok(_cli(tmp_path, "token", "issue", "--as", user, "--run", run, *extra)).stdout.strip()


def _token_events(tmp_path: Path) -> list[tuple[str, str | None, str]]:
    return [
        (e.reason, e.user_id, e.subject or "")
        for e in read_events(_audit(tmp_path))
        if e.kind is EventKind.OPERATOR_ACTION and e.reason.startswith("token_")
    ]


def _resolve(tmp_path: Path, token: str) -> TokenRejected | None:
    with _world(tmp_path) as world:
        try:
            TokenStore(world.store, world.clock).resolve(token)
        except TokenRejected as rejected:
            return rejected
    return None


def test_issue_prints_the_token_once_and_stores_only_its_hash(tmp_path: Path) -> None:
    run = _run(tmp_path, STAFF)
    result = _ok(_cli(tmp_path, "token", "issue", "--as", STAFF, "--run", run))
    token = result.stdout.strip()
    assert token.startswith("agt_") and "\n" not in token
    assert f"for run {run}" in result.stderr and token not in result.stderr
    assert _resolve(tmp_path, token) is None
    data = b"".join(p.read_bytes() for p in (tmp_path / "data").glob("*"))
    assert token.encode() not in data


def test_issue_can_write_the_token_to_a_private_file(tmp_path: Path) -> None:
    run = _run(tmp_path, STAFF)
    path = tmp_path / "run.token"
    result = _ok(
        _cli(tmp_path, "token", "issue", "--as", STAFF, "--run", run, "--token-file", str(path))
    )
    token = path.read_text().strip()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert token not in result.stdout + result.stderr
    assert f"token written to {path}" in result.stdout
    assert _resolve(tmp_path, token) is None


def test_issue_will_not_overwrite_a_file_and_leaves_no_live_token(tmp_path: Path) -> None:
    run = _run(tmp_path, STAFF)
    path = tmp_path / "run.token"
    path.write_text("precious")
    message = _error(
        _cli(tmp_path, "token", "issue", "--as", STAFF, "--run", run, "--token-file", str(path))
    )
    assert message.startswith("error: cannot write")
    assert path.read_text() == "precious"
    listing = _ok(_cli(tmp_path, "token", "list", "--as", STAFF)).stdout
    assert "revoked (operator)" in listing and "live" not in listing
    assert [reason for reason, _, _ in _token_events(tmp_path)] == ["token_issued", "token_revoked"]


def test_issue_and_revoke_are_recorded_with_the_actor(tmp_path: Path) -> None:
    run = _run(tmp_path, STAFF)
    _issue(tmp_path, STAFF, run)
    token_id = _ok(_cli(tmp_path, "token", "list", "--as", STAFF)).stdout.split()[0]
    _ok(_cli(tmp_path, "token", "revoke", token_id, "--as", OWNER))
    assert _token_events(tmp_path) == [
        ("token_issued", STAFF, token_id),
        ("token_revoked", OWNER, token_id),
    ]


def test_a_revoked_token_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path, STAFF)
    token = _issue(tmp_path, STAFF, run)
    token_id = _ok(_cli(tmp_path, "token", "list", "--as", STAFF)).stdout.split()[0]
    assert _ok(_cli(tmp_path, "token", "revoke", token_id, "--as", STAFF)).stdout == (
        f"revoked {token_id}\n"
    )
    rejected = _resolve(tmp_path, token)
    assert rejected is not None and rejected.reason is TokenReason.REVOKED
    assert _error(_cli(tmp_path, "token", "revoke", token_id, "--as", STAFF)) == (
        f"error: token {token_id!r} is already revoked\n"
    )


def test_list_shows_status_and_never_the_token(tmp_path: Path) -> None:
    run = _run(tmp_path, STAFF)
    token = _issue(tmp_path, STAFF, run, "--ttl-minutes", "30")
    result = _ok(_cli(tmp_path, "token", "list", "--as", STAFF))
    first, count = result.stdout.splitlines()
    assert f"run {run}  live  issued" in first and "expires" in first
    assert count == "1 tokens"
    assert token not in result.stdout


def test_an_owner_manages_a_staff_runs_tokens_but_staff_do_not_manage_the_owners(
    tmp_path: Path,
) -> None:
    staff_run, owner_run = _run(tmp_path, STAFF), _run(tmp_path, OWNER)
    _issue(tmp_path, OWNER, staff_run)
    assert _error(_cli(tmp_path, "token", "issue", "--as", STAFF, "--run", owner_run)) == (
        f"error: run {owner_run!r} not found\n"
    )
    owner_token = _issue(tmp_path, OWNER, owner_run)
    assert owner_token.startswith("agt_")
    # Staff list only their own runs' tokens; the owner lists both.
    staff = _ok(_cli(tmp_path, "token", "list", "--as", STAFF)).stdout
    assert owner_run not in staff and staff.endswith("1 tokens\n")
    assert _ok(_cli(tmp_path, "token", "list", "--as", OWNER)).stdout.endswith("2 tokens\n")


def test_another_customers_run_reads_like_a_missing_one(tmp_path: Path) -> None:
    kc_run, jr_run = _run(tmp_path, STAFF), _run(tmp_path, JR_OWNER)
    _issue(tmp_path, STAFF, kc_run)
    token_id = _ok(_cli(tmp_path, "token", "list", "--as", STAFF)).stdout.split()[0]

    for run in (kc_run, "no-such-run"):
        assert _error(_cli(tmp_path, "token", "issue", "--as", JR_OWNER, "--run", run)) == (
            f"error: run {run!r} not found\n"
        )
    assert _error(_cli(tmp_path, "token", "revoke", token_id, "--as", JR_OWNER)) == (
        f"error: token {token_id!r} not found\n"
    )
    assert _ok(_cli(tmp_path, "token", "list", "--as", JR_OWNER)).stdout == "0 tokens\n"
    # Their own run is unaffected, and still invisible from the other side.
    _issue(tmp_path, JR_OWNER, jr_run)
    assert jr_run not in _ok(_cli(tmp_path, "token", "list", "--as", OWNER)).stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--as", "nobody"], "error: unknown user 'nobody'\n"),
        (["--as", "kc-staff-gavin"], "error: user 'kc-staff-gavin' is not active\n"),
    ],
)
def test_an_unknown_or_suspended_actor_is_refused(
    tmp_path: Path, args: list[str], message: str
) -> None:
    run = _run(tmp_path, STAFF)
    assert _error(_cli(tmp_path, "token", "issue", *args, "--run", run)) == message
    assert _error(_cli(tmp_path, "token", "list", *args)) == message


def test_a_token_is_refused_once_its_user_is_suspended(tmp_path: Path) -> None:
    run = _run(tmp_path, STAFF)
    token = _issue(tmp_path, STAFF, run)
    _ok(_cli(tmp_path, "users", "suspend", STAFF, "--as", OWNER))
    rejected = _resolve(tmp_path, token)
    assert rejected is not None and rejected.reason is TokenReason.USER_SUSPENDED
    # No new token for a suspended user's run, either.
    assert _error(_cli(tmp_path, "token", "issue", "--as", OWNER, "--run", run)).startswith(
        "error: the run's user is suspended"
    )


@pytest.mark.parametrize("ttl", ["0", "-5", str(31 * 24 * 60), "x"])
def test_a_bad_lifetime_is_refused(tmp_path: Path, ttl: str) -> None:
    run = _run(tmp_path, STAFF)
    result = runner.invoke(
        app,
        ["token", "issue", "--as", STAFF, "--run", run, "--ttl-minutes", ttl, *_paths(tmp_path)],
    )
    assert result.exit_code == 2
    assert _ok(_cli(tmp_path, "token", "list", "--as", STAFF)).stdout == "0 tokens\n"
