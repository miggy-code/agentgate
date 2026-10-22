import os
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import agentgate
from agentgate.audit.base import AuditEvent, EventKind
from agentgate.audit.panoptes import PanoptesSink
from agentgate.cli import app
from agentgate.policy.schema import load_bundle
from agentgate.policy.version import bundle_version
from test_policy_testing import CASES, case, write_fixtures, write_policies, write_tree

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep git_revision from finding a repository above tmp_path or the developer's config.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))


def _version(policies: Path, customer_id: str = "acme") -> str:
    return bundle_version(load_bundle(policies / customer_id))


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"{agentgate.__version__}\n"


# policy lint


def test_lint_lists_each_bundle(tmp_path: Path) -> None:
    policies = write_policies(tmp_path / "policies", CASES)
    write_policies(policies, CASES, "beta")
    result = runner.invoke(app, ["policy", "lint", "--policies", str(policies)])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"acme  {_version(policies)}  3 rules  3 tools",
        f"beta  {_version(policies, 'beta')}  3 rules  3 tools",
    ]
    assert result.stderr == ""


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_lint_shows_the_git_revision_of_committed_policies(tmp_path: Path) -> None:
    policies = write_policies(tmp_path / "policies", CASES)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t.example", *args],
            cwd=policies,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()

    git("init", "-q")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "policies")
    revision = git("log", "-1", "--format=%h")
    result = runner.invoke(app, ["policy", "lint", "--policies", str(policies)])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"acme  {_version(policies)}  3 rules  3 tools  git {revision}\n"


def test_lint_error_goes_to_stderr(tmp_path: Path) -> None:
    policies = write_policies(tmp_path / "policies", CASES)
    (policies / "acme" / "tools.yaml").unlink()
    result = runner.invoke(app, ["policy", "lint", "--policies", str(policies)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert "missing tools.yaml" in result.stderr


def test_lint_missing_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["policy", "lint", "--policies", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "policies directory not found" in result.stderr


# policy test


def _test(policies: Path, fixtures: Path, *extra: str) -> tuple[int, str, str]:
    args = ["policy", "test", "--policies", str(policies), "--fixtures", str(fixtures), *extra]
    result = runner.invoke(app, args)
    return result.exit_code, result.stdout, result.stderr


def test_test_passes(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path)
    code, out, err = _test(policies, fixtures)
    assert code == 0, out + err
    assert out.splitlines() == [f"acme  {_version(policies)}  4/4 passed", "total  4/4 passed"]
    assert err == ""


def test_test_verbose_lists_passing_cases(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path)
    code, out, _ = _test(policies, fixtures, "--verbose")
    assert code == 0
    assert out.splitlines()[:4] == [f"ok   acme :: {entry['name']}" for entry in CASES]


def test_test_failures_exit_1(tmp_path: Path) -> None:
    policies, fixtures = write_tree(
        tmp_path, [CASES[0], case(name="wrong effect", expect="deny"), CASES[1]]
    )
    write_policies(policies, [case(name="ghost", user="nobody")], "beta")
    code, out, err = _test(policies, fixtures, "--verbose")
    assert code == 1
    assert out.splitlines() == [
        f"ok   acme :: {CASES[0]['name']}",
        "FAIL acme :: wrong effect :: expected deny, got allow (matched_allow [read-internal])",
        f"ok   acme :: {CASES[1]['name']}",
        "FAIL beta :: ghost :: unknown user 'nobody'",
        f"acme  {_version(policies)}  2/3 passed",
        f"beta  {_version(policies, 'beta')}  0/1 passed",
        "total  2/4 passed",
    ]
    assert err == ""


def test_test_policy_error_exits_1(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path)
    (policies / "acme" / "tests.yaml").unlink()
    code, out, err = _test(policies, fixtures)
    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "missing tests.yaml" in err


def test_test_fixture_error_exits_1(tmp_path: Path) -> None:
    policies, _ = write_tree(tmp_path)
    fixtures = write_fixtures(tmp_path / "bad-fixtures")
    (fixtures / "acme" / "users.yaml").unlink()
    code, out, err = _test(policies, fixtures)
    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert "missing users.yaml" in err


def test_output_has_no_colour_or_em_dashes(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path, [case(name="wrong", expect="deny")])
    _, out, _ = _test(policies, fixtures, "--verbose")
    assert "\x1b" not in out
    assert chr(0x2014) not in out  # em dash


# audit verify


def _audit_db(path: Path, events: int = 3) -> list[str]:
    sink = PanoptesSink(path)
    try:
        recorded = [
            AuditEvent(
                timestamp=datetime(2026, 10, 22, 12, 0, index, tzinfo=UTC),
                kind=EventKind.ALLOW,
                customer_id="acme",
                reason="matched_allow",
            )
            for index in range(events)
        ]
        for event in recorded:
            sink.record(event)
    finally:
        sink.close()
    return [event.event_id for event in recorded]


def test_audit_verify_intact_chain(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    _audit_db(db)
    result = runner.invoke(app, ["audit", "verify", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{db}  3 events  chain intact\n"
    assert result.stderr == ""


def test_audit_verify_tampered_chain(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    event_ids = _audit_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE events SET context = json_set(context, '$.agentgate.reason', 'forged')"
        " WHERE event_id = ?",
        (event_ids[1],),
    )
    conn.commit()
    conn.close()
    result = runner.invoke(app, ["audit", "verify", "--db", str(db)])
    assert result.exit_code == 1
    assert result.stdout == (
        f"{db}  3 events  chain BROKEN: Canonical event hash mismatch at event {event_ids[1]}\n"
    )
    assert "\x1b" not in result.stdout
    assert chr(0x2014) not in result.stdout  # em dash


def test_audit_verify_missing_file(tmp_path: Path) -> None:
    db = tmp_path / "data" / "audit.db"
    result = runner.invoke(app, ["audit", "verify", "--db", str(db)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"error: audit database not found: {db}\n"
    assert not db.exists()
    assert not db.parent.exists()


def test_audit_verify_defaults_to_data_audit_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _audit_db(tmp_path / "data" / "audit.db", events=1)
    result = runner.invoke(app, ["audit", "verify"])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{Path('data/audit.db')}  1 events  chain intact\n"
