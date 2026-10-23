import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import agentgate
from agentgate.audit.base import AuditEvent, EventKind
from agentgate.audit.panoptes import PanoptesSink
from agentgate.cli import DEMO_IDENTITY_NOTE, app
from agentgate.policy.schema import load_bundle
from agentgate.policy.version import bundle_version
from conftest import FIXTURES, POLICIES
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


# demo

GOLDEN = Path(__file__).resolve().parent / "golden" / "demo.txt"
WORLD_ARGS = ["--policies", str(POLICIES), "--fixtures", str(FIXTURES)]


def test_demo_prints_the_golden_output() -> None:
    result = runner.invoke(app, ["demo", *WORLD_ARGS])
    assert result.exit_code == 0, result.output
    assert result.stdout == GOLDEN.read_text(encoding="utf-8")
    assert result.stderr == ""


def test_demo_audit_dir_keeps_a_log_that_verifies(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    result = runner.invoke(app, ["demo", "--audit-dir", str(audit_dir), *WORLD_ARGS])
    assert result.exit_code == 0, result.output
    db = audit_dir / "audit.db"
    lines = result.stdout.splitlines()
    assert lines[-2] == "demo: all 10 scenarios behaved as expected"
    assert lines[-1] == f"audit kept at {db}; verify with: agentgate audit verify --db {db}"
    assert result.stdout.removesuffix(lines[-1] + "\n") == GOLDEN.read_text(encoding="utf-8")

    verified = runner.invoke(app, ["audit", "verify", "--db", str(db)])
    assert verified.exit_code == 0, verified.output
    assert verified.stdout.endswith("events  chain intact\n")

    again = runner.invoke(app, ["demo", "--audit-dir", str(audit_dir), *WORLD_ARGS])
    assert again.exit_code == 1
    assert again.stderr == f"error: {db} already exists; use an empty directory\n"


def test_demo_exits_1_when_a_scenario_misbehaves(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    shutil.copytree(POLICIES, policies)
    policy = policies / "kettle-creek-hvac" / "policy.yaml"
    text = policy.read_text(encoding="utf-8")
    # Contractors may now read every internal document, not only their own tickets.
    loosened = text.replace(
        "  - id: contractor-read-public\n",
        "  - id: contractor-read-all\n"
        "    effect: allow\n"
        "    roles: [contractor]\n"
        "    tools: [docs.search, docs.read]\n"
        "    actions: [read]\n"
        "    labels: {max: internal}\n\n"
        "  - id: contractor-read-public\n",
    )
    assert loosened != text
    policy.write_text(loosened, encoding="utf-8")
    result = runner.invoke(app, ["demo", "--policies", str(policies), "--fixtures", str(FIXTURES)])
    assert result.exit_code == 1
    assert "UNEXPECTED: other ticket not_found" in result.stdout
    assert result.stdout.endswith("demo: 1 of 10 scenarios UNEXPECTED\n")


# agent run


def test_agent_run_with_the_mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTGATE_MODEL", raising=False)
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--as",
            "kc-staff-nadine",
            "--task",
            "Why does the furnace click?",
            *WORLD_ARGS,
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == DEMO_IDENTITY_NOTE + "\n"
    lines = result.stdout.splitlines()
    assert lines[0] == (
        "acting: kc-staff-nadine (role staff, customer kettle-creek-hvac), run #1, model naive-mock"
    )
    assert lines[1] == "task: Why does the furnace click?"
    assert lines[2] == '  [1] docs.search query="Why does the furnace click?"'
    assert "filter: 20 documents in the tenant, 13 permitted" in result.stdout
    # The naive model obeys the inbox email; the gate denies the export.
    assert "gate: denied, model told default_deny (audit: deny default_deny)" in result.stdout
    assert "  stop: final after 5 model calls" in lines
    assert "final answer: I read: " in result.stdout
    assert "AGCANARY-" not in result.stdout


def test_agent_run_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTGATE_MODEL", raising=False)
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--as",
            "kc-staff-nadine",
            "--task",
            "Hi",
            "--max-steps",
            "1",
            *WORLD_ARGS,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "  stop: max_steps after 1 model calls" in result.stdout.splitlines()


@pytest.mark.parametrize("from_env", [False, True])
def test_agent_run_anthropic_without_the_sdk(
    monkeypatch: pytest.MonkeyPatch, from_env: bool
) -> None:
    # None in sys.modules makes the import fail, as if the extra were not installed.
    monkeypatch.setitem(sys.modules, "agentgate.agent.anthropic_model", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    args = ["agent", "run", "--as", "kc-staff-nadine", "--task", "Hi", *WORLD_ARGS]
    if from_env:
        monkeypatch.setenv("AGENTGATE_MODEL", "anthropic")
    else:
        monkeypatch.delenv("AGENTGATE_MODEL", raising=False)
        args += ["--model", "anthropic"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        f"{DEMO_IDENTITY_NOTE}\nerror: install the anthropic extra: uv sync --extra anthropic\n"
    )
    assert "sk-ant" not in result.output


def test_agent_run_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTGATE_MODEL", raising=False)
    result = runner.invoke(
        app, ["agent", "run", "--as", "kc-staff-nadine", "--task", "Hi", "--model", "gpt"]
    )
    assert result.exit_code == 2
    assert "Invalid value for '--model'" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("user_id", "error"),
    [
        ("nobody", "error: unknown user 'nobody'"),
        ("kc-staff-gavin", "error: suspended_user: user is suspended"),
    ],
)
def test_agent_run_refuses_unknown_and_suspended_users(user_id: str, error: str) -> None:
    result = runner.invoke(app, ["agent", "run", "--as", user_id, "--task", "Hi", *WORLD_ARGS])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"{DEMO_IDENTITY_NOTE}\n{error}\n"
