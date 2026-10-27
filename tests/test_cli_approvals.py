"""`agentgate approvals` end to end: a world on disk queues an approval, the CLI reviews and
decides it in separate invocations, and a reopened world runs or refuses it.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from agentgate.approvals import Approval, ApprovalStatus
from agentgate.audit.panoptes import PanoptesSink
from agentgate.cli import DEMO_IDENTITY_NOTE, app
from agentgate.clock import Clock, FixedClock
from agentgate.context import RunContext
from agentgate.gate import Denied, Executed, PendingApproval
from agentgate.models import Action
from agentgate.tools.builtin import list_outbox
from conftest import FIXTURES, KC, POLICIES, open_file_world, plain_text

runner = CliRunner()

DIRECTORY_CUSTOMER = "beatrice.l@mail-sample.example"
# A non-ASCII letter, so the test also checks that show escapes what it prints.
ARGS: dict[str, Any] = {
    "to": [DIRECTORY_CUSTOMER],
    "subject": "Your visit on Monday",
    "body": "Hi Beatrice, the technician arrives at 9. Café is on us.",
}


@dataclass
class Queued:
    tmp: Path
    ctx: RunContext
    approval: Approval

    @property
    def id(self) -> str:
        return self.approval.id


def _queue(tmp: Path, requester: str = "kc-staff-nadine", clock: Clock | None = None) -> Queued:
    """A world session in which requester's run asks to email a directory customer."""
    with open_file_world(tmp, clock) as world:
        ctx = world.gate.start_run(KC, requester)
        outcome = world.gate.call_tool(ctx, "email.send", Action.SEND, ARGS)
        assert isinstance(outcome, PendingApproval), outcome
        approval = world.gate.approvals.get(outcome.approval_id)
    assert approval is not None
    return Queued(tmp, ctx, approval)


def _cli(tmp: Path, *args: str) -> Result:
    paths = [
        "--db",
        str(tmp / "data" / "agentgate.db"),
        "--audit-db",
        str(tmp / "data" / "audit.db"),
        "--policies",
        str(POLICIES),
        "--fixtures",
        str(FIXTURES),
    ]
    result = runner.invoke(app, ["approvals", *args, *paths])
    # Every invocation says once, on stderr, that --as is not a login.
    assert result.stderr.startswith(DEMO_IDENTITY_NOTE + "\n"), result.stderr
    assert result.stderr.count(DEMO_IDENTITY_NOTE) == 1
    assert DEMO_IDENTITY_NOTE not in result.stdout
    assert "\x1b" not in result.output
    assert chr(0x2014) not in result.output  # em dash
    return result


def _error(result: Result) -> str:
    """The stderr text after the demo-identity note."""
    assert result.exit_code == 1, result.output
    return result.stderr.removeprefix(DEMO_IDENTITY_NOTE + "\n")


def _time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _line(approval: Approval, status: str = "pending") -> str:
    return (
        f"{approval.id}  {status}  email.send  {approval.requested_by}"
        f"  risk {approval.risk_score}  created {_time(approval.created_at)}"
        f"  expires {_time(approval.expires_at)}"
    )


@pytest.fixture
def queued(tmp_path: Path) -> Queued:
    return _queue(tmp_path)


# list


def test_list_shows_the_pending_approval_to_a_manager(queued: Queued) -> None:
    result = _cli(queued.tmp, "list", "--as", "kc-mgr-marisol")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [_line(queued.approval), f"{KC}  1 approvals"]
    assert result.stderr == DEMO_IDENTITY_NOTE + "\n"


def test_list_filters_by_status(queued: Queued) -> None:
    result = _cli(queued.tmp, "list", "--as", "kc-owner-dale", "--status", "approved")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{KC}  0 approvals\n"
    result = _cli(queued.tmp, "list", "--as", "kc-owner-dale", "--status", "pending")
    assert result.stdout.splitlines() == [_line(queued.approval), f"{KC}  1 approvals"]


@pytest.mark.parametrize(
    ("user", "message"),
    [
        ("kc-staff-nadine", "error: role 'staff' may not review approvals\n"),
        ("kc-contractor-hollis", "error: role 'contractor' may not review approvals\n"),
        ("kc-staff-gavin", "error: user 'kc-staff-gavin' is not active\n"),
        ("nobody", "error: unknown user 'nobody'\n"),
    ],
)
def test_list_refuses_non_approvers(queued: Queued, user: str, message: str) -> None:
    result = _cli(queued.tmp, "list", "--as", user)
    assert _error(result) == message
    assert result.stdout == ""


def test_list_is_scoped_to_the_acting_users_customer(queued: Queued) -> None:
    result = _cli(queued.tmp, "list", "--as", "jr-mgr-daniel")
    assert result.exit_code == 0, result.output
    assert result.stdout == "juniper-row-apothecary  0 approvals\n"


def test_as_is_required(queued: Queued) -> None:
    result = runner.invoke(app, ["approvals", "list", "--db", str(queued.tmp / "x.db")])
    assert result.exit_code == 2
    assert "--as" in plain_text(result.stderr)


# show


def test_show_prints_every_field_and_the_exact_args(queued: Queued) -> None:
    result = _cli(queued.tmp, "show", queued.id, "--as", "kc-mgr-terrence")
    assert result.exit_code == 0, result.output
    a = queued.approval
    lines = result.stdout.splitlines()
    assert lines[:17] == [
        f"id              {a.id}",
        f"customer        {KC}",
        "status          pending",
        "tool            email.send",
        "action          send",
        "requested_by    kc-staff-nadine",
        f"run_id          {queued.ctx.run_id}",
        f"risk            {a.risk_score}",
        "rule_ids        email-customers",
        f"policy_version  {a.policy_version}",
        f"args_hash       {a.args_hash}",
        f"created_at      {_time(a.created_at)}",
        f"expires_at      {_time(a.expires_at)}",
        "decided_by      -",
        "decided_at      -",
        "note            -",
        "executed_at     -",
    ]
    assert lines[17] == "args"
    shown = "\n".join(lines[18:])
    assert shown == json.dumps(ARGS, indent=2, sort_keys=True)
    assert "Caf\\u00e9" in shown
    assert json.loads(shown) == ARGS


@pytest.mark.parametrize("approval_id", ["own", "apr_missing"])
def test_show_hides_other_customers_and_missing_ids_alike(queued: Queued, approval_id: str) -> None:
    target = queued.id if approval_id == "own" else approval_id
    result = _cli(queued.tmp, "show", target, "--as", "jr-mgr-daniel")
    assert _error(result) == "error: approval not found\n"
    assert result.stdout == ""


def test_show_refuses_staff(queued: Queued) -> None:
    result = _cli(queued.tmp, "show", queued.id, "--as", "kc-staff-colby")
    assert _error(result) == "error: role 'staff' may not review approvals\n"


# approve and reject


def test_staff_cannot_approve(queued: Queued) -> None:
    result = _cli(queued.tmp, "approve", queued.id, "--as", "kc-staff-colby")
    assert _error(result) == (
        "error: approver_not_allowed: role 'staff' may not decide approvals\n"
    )


def test_requester_cannot_approve_their_own_request(tmp_path: Path) -> None:
    mine = _queue(tmp_path, requester="kc-mgr-marisol")
    result = _cli(tmp_path, "approve", mine.id, "--as", "kc-mgr-marisol")
    assert _error(result) == (
        "error: self_approval: an approval cannot be decided by its requester\n"
    )


def test_another_customers_manager_gets_not_found(queued: Queued) -> None:
    result = _cli(queued.tmp, "approve", queued.id, "--as", "jr-owner-marisol")
    assert _error(result) == f"error: not_found: approval {queued.id!r} not found\n"


def test_approve_then_execute_in_a_reopened_world(queued: Queued) -> None:
    result = _cli(queued.tmp, "approve", queued.id, "--as", "kc-mgr-marisol", "--note", "ok")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"approved {queued.id}\n"

    shown = _cli(queued.tmp, "show", queued.id, "--as", "kc-owner-dale").stdout.splitlines()
    assert "status          approved" in shown
    assert "decided_by      kc-mgr-marisol" in shown
    assert 'note            "ok"' in shown

    again = _cli(queued.tmp, "approve", queued.id, "--as", "kc-mgr-terrence")
    assert _error(again).startswith("error: not_pending: ")

    with open_file_world(queued.tmp) as world:
        outcome = world.gate.execute_approved(
            queued.ctx, queued.id, "email.send", Action.SEND, ARGS
        )
        assert isinstance(outcome, Executed), outcome
        [sent] = list_outbox(world.store, KC)
        assert sent.recipients == (DIRECTORY_CUSTOMER,)
        assert sent.run_id == queued.ctx.run_id

    listed = _cli(queued.tmp, "list", "--as", "kc-mgr-marisol", "--status", "executed")
    assert listed.stdout.splitlines() == [_line(queued.approval, "executed"), f"{KC}  1 approvals"]

    verify = runner.invoke(app, ["audit", "verify", "--db", str(queued.tmp / "data" / "audit.db")])
    assert verify.exit_code == 0, verify.output
    assert verify.stdout.endswith("chain intact\n")


def test_reject_means_it_never_runs(queued: Queued) -> None:
    result = _cli(queued.tmp, "reject", queued.id, "--as", "kc-mgr-terrence", "--note", "no")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"rejected {queued.id}\n"

    listed = _cli(queued.tmp, "list", "--as", "kc-mgr-terrence", "--status", "rejected")
    assert listed.stdout.splitlines() == [_line(queued.approval, "rejected"), f"{KC}  1 approvals"]
    later = _cli(queued.tmp, "approve", queued.id, "--as", "kc-mgr-marisol")
    assert _error(later) == f"error: not_pending: approval {queued.id!r} is already rejected\n"

    with open_file_world(queued.tmp) as world:
        outcome = world.gate.execute_approved(
            queued.ctx, queued.id, "email.send", Action.SEND, ARGS
        )
        assert outcome == Denied(reason="not_approved")
        assert list_outbox(world.store, KC) == []

    verify = runner.invoke(app, ["audit", "verify", "--db", str(queued.tmp / "data" / "audit.db")])
    assert verify.exit_code == 0, verify.output


# expire


def test_expire_marks_stale_approvals(tmp_path: Path) -> None:
    # Created two hours ago with a 60 minute ttl, so already past expiry by the CLI's clock.
    clock = FixedClock(datetime.now(UTC) - timedelta(hours=2))
    stale = _queue(tmp_path, clock=clock)

    refused = _cli(tmp_path, "expire", "--as", "kc-mgr-marisol")
    assert _error(refused) == "error: role 'manager' may not expire approvals; owners only\n"

    result = _cli(tmp_path, "expire", "--as", "kc-owner-dale")
    assert result.exit_code == 0, result.output
    assert result.stdout == "expired 1 approvals\n"
    assert _cli(tmp_path, "expire", "--as", "kc-owner-dale").stdout == "expired 0 approvals\n"

    listed = _cli(tmp_path, "list", "--as", "kc-mgr-marisol")
    assert listed.stdout.splitlines() == [_line(stale.approval, "expired"), f"{KC}  1 approvals"]
    late = _cli(tmp_path, "approve", stale.id, "--as", "kc-mgr-marisol")
    assert _error(late) == f"error: expired: approval {stale.id!r} has expired\n"

    with open_file_world(tmp_path) as world:
        approval = world.gate.approvals.get(stale.id)
        assert approval is not None and approval.status is ApprovalStatus.EXPIRED

    verify = runner.invoke(app, ["audit", "verify", "--db", str(tmp_path / "data" / "audit.db")])
    assert verify.exit_code == 0, verify.output


def _audit_count(tmp: Path) -> int:
    sink = PanoptesSink(tmp / "data" / "audit.db", create=False)
    try:
        return sink.count()
    finally:
        sink.close()


def test_list_and_show_compute_expiry_without_writing(tmp_path: Path) -> None:
    stale = _queue(tmp_path, clock=FixedClock(datetime.now(UTC) - timedelta(hours=2)))
    events = _audit_count(tmp_path)

    result = _cli(tmp_path, "list", "--as", "kc-owner-dale", "--status", "pending")
    assert result.stdout == f"{KC}  0 approvals\n"
    result = _cli(tmp_path, "list", "--as", "kc-owner-dale", "--status", "expired")
    assert result.stdout.splitlines() == [
        _line(stale.approval, "expired (pending sweep)"),
        f"{KC}  1 approvals",
    ]
    result = _cli(tmp_path, "list", "--as", "kc-mgr-marisol")
    assert result.stdout.splitlines()[0] == _line(stale.approval, "expired (pending sweep)")
    shown = _cli(tmp_path, "show", stale.id, "--as", "kc-owner-dale").stdout.splitlines()
    assert "status          expired (pending sweep)" in shown

    # Nothing was marked or recorded: the sweep is still the owner's expire command.
    assert _audit_count(tmp_path) == events
    with open_file_world(tmp_path) as world:
        approval = world.gate.approvals.get(stale.id)
        assert approval is not None and approval.status is ApprovalStatus.PENDING
    assert _cli(tmp_path, "expire", "--as", "kc-owner-dale").stdout == "expired 1 approvals\n"
    # The expiry, and the open of a command that changes state: one policy_loaded per customer.
    assert _audit_count(tmp_path) == events + 1 + 2
    result = _cli(tmp_path, "list", "--as", "kc-owner-dale", "--status", "expired")
    assert result.stdout.splitlines()[0] == _line(stale.approval, "expired")


# world errors


def test_policy_error_is_reported(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "approvals",
            "list",
            "--as",
            "kc-owner-dale",
            "--db",
            str(tmp_path / "agentgate.db"),
            "--audit-db",
            str(tmp_path / "audit.db"),
            "--policies",
            str(tmp_path / "nope"),
            "--fixtures",
            str(FIXTURES),
        ],
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        f"{DEMO_IDENTITY_NOTE}\nerror: {tmp_path / 'nope'}: policies directory not found\n"
    )


def test_defaults_use_the_data_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "policies").symlink_to(POLICIES)
    (tmp_path / "fixtures").symlink_to(FIXTURES)
    result = runner.invoke(app, ["approvals", "list", "--as", "kc-owner-dale"])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{KC}  0 approvals\n"
    assert (tmp_path / "data" / "agentgate.db").is_file()
    assert (tmp_path / "data" / "audit.db").is_file()
