"""The agentgate command line: policy lint and test, audit verify, and the approval queue.

Output is plain, deterministic text (customers sorted, cases in file order, no colour), so
it reads the same in a terminal, a CI log and a diff. Problems that stop a command go to
stderr as one "error: ..." line with exit status 1.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

import agentgate
from agentgate.approvals import Approval, ApprovalError, ApprovalStatus
from agentgate.audit.panoptes import PanoptesSink
from agentgate.fixtures import FixtureError
from agentgate.models import Role, User, UserStatus
from agentgate.policy.schema import PolicyError
from agentgate.policy.testing import load_policy_bundles, run_policy_tests
from agentgate.policy.version import bundle_version, git_revision
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    World,
    open_world,
)

app = typer.Typer(
    help="Permissions, approval and audit layer between an AI agent and a company's tools.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
policy_app = typer.Typer(help="Validate and test customer policies.", no_args_is_help=True)
app.add_typer(policy_app, name="policy")
audit_app = typer.Typer(help="Inspect the audit log.", no_args_is_help=True)
app.add_typer(audit_app, name="audit")
approvals_app = typer.Typer(
    help="Review and decide queued approvals. --as names the acting approver.",
    no_args_is_help=True,
)
app.add_typer(approvals_app, name="approvals")

PoliciesOption = Annotated[
    Path, typer.Option("--policies", help="Directory holding one folder per customer policy.")
]
FixturesOption = Annotated[
    Path, typer.Option("--fixtures", help="Directory holding the fixture customers.")
]


def _show_version(value: bool) -> None:
    if value:
        typer.echo(agentgate.__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Print the version and exit."
        ),
    ] = False,
) -> None:
    """agentgate decides what each agent run can read, write, spend and send."""


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


@policy_app.command("lint")
def lint(policies: PoliciesOption = Path("policies")) -> None:
    """Load and validate every policy bundle."""
    try:
        bundles = load_policy_bundles(policies)
    except PolicyError as exc:
        _fail(str(exc))
    for customer_id in sorted(bundles):
        bundle = bundles[customer_id]
        line = (
            f"{customer_id}  {bundle_version(bundle)}"
            f"  {len(bundle.policy.rules)} rules  {len(bundle.tools)} tools"
        )
        revision = git_revision(policies / customer_id)
        if revision is not None:
            line += f"  git {revision}"
        typer.echo(line)


@policy_app.command("test")
def test(
    policies: PoliciesOption = Path("policies"),
    fixtures: FixturesOption = Path("fixtures"),
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Also list the cases that passed.")
    ] = False,
) -> None:
    """Run every customer's tests.yaml against its policy and the fixtures."""
    try:
        report = run_policy_tests(policies, fixtures)
    except (PolicyError, FixtureError) as exc:
        _fail(str(exc))
    for customer in report.customers:
        for result in customer.results:
            name = f"{customer.customer_id} :: {result.case.name}"
            if not result.passed:
                typer.echo(f"FAIL {name} :: {result.problem}")
            elif verbose:
                typer.echo(f"ok   {name}")
    for customer in report.customers:
        typer.echo(
            f"{customer.customer_id}  {customer.policy_version}"
            f"  {customer.passed}/{len(customer.results)} passed"
        )
    typer.echo(f"total  {report.passed}/{report.total} passed")
    if report.failed:
        raise typer.Exit(code=1)


@audit_app.command("verify")
def verify(
    db: Annotated[Path, typer.Option("--db", help="Panoptes audit database to check.")] = Path(
        "data/audit.db"
    ),
) -> None:
    """Check the audit log's hash chain. Exits 1 if any stored event was changed."""
    try:
        # create=False: a mistyped path must fail, not verify a new empty log.
        sink = PanoptesSink(db, create=False)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        _fail(str(exc))
    try:
        result = sink.verify()
        count = sink.count()
    finally:
        sink.close()
    if result["valid"]:
        typer.echo(f"{db}  {count} events  chain intact")
        return
    line = f"{db}  {count} events  chain BROKEN: {result['error']}"
    if result.get("broken_event_id"):
        line += f" at event {result['broken_event_id']}"
    typer.echo(line)
    raise typer.Exit(code=1)


# approvals

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
NoteOption = Annotated[str | None, typer.Option("--note", help="Note stored with the decision.")]

DEMO_IDENTITY_NOTE = "note: --as is a demo identity; agentgate does not authenticate users"


@contextmanager
def _world(db: Path, audit_db: Path, policies: Path, fixtures: Path) -> Iterator[World]:
    # Printed before anything else, so even a failing command says it once.
    typer.echo(DEMO_IDENTITY_NOTE, err=True)
    try:
        world = open_world(db_path=db, audit_path=audit_db, policies=policies, fixtures=fixtures)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        _fail(str(exc))
    try:
        yield world
    finally:
        world.close()


def _approver(world: World, user_id: str) -> User:
    """The acting user, if active and holding an approver role in their customer's policy."""
    user = world.store.get_user(user_id)
    if user is None:
        _fail(f"unknown user {user_id!r}")
    if user.status is not UserStatus.ACTIVE:
        _fail(f"user {user_id!r} is not active")
    engine = world.engines.get(user.customer_id)
    if engine is None:
        _fail(f"customer {user.customer_id!r} has no policy")
    if user.role not in engine.bundle.policy.approvals.approver_roles:
        _fail(f"role {user.role.value!r} may not review approvals")
    return user


def _time(value: datetime | None) -> str:
    return "-" if value is None else value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    """List the acting approver's customer's approvals, oldest first."""
    with _world(db, audit_db, policies, fixtures) as world:
        user = _approver(world, as_user)
        # So approvals past their expiry show as expired, not pending.
        world.gate.expire_due()
        approvals = world.gate.approvals.list(user.customer_id, status)
    for approval in approvals:
        typer.echo(
            f"{approval.id}  {approval.status.value}  {approval.tool}"
            f"  {approval.requested_by}  risk {approval.risk_score}"
            f"  created {_time(approval.created_at)}  expires {_time(approval.expires_at)}"
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
    """Show every field of one approval, including the exact args that would run."""
    with _world(db, audit_db, policies, fixtures) as world:
        user = _approver(world, as_user)
        world.gate.expire_due()
        approval = world.gate.approvals.get(approval_id)
    # Another customer's approval reads exactly like a missing one.
    if approval is None or approval.customer_id != user.customer_id:
        _fail("approval not found")
    for line in _describe(approval):
        typer.echo(line)


def _describe(approval: Approval) -> list[str]:
    fields = [
        ("id", approval.id),
        ("customer", approval.customer_id),
        ("status", approval.status.value),
        ("tool", approval.tool),
        ("action", approval.action.value),
        ("requested_by", approval.requested_by),
        ("run_id", approval.run_id),
        ("risk", str(approval.risk_score)),
        ("rule_ids", ", ".join(approval.rule_ids) or "-"),
        ("policy_version", approval.policy_version),
        ("args_hash", approval.args_hash),
        ("created_at", _time(approval.created_at)),
        ("expires_at", _time(approval.expires_at)),
        ("decided_by", approval.decided_by or "-"),
        ("decided_at", _time(approval.decided_at)),
        # Free text from an approver, so quoted and escaped like the args.
        ("note", "-" if approval.note is None else _json(approval.note)),
        ("executed_at", _time(approval.executed_at)),
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
    with _world(db, audit_db, policies, fixtures) as world:
        # The gate checks the approver itself and records the decision or the refusal.
        try:
            if approve:
                decided = world.gate.approve(approval_id, as_user, note)
            else:
                decided = world.gate.reject(approval_id, as_user, note)
        except ApprovalError as err:
            _fail(f"{err.reason.value}: {err}")
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
    with _world(db, audit_db, policies, fixtures) as world:
        user = _approver(world, as_user)
        if user.role is not Role.OWNER:
            _fail(f"role {user.role.value!r} may not expire approvals; owners only")
        expired = world.gate.expire_due()
    typer.echo(f"expired {len(expired)} approvals")


def main() -> None:
    app()
