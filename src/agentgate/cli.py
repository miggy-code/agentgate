"""The agentgate command line: policy lint and test, audit verify, the approval queue, users,
the spend ledger, the demo, one-off agent runs, the MCP gateway, the red-team suite and the
latency bench.

Output is plain, deterministic text (customers sorted, cases in file order, no colour), so
it reads the same in a terminal, a CI log and a diff. Problems that stop a command go to
stderr as one "error: ..." line with exit status 1. That includes database problems while a
command runs (a read-only, locked or corrupt file): never a traceback.

Commands that open a world on disk refuse a --db and --audit-db that are the same file, a
--db that is a Panoptes audit log and an --audit-db that is an agentgate store, before
anything is written. Opening a world also syncs the fixtures, reclaims stale spend
reservations and tightens a store file other users could read; when any of that changed
something it says so on stderr, after the demo-identity note, and never on stdout.
"""

import json
import shlex
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import JsonValue

import agentgate
from agentgate.agent.loop import run_agent, tool_specs
from agentgate.agent.mock import NaiveModel
from agentgate.agent.model import Model
from agentgate.approvals import Approval, ApprovalError, ApprovalStatus
from agentgate.audit.base import AuditEvent, EventKind, MemorySink, ReferencedField
from agentgate.audit.panoptes import read_events, verify_log
from agentgate.demo import AUDIT_FILE, Names, format_transcript, run_demo
from agentgate.fixtures import FixtureError
from agentgate.gate import IdentityError
from agentgate.models import Role, User, UserStatus
from agentgate.policy.schema import PolicyError
from agentgate.policy.testing import load_policy_bundles, run_policy_tests
from agentgate.policy.version import bundle_version, git_revision
from agentgate.redteam import bench as redteam_bench
from agentgate.redteam import report as redteam_report
from agentgate.redteam.baseline import (
    DEFAULT_BASELINE,
    BaselineError,
    baseline_from,
    check_against_baseline,
    load_baseline,
    write_baseline,
)
from agentgate.redteam.core import Category, SuiteResult, run_suite
from agentgate.redteam.transport import Transport
from agentgate.store import Store
from agentgate.tools.builtin import default_tools
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    DEFAULT_RESERVATION_LEASE,
    OpenRecord,
    OperatorReason,
    StoreCheck,
    UserAdminError,
    World,
    bundle_sha256,
    check_store,
    open_memory_world,
    open_world,
    reinstate_user,
    suspend_user,
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
agent_app = typer.Typer(help="Run the reference agent through the gate.", no_args_is_help=True)
app.add_typer(agent_app, name="agent")
gateway_app = typer.Typer(
    help="Serve a user's tools to an MCP client, every call through the gate.",
    no_args_is_help=True,
)
app.add_typer(gateway_app, name="gateway")
users_app = typer.Typer(
    help="List, suspend and reinstate users. --as names the acting user.", no_args_is_help=True
)
app.add_typer(users_app, name="users")
spend_app = typer.Typer(help="Operate the spend ledger.", no_args_is_help=True)
app.add_typer(spend_app, name="spend")

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
    """Check the audit log's hash chain. Exits 1 if any stored event was changed, or any
    event is outside the chain. Read-only: a snapshot of the file is verified."""
    try:
        # A mistyped path fails here; nothing is created or changed.
        result, count = verify_log(db)
    except sqlite3.Error as exc:
        _fail(_database_error(exc))
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
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


def _database_error(exc: sqlite3.Error) -> str:
    # sqlite3 messages name the problem ("database is locked", "attempt to write a readonly
    # database") but not the data, so they are safe to print.
    return f"database error: {exc}"


# audit show, policy and check-store


def _reading_user(store: Store, user_id: str, *, owners_only: bool) -> User:
    """The acting user of a read-only audit command: active, an owner (or a manager when
    owners_only is false)."""
    user = store.get_user(user_id)
    if user is None:
        _fail(f"unknown user {user_id!r}")
    if user.status is not UserStatus.ACTIVE:
        _fail(f"user {user_id!r} is not active")
    allowed = {Role.OWNER} if owners_only else {Role.OWNER, Role.MANAGER}
    if user.role not in allowed:
        who = "owners only" if owners_only else "owners and managers only"
        _fail(f"role {user.role.value!r} may not read the audit log; {who}")
    return user


@contextmanager
def _audit_reader(
    db: Path, audit_db: Path, as_user: str, *, owners_only: bool
) -> Iterator[tuple[Store, User, list[AuditEvent]]]:
    """The store opened read-only, the acting user and every event in the log."""
    typer.echo(DEMO_IDENTITY_NOTE, err=True)
    try:
        store = Store(db, readonly=True)
    except sqlite3.Error as exc:
        _fail(_database_error(exc))
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    try:
        user = _reading_user(store, as_user, owners_only=owners_only)
        try:
            events = read_events(audit_db)
        except (OSError, ValueError) as exc:
            _fail(str(exc))
        yield store, user, events
    except sqlite3.Error as exc:
        _fail(_database_error(exc))
    finally:
        store.close()


def _shown(value: str) -> str:
    """A stored value as one line of plain ASCII: anything else is shown escaped."""
    if value.isascii() and value.isprintable() and " " not in value:
        return value
    return json.dumps(value, ensure_ascii=True)


def _shown_detail(value: JsonValue) -> str:
    if isinstance(value, str):
        return _shown(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ",".join(_shown(str(item)) for item in value) or "[]"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _field(event: AuditEvent, name: ReferencedField) -> str:
    value = getattr(event, name)
    if value is None:
        return "-"
    reference = event.references.get(name)
    if reference is not None:
        return f"{reference.ref} ({reference.why} value)"
    return _shown(str(value))


def _event_line(event: AuditEvent) -> str:
    stamp = event.timestamp.astimezone(UTC).isoformat(timespec="milliseconds")
    parts = [
        stamp.replace("+00:00", "Z"),
        event.kind.value,
        _shown(event.reason),
        f"user={_field(event, 'user_id')}",
        f"role={'-' if event.role is None else event.role.value}",
        f"tool={_field(event, 'tool')}",
        f"subject={_field(event, 'subject')}",
        f"rules={','.join(_shown(r) for r in event.rule_ids) or '-'}",
        f"policy={_shown(event.policy_version) if event.policy_version else '-'}",
    ]
    if event.run_id is not None:
        parts.append(f"run={_field(event, 'run_id')}")
    parts += [f"{key}={_shown_detail(value)}" for key, value in sorted(event.details.items())]
    return "  ".join(parts)


@audit_app.command("show")
def audit_show(
    as_user: AsOption,
    db: Annotated[Path, typer.Option("--db", help="agentgate database, read-only.")] = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    run: Annotated[
        str | None, typer.Option("--run", metavar="RUN_ID", help="Only this run's events.")
    ] = None,
    customer: Annotated[
        str | None,
        typer.Option(
            "--customer", metavar="ID", help="Must be the acting user's own customer, if given."
        ),
    ] = None,
    kind: Annotated[
        EventKind | None, typer.Option("--kind", help="Only events of this kind.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Only the last N matching events.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the stored events as JSON instead.")
    ] = False,
) -> None:
    """Print the acting user's customer's audit events, oldest first, one per line.

    Owners and managers only. Read-only: neither file is changed. A value the log holds as a
    reference prints as its ref: and why ("unknown value": it named nothing the store knew).
    Run `agentgate audit verify` to check the chain itself."""
    with _audit_reader(db, audit_db, as_user, owners_only=False) as (_, user, events):
        # Another customer reads exactly like a missing one.
        if customer is not None and customer != user.customer_id:
            _fail(f"customer {customer!r} not found")
        shown = [
            event
            for event in events
            if event.customer_id == user.customer_id
            and (run is None or event.run_id == run)
            and (kind is None or event.kind is kind)
        ]
    if limit is not None:
        shown = shown[-limit:]
    if as_json:
        typer.echo(
            json.dumps(
                [event.model_dump(mode="json") for event in shown],
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
        )
        return
    for event in shown:
        typer.echo(_event_line(event))
    typer.echo(f"{user.customer_id}  {len(shown)} events")


@audit_app.command("policy")
def audit_policy(
    version: Annotated[str, typer.Argument(metavar="VERSION", help="A policy version hash.")],
    as_user: AsOption,
    db: Annotated[Path, typer.Option("--db", help="agentgate database, read-only.")] = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
) -> None:
    """Print the policy bundle stored for a version, and check it against the chain.

    The bundle is the canonical JSON the version is computed over. Its sha256 must match
    the snapshot row, the version, and every policy_loaded event for it on the chain.
    Exits 1 on any mismatch. Owners and managers only; read-only."""
    with _audit_reader(db, audit_db, as_user, owners_only=False) as (store, user, events):
        snapshot = store.get_policy_snapshot(version) if version.isprintable() else None
        loaded = [
            event
            for event in events
            if event.kind is EventKind.POLICY_LOADED
            and event.customer_id == user.customer_id
            and event.policy_version == version
        ]
        # A version is the hash of the bundle alone, so two customers with the same rules
        # share one snapshot; the customer's own chain shows it loaded that version. Any
        # other customer's version reads exactly like a missing one.
        if snapshot is None or (snapshot.customer_id != user.customer_id and not loaded):
            _fail(f"policy version {version!r} not found")
    digest = bundle_sha256(snapshot.bundle)
    try:
        bundle = json.loads(snapshot.bundle)
    except ValueError:
        bundle = None
    problems: list[str] = []
    if bundle is None:
        problems.append("the stored bundle is not JSON")
    else:
        typer.echo(json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=True))
    if digest != snapshot.sha256:
        problems.append("the stored bundle does not match its stored sha256")
    if "sha256:" + digest[:16] != version:
        problems.append("the stored bundle does not hash to its version")
    recorded = [event.details.get("bundle_sha256") for event in loaded]
    if any(value != digest for value in recorded):
        problems.append("the stored bundle does not match a policy_loaded event")
    for problem in problems:
        typer.echo(f"MISMATCH {problem}")
    on_chain = f"{len(loaded)} policy_loaded events"
    if not loaded:
        on_chain += " (checked against the version only)"
    status = "ok" if not problems else "MISMATCH"
    typer.echo(f"policy {version}  {user.customer_id}  sha256 {digest}  {on_chain}  {status}")
    if problems:
        raise typer.Exit(code=1)


_CHECKED_KINDS = (
    ("approval", "approvals"),
    ("email.send", "outbox emails"),
    ("parts.order", "parts orders"),
    ("refund.issue", "refunds"),
    ("notes.write", "notes"),
)


@audit_app.command("check-store")
def audit_check_store(
    as_user: AsOption,
    db: Annotated[Path, typer.Option("--db", help="agentgate database, read-only.")] = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
) -> None:
    """Check the acting owner's customer's approval, outbox, parts order, refund and note
    rows against the audit chain. Exits 1 if any row does not match, or the chain is broken.

    The chain is verified first. Each approval row's args and args_hash must match the
    approval_args_hash its events carry, and each side-effect row must hash to the args_hash
    of the executed event that names it. Owners only; read-only."""
    with _audit_reader(db, audit_db, as_user, owners_only=True) as (store, user, events):
        try:
            verified, _ = verify_log(audit_db)
        except (OSError, ValueError, RuntimeError) as exc:
            _fail(str(exc))
        if not verified["valid"]:
            typer.echo(f"{audit_db}  chain BROKEN: {verified['error']}")
            raise typer.Exit(code=1)
        result = check_store(store, events, user.customer_id)
    for line in _check_lines(result):
        typer.echo(line)
    if not result.ok:
        raise typer.Exit(code=1)


def _check_lines(result: StoreCheck) -> list[str]:
    lines = [f"MISMATCH {m.kind} {m.id}: {m.problem}" for m in result.mismatches]
    if result.unchecked:
        lines.append(
            f"note: {len(result.unchecked)} rows were recorded before the chain carried"
            " hashes; only their ids were checked"
        )
    counts = ", ".join(f"{result.checked.get(kind, 0)} {label}" for kind, label in _CHECKED_KINDS)
    lines.append(f"{result.customer_id}  {counts} checked  {len(result.mismatches)} mismatches")
    return lines


@contextmanager
def _world(
    db: Path,
    audit_db: Path,
    policies: Path,
    fixtures: Path,
    *,
    reservation_lease: timedelta | None = DEFAULT_RESERVATION_LEASE,
    record_open: OpenRecord = OpenRecord.ALL,
) -> Iterator[World]:
    # Printed before anything else, so even a failing command says it once.
    typer.echo(DEMO_IDENTITY_NOTE, err=True)
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
        _fail(_database_error(exc))
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    try:
        _report_open(world, db)
        yield world
    except sqlite3.Error as exc:
        _fail(_database_error(exc))
    finally:
        world.close()


def _report_open(world: World, db: Path) -> None:
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


def _owner(world: World, user_id: str) -> User:
    """The acting user, if active and an owner of their customer."""
    user = world.store.get_user(user_id)
    if user is None:
        _fail(f"unknown user {user_id!r}")
    if user.status is not UserStatus.ACTIVE:
        _fail(f"user {user_id!r} is not active")
    if user.role is not Role.OWNER:
        _fail(f"role {user.role.value!r} may not do this; owners only")
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
    """List the acting approver's customer's approvals, oldest first.

    Read-only: an approval past its expiry is shown as "expired (pending sweep)" and counts
    as expired for --status, but only `approvals expire` marks it."""
    with _world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = _approver(world, as_user)
        now = world.clock.now()
        approvals = [
            approval
            for approval in world.gate.approvals.list(user.customer_id)
            if status is None or _effective_status(approval, now) is status
        ]
    for approval in approvals:
        typer.echo(
            f"{approval.id}  {_shown_status(approval, now)}  {approval.tool}"
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
    """Show every field of one approval, including the exact args that would run.

    Read-only, like list: an overdue approval shows as "expired (pending sweep)"."""
    with _world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = _approver(world, as_user)
        now = world.clock.now()
        approval = world.gate.approvals.get(approval_id)
    # Another customer's approval reads exactly like a missing one.
    if approval is None or approval.customer_id != user.customer_id:
        _fail("approval not found")
    for line in _describe(approval, now):
        typer.echo(line)


# Statuses an approval leaves when its expiry passes (ApprovalQueue.expire_due).
_EXPIRING = frozenset({ApprovalStatus.PENDING, ApprovalStatus.APPROVED})
PENDING_SWEEP = "expired (pending sweep)"


def _effective_status(approval: Approval, now: datetime) -> ApprovalStatus:
    """The status the approval has in effect: expired once past its expiry, swept or not.
    The queue refuses to decide or run an overdue approval either way."""
    if approval.status in _EXPIRING and now >= approval.expires_at:
        return ApprovalStatus.EXPIRED
    return approval.status


def _shown_status(approval: Approval, now: datetime) -> str:
    effective = _effective_status(approval, now)
    if effective is not approval.status:
        return PENDING_SWEEP
    return approval.status.value


def _describe(approval: Approval, now: datetime) -> list[str]:
    fields = [
        ("id", approval.id),
        ("customer", approval.customer_id),
        ("status", _shown_status(approval, now)),
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


# users


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
    with _world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = world.store.get_user(as_user)
        if user is None:
            _fail(f"unknown user {as_user!r}")
        if user.status is not UserStatus.ACTIVE:
            _fail(f"user {as_user!r} is not active")
        # Another customer reads exactly like a missing one.
        if customer is not None and customer != user.customer_id:
            _fail(f"customer {customer!r} not found")
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
    with _world(db, audit_db, policies, fixtures) as world:
        try:
            suspended = suspend_user(world, user_id, acting_user_id=as_user, reason=reason)
        except UserAdminError as exc:
            _fail(str(exc))
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
    with _world(db, audit_db, policies, fixtures) as world:
        try:
            reinstated = reinstate_user(world, user_id, acting_user_id=as_user, reason=reason)
        except UserAdminError as exc:
            _fail(str(exc))
    typer.echo(f"reinstated {reinstated.id}  status {reinstated.status.value}")


# spend


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
    with _world(db, audit_db, policies, fixtures, reservation_lease=None) as world:
        owner = _owner(world, as_user)
        released = world.gate.release_stale(timedelta(minutes=older_than), actor=owner)
    typer.echo(f"released {released} stale reservations older than {older_than} minutes")


# demo


@app.command("demo")
def demo(
    audit_dir: Annotated[
        Path | None,
        typer.Option(
            "--audit-dir", help="Keep the Panoptes audit database here instead of discarding it."
        ),
    ] = None,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Run every demo scenario offline. Exits 1 if any scenario misbehaves."""
    try:
        passed = run_demo(sys.stdout, audit_dir=audit_dir, policies=policies, fixtures=fixtures)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        _fail(str(exc))
    if audit_dir is not None:
        path = audit_dir / AUDIT_FILE
        typer.echo(f"audit kept at {path}; verify with: agentgate audit verify --db {path}")
    if not passed:
        raise typer.Exit(code=1)


# agent


class ModelChoice(StrEnum):
    MOCK = "mock"
    ANTHROPIC = "anthropic"


ANTHROPIC_INSTALL = "install the anthropic extra: uv sync --extra anthropic"


def _model(choice: ModelChoice) -> Model:
    if choice is ModelChoice.MOCK:
        return NaiveModel()
    try:
        # Imported only when asked for, so the mock path never needs the SDK.
        from agentgate.agent.anthropic_model import AnthropicModel
    except ImportError:
        _fail(ANTHROPIC_INSTALL)
    try:
        return AnthropicModel.from_env()
    except ValueError as exc:
        _fail(str(exc))


@agent_app.command("run")
def agent_run(
    as_user: AsOption,
    task: Annotated[str, typer.Option("--task", help="What the user asks the agent to do.")],
    model: Annotated[
        ModelChoice,
        typer.Option(
            "--model",
            envvar="AGENTGATE_MODEL",
            help="mock (offline, NaiveModel) or anthropic (needs the extra and credentials).",
        ),
    ] = ModelChoice.MOCK,
    max_steps: Annotated[
        int, typer.Option("--max-steps", min=1, help="Model calls before the run stops.")
    ] = 12,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Run one task as a user in a fresh in-memory world and print every gate decision."""
    typer.echo(DEMO_IDENTITY_NOTE, err=True)
    agent_model = _model(model)
    sink = MemorySink()
    try:
        world = open_memory_world(policies=policies, fixtures=fixtures, sink=sink)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        _fail(str(exc))
    with world, _database_errors():
        user = world.store.get_user(as_user)
        if user is None:
            _fail(f"unknown user {as_user!r}")
        customer = world.store.get_customer(user.customer_id)
        engine = world.engines.get(user.customer_id)
        if customer is None or engine is None:
            _fail(f"customer {user.customer_id!r} has no policy")
        try:
            ctx = world.gate.start_run(user.customer_id, user.id)
        except IdentityError as err:
            _fail(f"{err.reason}: {err}")
        names = Names()
        typer.echo(
            f"acting: {user.id} (role {user.role.value}, customer {user.customer_id}),"
            f" {names.run(ctx.run_id)}, model {agent_model.name}"
        )
        typer.echo(f"task: {task}")
        transcript = run_agent(
            world.gate,
            ctx,
            agent_model,
            task,
            specs=tool_specs(engine, default_tools()),
            company=customer.name,
            max_steps=max_steps,
        )
        for line in format_transcript(transcript, sink.events, names, world.store):
            typer.echo(line)


@contextmanager
def _database_errors() -> Iterator[None]:
    try:
        yield
    except sqlite3.Error as exc:
        _fail(_database_error(exc))


# gateway


@gateway_app.command("serve")
def gateway_serve(
    as_user: AsOption,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Serve MCP over stdio for one run of the acting user until the client disconnects.

    The run starts before anything is served, so an unknown or suspended user exits 1
    without a byte on stdout. stdout carries only the MCP stream; everything else, the
    banner included, goes to stderr. A suspension (`users suspend`, or the fixtures on a
    later open) refuses the run's next call.
    """
    # Imported here, so the other commands do not pay for loading the MCP SDK.
    from agentgate.gateway import gateway_tools, serve_stdio

    # Record what changed on open (a newly stored policy version, a fixture change, reclaimed
    # spend), so a policy the gateway loads first is on the chain before its run's events.
    with _world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = world.store.get_user(as_user)
        if user is None:
            _fail(f"unknown user {as_user!r}")
        try:
            ctx = world.gate.start_run(user.customer_id, user.id)
        except IdentityError as err:
            _fail(f"{err.reason}: {err}")
        tools = gateway_tools(world, ctx)
        typer.echo(
            f"gateway: run {ctx.run_id} for {user.id} ({user.customer_id}), {len(tools)} tools",
            err=True,
        )
        try:
            serve_stdio(world, ctx)
        except KeyboardInterrupt:
            raise typer.Exit(code=130) from None


# red team

DEFAULT_REDTEAM_JSON = Path("reports/redteam.json")
DEFAULT_REDTEAM_MD = Path("reports/redteam.md")
# Where a run over each transport writes its reports unless --json and --md say otherwise.
REDTEAM_REPORTS: dict[Transport, tuple[Path, Path]] = {
    Transport.GATE: (DEFAULT_REDTEAM_JSON, DEFAULT_REDTEAM_MD),
    Transport.MCP: (Path("reports/redteam-mcp.json"), Path("reports/redteam-mcp.md")),
}
DEFAULT_BENCH_JSON = Path("reports/bench.json")
DEFAULT_BENCH_MD = Path("reports/bench.md")
JsonOption = Annotated[Path, typer.Option("--json", help="Write the JSON report here.")]
MdOption = Annotated[Path, typer.Option("--md", help="Write the markdown report here.")]


def _command(name: str, *options: tuple[str, object | None, object]) -> str:
    """The command as typed, from its options: each (flag, value, default) whose value is
    not the default. A True value is a bare flag; a list repeats the flag."""
    parts = ["agentgate", name]
    for flag, value, default in options:
        if value is None or value == default:
            continue
        if value is True:
            parts.append(flag)
        elif isinstance(value, list):
            for item in value:
                parts += [flag, str(item)]
        else:
            parts += [flag, str(value)]
    return shlex.join(parts)


def _write_reports(json_path: Path, md_path: Path, data: object, markdown: str) -> None:
    try:
        redteam_report.write_text(json_path, json.dumps(data, indent=2, ensure_ascii=True) + "\n")
        redteam_report.write_text(md_path, markdown)
    except OSError as exc:
        _fail(str(exc))
    typer.echo(f"reports written: {json_path}, {md_path}")


def _scenario_lines(result: SuiteResult) -> list[str]:
    lines: list[str] = []
    for scenario in result.results:
        mark = "ok  " if scenario.passed else "FAIL"
        lines.append(f"{mark} {scenario.id}  {scenario.title}")
        for check in scenario.checks:
            if not check.passed:
                detail = f": {check.detail}" if check.detail else ""
                lines.append(f"       check failed: {check.name}{detail}")
        lines += [f"       leak: {leak}" for leak in scenario.leaks]
        if scenario.error is not None:
            lines.append(f"       error: {scenario.error}")
    return lines


@app.command("redteam")
def redteam(
    category: Annotated[
        list[Category] | None,
        typer.Option("--category", help="Only this category; repeat for more."),
    ] = None,
    transport: Annotated[
        Transport,
        typer.Option(
            "--transport",
            help=(
                "gate: call the gate in-process. mcp: send every agent-side tool call through"
                " an MCP client session to the gateway."
            ),
        ),
    ] = Transport.GATE,
    json_path: Annotated[
        Path | None,
        typer.Option(
            "--json",
            help="Write the JSON report here. (default: reports/redteam.json, or"
            " reports/redteam-mcp.json with --transport mcp)",
            show_default=False,
        ),
    ] = None,
    md_path: Annotated[
        Path | None,
        typer.Option(
            "--md",
            help="Write the markdown report here. (default: reports/redteam.md, or"
            " reports/redteam-mcp.md with --transport mcp)",
            show_default=False,
        ),
    ] = None,
    baseline: Annotated[
        Path, typer.Option("--baseline", help="Scenarios that must exist and pass.")
    ] = DEFAULT_BASELINE,
    write: Annotated[
        bool,
        typer.Option(
            "--write-baseline",
            help="Write the baseline from this run. Refused if any scenario failed.",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="List every scenario and its failed checks.")
    ] = False,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Run the red-team suite. Exits 1 if any scenario failed or the baseline check found
    problems. The baseline applies to both transports."""
    if write and category:
        _fail("--write-baseline needs the whole suite; drop --category")
    default_json, default_md = REDTEAM_REPORTS[transport]
    command = _command(
        "redteam",
        ("--category", [c.value for c in category or []], []),
        ("--transport", transport.value, Transport.GATE.value),
        ("--json", json_path, None),
        ("--md", md_path, None),
        ("--baseline", baseline, DEFAULT_BASELINE),
        ("--write-baseline", write, False),
        ("--verbose", verbose, False),
        ("--policies", policies, DEFAULT_POLICIES),
        ("--fixtures", fixtures, DEFAULT_FIXTURES),
    )
    json_path = default_json if json_path is None else json_path
    md_path = default_md if md_path is None else md_path
    try:
        result = run_suite(
            categories=category or None,
            policies=policies,
            fixtures=fixtures,
            transport=transport,
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        _fail(str(exc))
    meta = redteam_report.collect_meta(command)
    markdown = redteam_report.to_markdown(result, meta)
    if verbose:
        for line in _scenario_lines(result):
            typer.echo(line)
        typer.echo("")
    typer.echo(markdown, nl=False)
    typer.echo("")
    _write_reports(json_path, md_path, redteam_report.to_json(result, meta), markdown)

    if write:
        if not result.all_passed:
            _fail(f"refusing to write {baseline}: {len(result.failed)} scenarios failed")
        _write_baseline(baseline, result)
        return

    problems = _baseline_problems(baseline, result, category)
    if not result.all_passed or problems:
        raise typer.Exit(code=1)


def _baseline_problems(
    path: Path, result: SuiteResult, categories: list[Category] | None
) -> list[str] | None:
    if not path.is_file():
        typer.echo(f"warning: baseline {path} not found; baseline check skipped", err=True)
        return None
    try:
        loaded = load_baseline(path)
    except (OSError, BaselineError) as exc:
        _fail(str(exc))
    problems = check_against_baseline(result, loaded, categories=categories or None)
    for problem in problems:
        typer.echo(f"BASELINE {problem}")
    typer.echo(f"baseline {path}: {len(problems)} problems")
    return problems


def _write_baseline(path: Path, result: SuiteResult) -> None:
    previous = None
    if path.is_file():
        try:
            previous = load_baseline(path)
        except (OSError, BaselineError):
            previous = None
    written = baseline_from(result)
    try:
        write_baseline(path, written)
    except OSError as exc:
        _fail(str(exc))
    if previous is not None:
        added = sorted(written.ids - previous.ids)
        removed = sorted(previous.ids - written.ids)
        typer.echo(f"baseline added: {', '.join(added) or 'none'}")
        typer.echo(f"baseline removed: {', '.join(removed) or 'none'}")
    typer.echo(f"baseline written: {path}  {len(written.ids)} scenarios")


@app.command("bench")
def bench(
    iterations: Annotated[
        int, typer.Option("--iterations", min=1, help="Timed calls per subject.")
    ] = redteam_bench.DEFAULT_ITERATIONS,
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Untimed calls per subject first.")
    ] = redteam_bench.DEFAULT_WARMUP,
    busy_day: Annotated[
        int,
        typer.Option(
            "--busy-day",
            min=0,
            help="Reservations already in the customer day for the busy-day section; 0 skips.",
        ),
    ] = redteam_bench.DEFAULT_BUSY_DAY,
    json_path: JsonOption = DEFAULT_BENCH_JSON,
    md_path: MdOption = DEFAULT_BENCH_MD,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Measure the latency the gate adds to a tool call. Times are machine-dependent."""
    command = _command(
        "bench",
        ("--iterations", iterations, redteam_bench.DEFAULT_ITERATIONS),
        ("--warmup", warmup, redteam_bench.DEFAULT_WARMUP),
        ("--busy-day", busy_day, redteam_bench.DEFAULT_BUSY_DAY),
        ("--json", json_path, DEFAULT_BENCH_JSON),
        ("--md", md_path, DEFAULT_BENCH_MD),
        ("--policies", policies, DEFAULT_POLICIES),
        ("--fixtures", fixtures, DEFAULT_FIXTURES),
    )
    try:
        result = redteam_bench.run_bench(
            iterations=iterations,
            warmup=warmup,
            busy_day=busy_day,
            policies=policies,
            fixtures=fixtures,
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        _fail(str(exc))
    meta = redteam_report.collect_meta(command)
    markdown = redteam_bench.to_markdown(result, meta)
    typer.echo(markdown, nl=False)
    typer.echo("")
    _write_reports(json_path, md_path, redteam_bench.to_json(result, meta), markdown)


def main() -> None:
    app()
