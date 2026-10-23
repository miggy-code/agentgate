"""The agentgate command line: policy lint and test, audit verify, the approval queue, the
demo, one-off agent runs, the red-team suite and the latency bench.

Output is plain, deterministic text (customers sorted, cases in file order, no colour), so
it reads the same in a terminal, a CI log and a diff. Problems that stop a command go to
stderr as one "error: ..." line with exit status 1.
"""

import json
import shlex
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

import agentgate
from agentgate.agent.loop import run_agent, tool_specs
from agentgate.agent.mock import NaiveModel
from agentgate.agent.model import Model
from agentgate.approvals import Approval, ApprovalError, ApprovalStatus
from agentgate.audit.base import MemorySink
from agentgate.audit.panoptes import PanoptesSink
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
from agentgate.tools.builtin import default_tools
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    World,
    open_memory_world,
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
agent_app = typer.Typer(help="Run the reference agent through the gate.", no_args_is_help=True)
app.add_typer(agent_app, name="agent")

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
    with world:
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


# red team

DEFAULT_REDTEAM_JSON = Path("reports/redteam.json")
DEFAULT_REDTEAM_MD = Path("reports/redteam.md")
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
    json_path: JsonOption = DEFAULT_REDTEAM_JSON,
    md_path: MdOption = DEFAULT_REDTEAM_MD,
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
    problems."""
    if write and category:
        _fail("--write-baseline needs the whole suite; drop --category")
    command = _command(
        "redteam",
        ("--category", [c.value for c in category or []], []),
        ("--json", json_path, DEFAULT_REDTEAM_JSON),
        ("--md", md_path, DEFAULT_REDTEAM_MD),
        ("--baseline", baseline, DEFAULT_BASELINE),
        ("--write-baseline", write, False),
        ("--verbose", verbose, False),
        ("--policies", policies, DEFAULT_POLICIES),
        ("--fixtures", fixtures, DEFAULT_FIXTURES),
    )
    try:
        result = run_suite(categories=category or None, policies=policies, fixtures=fixtures)
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
