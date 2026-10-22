"""The agentgate command line: policy lint, policy test and audit verify.

Output is plain, deterministic text (customers sorted, cases in file order, no colour), so
it reads the same in a terminal, a CI log and a diff. Problems that stop a command go to
stderr as one "error: ..." line with exit status 1.
"""

import sqlite3
from pathlib import Path
from typing import Annotated, NoReturn

import typer

import agentgate
from agentgate.audit.panoptes import PanoptesSink
from agentgate.fixtures import FixtureError
from agentgate.policy.schema import PolicyError
from agentgate.policy.testing import load_policy_bundles, run_policy_tests
from agentgate.policy.version import bundle_version, git_revision

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


def main() -> None:
    app()
