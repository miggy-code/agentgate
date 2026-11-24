"""`agentgate policy`: lint the policies and run their tests."""

from pathlib import Path
from typing import Annotated

import typer

from agentgate.cli._shared import (
    FixturesOption,
    PoliciesOption,
    fail,
)
from agentgate.fixtures import FixtureError
from agentgate.policy.schema import PolicyError
from agentgate.policy.testing import load_policy_bundles, run_policy_tests
from agentgate.policy.version import bundle_version, git_revision

policy_app = typer.Typer(help="Validate and test customer policies.", no_args_is_help=True)


@policy_app.command("lint")
def lint(policies: PoliciesOption = Path("policies")) -> None:
    """Load and validate every policy bundle."""
    try:
        bundles = load_policy_bundles(policies)
    except PolicyError as exc:
        fail(str(exc))
    for customer_id in sorted(bundles):
        bundle = bundles[customer_id]
        line = (
            f"{customer_id}  {bundle_version(bundle)}"
            f"  {len(bundle.policy.rules)} rules  {len(bundle.tools)} tools"
        )
        if bundle.policy.models is not None:
            line += f"  {len(bundle.policy.models.allow)} models"
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
        fail(str(exc))
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
