"""The agentgate command line: policy lint and test, the audit log, the approval queue,
users, the spend ledger, model-proxy run tokens, the demo, one-off agent runs, the MCP
gateway, the model proxy, the red-team suite and the latency bench. One module per command
group; this one assembles them.

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

from typing import Annotated

import typer

import agentgate
from agentgate.cli import (
    agent,
    approvals,
    audit,
    gateway,
    model_proxy,
    policy,
    redteam,
    spend,
    tokens,
    users,
)
from agentgate.cli._shared import DEMO_IDENTITY_NOTE

__all__ = ["DEMO_IDENTITY_NOTE", "app", "main"]

app = typer.Typer(
    help="Permissions, approval and audit layer between an AI agent and a company's tools.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.add_typer(policy.policy_app, name="policy")
app.add_typer(audit.audit_app, name="audit")
app.add_typer(approvals.approvals_app, name="approvals")
app.add_typer(agent.agent_app, name="agent")
app.add_typer(gateway.gateway_app, name="gateway")
app.add_typer(users.users_app, name="users")
app.add_typer(spend.spend_app, name="spend")
app.add_typer(tokens.token_app, name="token")
app.add_typer(model_proxy.model_proxy_app, name="model-proxy")
app.add_typer(model_proxy.doctor_app, name="doctor")
app.command("demo")(agent.demo)
app.command("init-experiment")(model_proxy.init_experiment)
app.command("redteam")(redteam.redteam)
app.command("bench")(redteam.bench)


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


def main() -> None:
    app()
