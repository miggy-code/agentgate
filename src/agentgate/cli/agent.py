"""`agentgate demo` and `agentgate agent run`: the reference agent through the gate."""

import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from agentgate.agent.loop import run_agent, tool_specs
from agentgate.agent.mock import NaiveModel
from agentgate.agent.model import Model
from agentgate.audit.base import (
    MemorySink,
)
from agentgate.cli._shared import (
    DEMO_IDENTITY_NOTE,
    AsOption,
    FixturesOption,
    PoliciesOption,
    database_error,
    fail,
)
from agentgate.demo import AUDIT_FILE, Names, format_transcript, run_demo
from agentgate.gate import IdentityError
from agentgate.tools.builtin import default_tools
from agentgate.world import (
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    open_memory_world,
)

agent_app = typer.Typer(help="Run the reference agent through the gate.", no_args_is_help=True)


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
        fail(str(exc))
    if audit_dir is not None:
        path = audit_dir / AUDIT_FILE
        typer.echo(f"audit kept at {path}; verify with: agentgate audit verify --db {path}")
    if not passed:
        raise typer.Exit(code=1)


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
        fail(ANTHROPIC_INSTALL)
    try:
        return AnthropicModel.from_env()
    except ValueError as exc:
        fail(str(exc))


@contextmanager
def _database_errors() -> Iterator[None]:
    try:
        yield
    except sqlite3.Error as exc:
        fail(database_error(exc))


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
        fail(str(exc))
    with world, _database_errors():
        user = world.store.get_user(as_user)
        if user is None:
            fail(f"unknown user {as_user!r}")
        customer = world.store.get_customer(user.customer_id)
        engine = world.engines.get(user.customer_id)
        if customer is None or engine is None:
            fail(f"customer {user.customer_id!r} has no policy")
        try:
            ctx = world.gate.start_run(user.customer_id, user.id)
        except IdentityError as err:
            fail(f"{err.reason}: {err}")
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
