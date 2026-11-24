"""`agentgate gateway serve`: a user's tools behind an MCP server."""

from pathlib import Path
from typing import Annotated

import typer

from agentgate.cli._shared import (
    AsOption,
    AuditDbOption,
    DbOption,
    FixturesOption,
    PoliciesOption,
    fail,
    open_cli_world,
)
from agentgate.cli.model_proxy import KeysFileOption, proxy_config
from agentgate.gate import IdentityError
from agentgate.model.pricing import RateTableError
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    OpenRecord,
)

gateway_app = typer.Typer(
    help="Serve a user's tools to an MCP client, every call through the gate.",
    no_args_is_help=True,
)


@gateway_app.command("serve")
def gateway_serve(
    as_user: AsOption,
    model_proxy: Annotated[
        bool,
        typer.Option(
            "--model-proxy",
            help="Also serve model APIs for this run, so tool and model calls share its caps. "
            "Needs --upstream, --rate-table and --token-file.",
        ),
    ] = False,
    upstream: Annotated[
        list[str] | None,
        typer.Option(
            "--upstream", metavar="PROVIDER=URL", help="With --model-proxy: as model-proxy serve."
        ),
    ] = None,
    rate_table: Annotated[
        Path | None, typer.Option("--rate-table", help="With --model-proxy: the rate table.")
    ] = None,
    keys_file: KeysFileOption = None,
    token_file: Annotated[
        Path | None,
        typer.Option(
            "--token-file",
            help="With --model-proxy: write the run's token to this new file (mode 0600). "
            "stdout is the MCP stream, so the token is never printed.",
        ),
    ] = None,
    model_port: Annotated[
        int, typer.Option("--model-port", help="With --model-proxy: port to bind (0 picks one).")
    ] = 0,
    max_usd: Annotated[
        str | None,
        typer.Option("--max-usd", help="With --model-proxy: cap the run's model spend."),
    ] = None,
    max_calls: Annotated[
        int | None,
        typer.Option("--max-calls", min=1, help="With --model-proxy: cap the run's model calls."),
    ] = None,
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

    With --model-proxy the same run also serves model APIs on a loopback port (see
    `model-proxy serve`): its model calls reserve against the same per-run and per-day caps as
    its tool calls, and its token is written to --token-file.
    """
    # Imported here, so the other commands do not pay for loading the MCP SDK.
    from agentgate.gateway import gateway_tools, serve_stdio

    config = None
    if model_proxy:
        if not upstream or rate_table is None or token_file is None:
            fail("--model-proxy needs --upstream, --rate-table and --token-file")
        config = proxy_config(
            upstream,
            keys_file,
            rate_table,
            port=model_port,
            max_usd=max_usd,
            max_calls=max_calls,
            token_file=token_file,
        )
    elif (
        any(
            value is not None
            for value in (upstream, rate_table, keys_file, token_file, max_usd, max_calls)
        )
        or model_port
    ):
        fail("model options need --model-proxy")

    # Record what changed on open (a newly stored policy version, a fixture change, reclaimed
    # spend), so a policy the gateway loads first is on the chain before its run's events.
    with open_cli_world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        user = world.store.get_user(as_user)
        if user is None:
            fail(f"unknown user {as_user!r}")
        try:
            ctx = world.gate.start_run(user.customer_id, user.id)
        except IdentityError as err:
            fail(f"{err.reason}: {err}")
        tools = gateway_tools(world, ctx)
        typer.echo(
            f"gateway: run {ctx.run_id} for {user.id} ({user.customer_id}), {len(tools)} tools",
            err=True,
        )
        proxy = None
        if config is not None:
            from agentgate.model_proxy.proxy import ModelProxy, ProxySetupError

            proxy = ModelProxy(world, user.id, config)
            try:
                bound = proxy.start(ctx)
            except ProxySetupError as exc:
                fail(str(exc))
            except (OSError, RateTableError) as exc:
                fail(f"cannot start the model proxy: {exc}")
            typer.echo(
                f"model proxy: same run, http://{config.host}:{bound}, token in {token_file}",
                err=True,
            )
        try:
            serve_stdio(world, ctx)
        except KeyboardInterrupt:
            raise typer.Exit(code=130) from None
        finally:
            if proxy is not None:
                proxy.stop()
