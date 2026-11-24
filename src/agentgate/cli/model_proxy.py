"""`agentgate model-proxy`, `doctor model-proxy` and `init-experiment`: serve model APIs
through the gate, check that a proxy can start, and set up a one-customer experiment."""

import contextlib
import json
import os
import signal
import threading
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import JsonValue

from agentgate.experiment import (
    DEFAULT_CUSTOMER,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_USER,
    ExperimentError,
    ExperimentModel,
    unlink_if_same,
    write_experiment,
)
from agentgate.model.admission import Caps
from agentgate.model.keys import KeyConfigError, load_provider_keys
from agentgate.model.pricing import RateTableError
from agentgate.model.tokens import (
    DEFAULT_TTL,
    MAX_TTL,
    write_private_file,
)
from agentgate.model_proxy.proxy import ProxyConfig
from agentgate.model_proxy.server import DEFAULT_MAX_TOKEN_TTL
from agentgate.policy.schema import ModelProvider
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    OpenRecord,
)

if TYPE_CHECKING:
    from agentgate.model_proxy.proxy import ModelProxy, ProxyConfig
from agentgate.cli._shared import (
    DEMO_IDENTITY_NOTE,
    AsOption,
    AuditDbOption,
    DbOption,
    FixturesOption,
    PoliciesOption,
    fail,
    open_cli_world,
)

model_proxy_app = typer.Typer(
    help="Serve model APIs to a run's client: every request through the gate.",
    no_args_is_help=True,
)


doctor_app = typer.Typer(help="Check that a service can start.", no_args_is_help=True)


UpstreamOption = Annotated[
    list[str],
    typer.Option(
        "--upstream",
        metavar="PROVIDER=URL",
        help="The origin serving a provider (anthropic, openai, openai_compatible, gemini), "
        "e.g. openai_compatible=https://api.deepseek.com. One per wire protocol; repeat it.",
    ),
]


RateTableOption = Annotated[
    Path,
    typer.Option(
        "--rate-table",
        help="Panoptes-format rate table (JSON). It prices the reservation and the settlement.",
    ),
]


KeysFileOption = Annotated[
    Path | None,
    typer.Option(
        "--keys-file",
        help="Owner-only JSON of provider keys. Without it, AGENTGATE_<PROVIDER>_API_KEY.",
    ),
]


def proxy_config(
    upstream: list[str],
    keys_file: Path | None,
    rate_table: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    max_usd: str | None = None,
    max_calls: int | None = None,
    ttl_minutes: int | None = None,
    token_file: Path | None = None,
) -> "ProxyConfig":
    from agentgate.model_proxy.proxy import ProxyConfig

    upstreams: dict[ModelProvider, str] = {}
    for item in upstream:
        name, _, url = item.partition("=")
        try:
            provider = ModelProvider(name)
        except ValueError:
            fail(f"unknown provider in --upstream (expected PROVIDER=URL): {name!r}")
        if not url or provider in upstreams:
            fail("--upstream is PROVIDER=URL, once per provider")
        upstreams[provider] = url
    try:
        keys = load_provider_keys(keys_file)
        caps = Caps(
            max_usd=None if max_usd is None else Decimal(max_usd),
            max_calls=max_calls,
        )
    except (KeyConfigError, ArithmeticError, ValueError) as exc:
        fail(str(exc))
    return ProxyConfig(
        upstreams=upstreams,
        keys=keys,
        rate_table=rate_table,
        host=host,
        port=port,
        caps=caps,
        token_file=token_file,
        token_ttl=DEFAULT_TTL if ttl_minutes is None else timedelta(minutes=ttl_minutes),
    )


@model_proxy_app.command("serve")
def model_proxy_serve(
    as_user: AsOption,
    upstream: UpstreamOption,
    rate_table: RateTableOption,
    keys_file: KeysFileOption = None,
    host: Annotated[str, typer.Option("--host", help="Loopback address to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind (0 picks a free one).")] = 0,
    token_file: Annotated[
        Path | None,
        typer.Option(
            "--token-file", help="Write the run's token to this new file (mode 0600), not stdout."
        ),
    ] = None,
    ttl_minutes: Annotated[
        int | None, typer.Option("--ttl-minutes", min=1, help="The token's lifetime, at most.")
    ] = None,
    max_usd: Annotated[
        str | None, typer.Option("--max-usd", help="Cap the run's model spend below the policy's.")
    ] = None,
    max_calls: Annotated[
        int | None,
        typer.Option("--max-calls", min=1, help="Cap the run's calls below the policy's."),
    ] = None,
    ready_file: Annotated[
        Path | None,
        typer.Option(
            "--ready-file",
            help="Once listening, write a new JSON file (mode 0600) with the url, port, run id, "
            "user and the caps in effect. Removed on stop.",
        ),
    ] = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Serve model APIs for one run of the acting user until interrupted.

    The client sends the run's token where it would send its provider key; the proxy
    decides each request against the policy's models section, reserves its worst-case cost,
    swaps in the provider's key, and settles what the provider reports. Everything, the
    gate's decisions and Panoptes' observation of each exchange, goes to one audit chain.
    Stopping drains in-flight exchanges, then ends the run and revokes its token.

    --ready-file tells a harness the proxy is up without parsing stderr: it appears, complete,
    only once the port is listening (and the token file is written), and states the caps the
    proxy enforces, the policy's tightened by --max-usd and --max-calls."""
    from agentgate.model_proxy.proxy import ModelProxy, ProxySetupError

    if ready_file is not None and (ready_file.exists() or ready_file.is_symlink()):
        fail(f"--ready-file {ready_file} already exists")

    config = proxy_config(
        upstream,
        keys_file,
        rate_table,
        host=host,
        port=port,
        max_usd=max_usd,
        max_calls=max_calls,
        ttl_minutes=ttl_minutes,
        token_file=token_file,
    )
    with open_cli_world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        proxy = ModelProxy(world, as_user, config)
        try:
            bound = proxy.start()
        except ProxySetupError as exc:
            fail(str(exc))
        except (OSError, RateTableError) as exc:
            fail(f"cannot start: {exc}")
        run_id = proxy.run.run_id
        ready_fd: int | None = None
        try:
            # A supervisor stops a service with SIGTERM: end the run as for an interrupt.
            # Installed before the ready file exists, so a SIGTERM sent the moment a harness
            # sees it still cleans it up.
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            typer.echo(
                f"model proxy: run {proxy.run.run_id} for {as_user}, http://{host}:{bound}",
                err=True,
            )
            if proxy.token is not None:
                typer.echo(proxy.token)
            else:
                typer.echo(f"token written to {token_file}", err=True)
            if ready_file is not None:
                try:
                    info = _ready_info(proxy, as_user, host, bound, token_file)
                    ready_fd = _write_ready_file(ready_file, info)
                except OSError as exc:
                    fail(f"cannot write {ready_file}: {exc.strerror}")
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            # A second SIGTERM must not interrupt the cleanup the first one started.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                proxy.stop()
            finally:
                if ready_file is not None and ready_fd is not None:
                    try:
                        _remove_own_ready_file(ready_file, run_id, ready_fd)
                    finally:
                        os.close(ready_fd)


def _ready_info(
    proxy: "ModelProxy", user_id: str, host: str, port: int, token_file: Path | None
) -> dict[str, JsonValue]:
    """What --ready-file states: where the proxy listens, for which run, under which caps."""
    limits = proxy.limits()
    return {
        "url": f"http://{host}:{port}",
        "host": host,
        "port": port,
        "pid": os.getpid(),
        "run_id": proxy.run.run_id,
        "user": user_id,
        "customer": proxy.run.customer_id,
        # Amounts are decimal strings, as in the policy, so no float rounding creeps in.
        "max_usd": _plain_decimal(limits.per_run_usd),
        "max_calls": limits.max_calls_per_run,
        "per_customer_day_usd": _plain_decimal(limits.per_customer_day_usd),
        "token_file": None if token_file is None else str(token_file),
    }


def _plain_decimal(value: Decimal) -> str:
    return "0" if value.is_zero() else format(value.normalize(), "f")


def _remove_own_ready_file(path: Path, run_id: str, fd: int, *, key: str = "run_id") -> None:
    """Remove path only if it is still the file this process wrote for this run (or, with
    key, whatever the file names under that key).

    The file must be the one still open as fd since publishing and name this pid and run, so
    another process's replacement is left alone (see `unlink_if_same` for the remaining
    microsecond window). A ready file can still outlive the process (SIGKILL, a crash):
    consumers must treat it as a hint and check that pid is alive and url answers before
    relying on it."""
    with contextlib.suppress(OSError, ValueError):
        if path.is_symlink():
            return
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("pid") == os.getpid() and info.get(key) == run_id:
            unlink_if_same(path, fd)


def _write_ready_file(path: Path, info: dict[str, JsonValue]) -> int:
    """Publish info at path atomically and return a read-only descriptor of it.

    A reader that sees the file sees all of it. The caller keeps the descriptor open until
    cleanup, so the published inode cannot be reused by a replacement, and closes it.

    The JSON goes to a private temporary file beside path, which is then hard-linked into
    place. A link never replaces an existing file, so path is still created exclusively.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_private_file(temporary, json.dumps(info, indent=2) + "\n")
    try:
        fd = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.link(temporary, path)
        except BaseException:
            os.close(fd)
            raise
        return fd
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


@model_proxy_app.command("server")
def model_proxy_server(
    upstream: UpstreamOption,
    rate_table: RateTableOption,
    keys_file: Annotated[
        Path, typer.Option("--keys-file", help="Owner-only JSON of provider keys (required).")
    ],
    host: Annotated[str, typer.Option("--host", help="Loopback address to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind (0 picks a free one).")] = 0,
    control_socket: Annotated[
        Path | None,
        typer.Option(
            "--control-socket",
            help="Path of the control socket. Its directory must not exist: it is created "
            "0700 and removed on stop. Default: a new private temporary directory.",
        ),
    ] = None,
    max_token_ttl: Annotated[
        int,
        typer.Option(
            "--max-token-ttl",
            min=1,
            help="Longest lifetime of a run's token, in seconds, and its default.",
        ),
    ] = int(DEFAULT_MAX_TOKEN_TTL.total_seconds()),
    ready_file: Annotated[
        Path | None,
        typer.Option(
            "--ready-file",
            help="Once listening, write a new JSON file (mode 0600) with the model url, the "
            "control socket, the pid and the instance id. Removed on stop.",
        ),
    ] = None,
    identity_config: Annotated[
        Path | None,
        typer.Option(
            "--identity-config",
            help="Trust configuration (YAML): the issuers whose signed identity assertions "
            "start runs. With it, a start naming a user is refused (assertion_required).",
        ),
    ] = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Serve model APIs for many runs of one world until interrupted.

    A host on this machine starts and ends runs on the control socket (see
    agentgate.model_proxy.client.ControlClient): each start returns the run's token, which its
    client sends to the model url in place of a provider key. Only processes of the user
    running the server can reach the socket. Each request is decided as with `model-proxy
    serve`, under the caps its run was started with. Stopping (SIGTERM or an interrupt) drains
    in-flight requests, then ends every run this server started and revokes their tokens.
    Runs of a server that died are ended when the next one starts.

    With --identity-config the host hands over a signed identity assertion instead of naming
    the user, and the server verifies it before it starts the run (see agentgate.identity)."""
    from agentgate.identity import IdentityConfigError, TrustConfig, load_identity_config
    from agentgate.model_proxy.proxy import ProxySetupError
    from agentgate.model_proxy.server import MultiRunProxy, ServerConfig

    if ready_file is not None and (ready_file.exists() or ready_file.is_symlink()):
        fail(f"--ready-file {ready_file} already exists")
    if max_token_ttl > MAX_TTL.total_seconds():
        fail(f"--max-token-ttl is at most {int(MAX_TTL.total_seconds())} seconds")
    trust: TrustConfig | None = None
    note = DEMO_IDENTITY_NOTE
    if identity_config is not None:
        try:
            trust = load_identity_config(identity_config)
        except IdentityConfigError as exc:
            fail(str(exc))
        names = ", ".join(issuer.name for issuer in trust.issuers)
        note = f"note: runs start only from a verified identity assertion (issuers: {names})"
    base = proxy_config(upstream, keys_file, rate_table, host=host, port=port)
    config = ServerConfig(
        upstreams=dict(base.upstreams),
        keys=base.keys,
        rate_table=rate_table,
        host=host,
        port=port,
        control_socket=control_socket,
        max_token_ttl=timedelta(seconds=max_token_ttl),
        identity=trust,
    )
    with open_cli_world(
        db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES, identity_note=note
    ) as world:
        server = MultiRunProxy(world, config)
        try:
            _, socket_path = server.start()
        except ProxySetupError as exc:
            fail(str(exc))
        except ValueError as exc:
            # Panoptes refuses a non-loopback --host with a ValueError.
            fail(f"cannot start: {exc}")
        except (OSError, RateTableError) as exc:
            fail(f"cannot start: {exc}")
        ready_fd: int | None = None
        try:
            # As `serve`: SIGTERM ends the runs as an interrupt does, and is handled before
            # the ready file exists.
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            if server.recovered:
                typer.echo(f"note: ended {server.recovered} runs of stopped servers", err=True)
            for name, problem in sorted(server.identity_problems.items()):
                typer.echo(
                    f"warning: identity issuer {name}: keys not loaded ({problem});"
                    " its assertions are refused until they load",
                    err=True,
                )
            typer.echo(
                f"model proxy server {server.instance_id}: {server.model_url},"
                f" control socket {socket_path}",
                err=True,
            )
            if ready_file is not None:
                info: dict[str, JsonValue] = {
                    "model_url": server.model_url,
                    "control_socket": str(socket_path),
                    "pid": os.getpid(),
                    "instance_id": server.instance_id,
                }
                try:
                    ready_fd = _write_ready_file(ready_file, info)
                except OSError as exc:
                    fail(f"cannot write {ready_file}: {exc.strerror}")
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                server.stop()
            finally:
                if ready_file is not None and ready_fd is not None:
                    try:
                        _remove_own_ready_file(
                            ready_file, server.instance_id, ready_fd, key="instance_id"
                        )
                    finally:
                        os.close(ready_fd)


def init_experiment(
    directory: Annotated[Path, typer.Argument(help="Writes fixtures/ and policies/ here.")],
    model: Annotated[
        list[str],
        typer.Option(
            "--model",
            metavar="PROVIDER:NAME",
            help="A model the user may call, e.g. openai_compatible:deepseek-chat. Repeatable.",
        ),
    ],
    max_usd: Annotated[str, typer.Option("--max-usd", help="The run's spend cap (USD).")],
    max_calls: Annotated[
        int, typer.Option("--max-calls", min=1, max=10_000, help="The run's call cap.")
    ],
    day_usd: Annotated[
        str | None,
        typer.Option("--day-usd", help="The customer's daily spend cap (default: --max-usd)."),
    ] = None,
    max_output_tokens: Annotated[
        int, typer.Option("--max-output-tokens", min=1, help="Per-request output token cap.")
    ] = DEFAULT_MAX_OUTPUT_TOKENS,
    max_request_bytes: Annotated[
        int, typer.Option("--max-request-bytes", min=1, help="Per-request body size cap.")
    ] = DEFAULT_MAX_REQUEST_BYTES,
    customer: Annotated[str, typer.Option("--customer", help="Customer id.")] = DEFAULT_CUSTOMER,
    user: Annotated[str, typer.Option("--user", help="The one user's id.")] = DEFAULT_USER,
    allow_server_features: Annotated[
        bool,
        typer.Option(
            "--allow-server-features",
            help="Let requests carry provider-side features: prompt-cache writes, "
            "provider-run tools and provider-side state (background runs, stored "
            "conversations). Work the provider does on its side can cost more than the "
            "reservation computed from the request text (see docs/limits.md). Off by "
            "default; some agent CLIs need it.",
        ),
    ] = False,
) -> None:
    """Write a one-user customer and a models-only policy for an experiment.

    The policy has only spend and models sections: the user may call the listed models under
    the caps, every other model and every tool call is denied. Existing files are never
    overwritten. Then serve it with `model-proxy serve --as <user> --policies <dir>/policies
    --fixtures <dir>/fixtures ...`."""
    try:
        usd = Decimal(max_usd)
        day = None if day_usd is None else Decimal(day_usd)
    except ArithmeticError:
        fail("invalid amount: --max-usd and --day-usd are decimal USD amounts")
    try:
        written = write_experiment(
            directory,
            [ExperimentModel.parse(item) for item in model],
            max_usd=usd,
            max_calls=max_calls,
            day_usd=day,
            max_output_tokens=max_output_tokens,
            max_request_bytes=max_request_bytes,
            customer=customer,
            user=user,
            allow_server_features=allow_server_features,
        )
    except ExperimentError as exc:
        fail(str(exc))
    typer.echo(f"fixtures {written.fixtures}")
    typer.echo(f"policies {written.policies}")
    typer.echo(f"user {written.user}")


@doctor_app.command("model-proxy")
def doctor_model_proxy(
    as_user: AsOption,
    upstream: UpstreamOption,
    rate_table: RateTableOption,
    keys_file: KeysFileOption = None,
    db: DbOption = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Check that `model-proxy serve` would start, without starting it: one upstream per
    wire protocol, a provider key for each, a policy model for them, one rate table that
    prices the reservation and the settlement alike, and a Panoptes audit log."""
    from agentgate.model_proxy.proxy import ModelProxy, ProxySetupError

    config = proxy_config(upstream, keys_file, rate_table)
    with open_cli_world(db, audit_db, policies, fixtures, record_open=OpenRecord.CHANGES) as world:
        try:
            ModelProxy(world, as_user, config).validate()
        except ProxySetupError as exc:
            typer.echo(f"model-proxy: FAIL: {exc}")
            raise typer.Exit(code=1) from None
    typer.echo("model-proxy: ok")
