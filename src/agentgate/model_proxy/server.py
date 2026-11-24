"""The multi-run model proxy: one long-running process serving model calls for many runs of
one world (`agentgate model-proxy server`).

It has two listeners. The model listener is the same Panoptes proxy with agentgate's hooks
as `model-proxy serve`, on a loopback port, and a client presents its run's token there in
place of a provider key. The control listener is a Unix domain socket on which a host on the
same machine starts a run for a user, gets back the run's token, and ends the run later.

The control socket has no authentication of its own: its directory is created new, mode
0700, and the socket is 0600, so only processes of the user running the server can reach it.
That is the same trust as `--as` on the command line. There is no TCP control port.

The proxy serves only the runs it started. A token of any other run of the world (a `serve`
process's, another server instance's) is refused as `not_served`. Each run is held to the
caps it was started with, stored in `run_limits` in the transaction that starts it, which
can only tighten the policy's. Which instance (and process) started a run is stored in
`served_runs`, so a server that dies without stopping has its runs ended by the next one to
start on the world; until then their tokens live until their TTL.
"""

import contextlib
import json
import logging
import os
import re
import secrets
import socket
import socketserver
import stat
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from panoptes.db import AuditDB
from panoptes.pricing import PricingTable
from panoptes.proxy import ProxyServer

from agentgate._json import no_duplicate_keys
from agentgate.audit.base import OperatorAction
from agentgate.context import RunContext
from agentgate.gate import IdentityError
from agentgate.model.admission import Caps
from agentgate.model.keys import ProviderKeys
from agentgate.model.pricing import RateTable
from agentgate.model.tokens import (
    MAX_TTL,
    TokenReason,
    TokenRecord,
    TokenRejected,
    TokenStore,
)
from agentgate.model_proxy.proxy import (
    SWEEP_INTERVAL,
    ModelGatewayHooks,
    ProxySetupError,
    audit_db,
    check_upstreams,
    load_rates,
    proxy_server,
    served_providers,
)
from agentgate.policy.schema import ModelProvider, SpendLimits
from agentgate.store import RunLimits, ServedRun
from agentgate.world import DEFAULT_RESERVATION_LEASE, World

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKEN_TTL = timedelta(hours=1)
DEFAULT_CONTROL_WORKERS = 16
SOCKET_NAME = "control.sock"
# A control request is a few hundred bytes; anything far larger is not one.
MAX_CONTROL_BODY = 4096
# How long a control connection may take to send its request.
CONTROL_TIMEOUT_SECONDS = 10.0
# How long stopping waits for in-flight control requests.
CONTROL_DRAIN_SECONDS = 10.0
_MAX_ID_LENGTH = 200


class RunEndReason(StrEnum):
    """Why the server ended a run, in its `run_ended` event."""

    # A host asked (`POST /runs/{run_id}/end`).
    REQUESTED = "requested"
    # The server stopped.
    SHUTDOWN = "shutdown"
    # The instance that started the run died; the next instance to start ended it.
    RECOVERED = "recovered"


class UnknownRun(LookupError):
    """The run is unknown, or this server did not start it."""


class ServerStopping(RuntimeError):
    """The server is stopping and starts no more runs."""


@dataclass(frozen=True)
class ServerConfig:
    # Origin of the upstream serving each policy provider (at most one per wire protocol).
    upstreams: dict[ModelProvider, str]
    keys: ProviderKeys
    rate_table: Path
    host: str = "127.0.0.1"
    port: int = 0
    # The control socket's path. Its directory must not exist: the server creates it 0700.
    # None: a new private directory under the system's temporary directory.
    control_socket: Path | None = None
    # The longest lifetime a run's token may be given, and the one it gets by default.
    max_token_ttl: timedelta = DEFAULT_MAX_TOKEN_TTL
    control_workers: int = DEFAULT_CONTROL_WORKERS
    lease: timedelta = DEFAULT_RESERVATION_LEASE
    sweep_interval: timedelta = SWEEP_INTERVAL


@dataclass(frozen=True)
class StartedRun:
    run: RunContext
    # Shown once: the store keeps only its hash.
    token: str
    expires_at: datetime
    # The run's caps in effect: the policy's spend section tightened by the run's caps.
    limits: SpendLimits

    def __repr__(self) -> str:
        return f"StartedRun(run={self.run!r}, expires_at={self.expires_at!r})"


class ServedRuns:
    """The multi-run RunSource: a token resolves to whatever run it names, if this server
    started that run, under the caps that run was started with."""

    def __init__(self, proxy: "MultiRunProxy") -> None:
        self._proxy = proxy

    def resolve(self, token: str | None) -> RunContext:
        ctx = self._proxy.tokens.resolve(token)
        if not self._proxy.serves(ctx.run_id):
            raise TokenRejected(TokenReason.NOT_SERVED, ctx.run_id)
        return ctx

    def caps(self, run: RunContext) -> Caps | None:
        stored = self._proxy.world.store.get_run_limits(run.run_id)
        if stored is None:
            return None
        return Caps(max_usd=stored.max_usd, max_calls=stored.max_calls)


class MultiRunProxy:
    """Model calls for many runs of one world. `start()` binds both listeners; `stop()`
    drains them and ends every run this instance started."""

    def __init__(self, world: World, config: ServerConfig) -> None:
        self.world = world
        self._config = config
        self.instance_id = secrets.token_hex(8)
        self.tokens = TokenStore(world.store, world.clock)
        self._lock = threading.Lock()
        # The runs this instance started and has not ended. An ended run leaves memory (its
        # tokens are revoked), so a long-running server does not grow with every run.
        self._live: dict[str, RunContext] = {}
        # Set once stop() has ended the live runs: a run registered after that is ended at
        # once.
        self._closed = False
        self._stopping = threading.Event()
        self._server: ProxyServer | None = None
        self._control: ControlServer | None = None
        self._control_thread: threading.Thread | None = None
        self._sweeper: threading.Thread | None = None
        self._socket_dir: Path | None = None
        self.socket_path: Path | None = None
        self.model_port = 0
        # Runs of dead instances this start ended.
        self.recovered = 0

    @property
    def model_url(self) -> str:
        return f"http://{self._config.host}:{self.model_port}"

    def validate(self) -> tuple[RateTable, PricingTable, AuditDB]:
        """Everything that must hold before the server serves, or ProxySetupError: one
        upstream per wire protocol and a key for each, a customer whose policy lists a model
        for the served providers, one rate table read both ways, a Panoptes audit log to
        share, and a token lifetime within MAX_TTL."""
        config = self._config
        check_upstreams(config.upstreams, config.keys)
        if not timedelta(0) < config.max_token_ttl <= MAX_TTL:
            raise ProxySetupError(f"the token lifetime must be above zero and at most {MAX_TTL}")
        if config.control_workers < 1:
            raise ProxySetupError("the control listener needs at least one worker")
        listed = {
            model
            for engine in self.world.engines.values()
            for provider, model in engine.listed_models()
            if provider in config.upstreams
        }
        if not listed:
            raise ProxySetupError("no customer's policy lists a model for the served providers")
        rates, pricing = load_rates(config.rate_table, listed)
        return rates, pricing, audit_db(self.world)

    def start(self) -> tuple[int, Path]:
        """Validate, end the runs of dead instances, bind both listeners and serve. Returns
        the model port and the control socket's path."""
        if self._server is not None or self._stopping.is_set():
            raise RuntimeError("the server has already been started")
        config = self._config
        rates, pricing, sink_db = self.validate()
        self.recovered = self._end_dead_instances()
        socket_path = self._make_socket_dir()
        try:
            self._control = ControlServer(socket_path, self, workers=config.control_workers)
            self.socket_path = socket_path
            _secure_socket(socket_path, self._socket_dir)
            hooks = ModelGatewayHooks(
                self.world.gate,
                self.tokens,
                ServedRuns(self),
                config.keys,
                rates,
                served_providers(config.upstreams),
            )
            self._server = proxy_server(
                config.host, config.port, hooks, sink_db, config.upstreams, pricing
            )
            self.model_port = self._server.start()
        except BaseException:
            self._close_listeners()
            self._remove_socket()
            raise
        self._control_thread = threading.Thread(
            target=self._control.serve_forever, name="model-proxy-control", daemon=True
        )
        self._control_thread.start()
        self._sweeper = threading.Thread(target=self._sweep, name="model-proxy-sweep", daemon=True)
        self._sweeper.start()
        return self.model_port, socket_path

    def serves(self, run_id: str) -> bool:
        """Whether this instance started the run and has not ended it."""
        with self._lock:
            return run_id in self._live

    def live_runs(self) -> int:
        with self._lock:
            return len(self._live)

    def start_run(
        self,
        customer_id: str,
        user_id: str,
        ttl: timedelta | None = None,
        caps: Caps | None = None,
    ) -> StartedRun:
        """Start a run for a user of a customer and issue its one token.

        The gate starts the run, so an unknown, suspended or other customer's user is
        refused (IdentityError) and the refusal is recorded. The caps are stored with the
        run and only tighten the policy's; the token's lifetime is at most the server's.
        """
        if self._stopping.is_set():
            raise ServerStopping("the server is stopping")
        config = self._config
        ttl = config.max_token_ttl if ttl is None else min(ttl, config.max_token_ttl)
        if ttl <= timedelta(0):
            raise ValueError("the token lifetime must be above zero")
        limits = RunLimits(
            None if caps is None else caps.max_usd, None if caps is None else caps.max_calls
        )

        def record(ctx: RunContext) -> None:
            self.world.store.add_run_limits(ctx.run_id, limits)
            self.world.store.add_served_run(
                ServedRun(ctx.run_id, self.instance_id, os.getpid(), ctx.started_at)
            )

        ctx = self.world.gate.start_run(customer_id, user_id, record=record)
        with self._lock:
            self._live[ctx.run_id] = ctx
            closed = self._closed
        if closed:
            self.end_run(ctx.run_id, RunEndReason.SHUTDOWN)
            raise ServerStopping("the server is stopping")
        user = self.world.store.get_user(user_id)
        issued = self.tokens.issue(
            ctx.run_id,
            ttl,
            record=lambda made: self.world.gate.record_token_event(
                OperatorAction.TOKEN_ISSUED, made, customer_id=ctx.customer_id, actor=user
            ),
        )
        spend = self.world.engines[ctx.customer_id].bundle.policy.spend
        effective = spend if caps is None else caps.tighten(spend)
        return StartedRun(ctx, issued.token, issued.record.expires_at, effective)

    def end_run(self, run_id: str, reason: RunEndReason = RunEndReason.REQUESTED) -> bool:
        """End a run this instance started: revoke its tokens, mark it ended and record why.
        Returns False for a run already ended; raises UnknownRun for one it did not start."""
        with self._lock:
            ctx = self._live.get(run_id)
        if ctx is None:
            served = self.world.store.get_served_run(run_id)
            if served is None or served.instance_id != self.instance_id:
                raise UnknownRun(run_id)
            return False
        self._end(ctx, reason)
        with self._lock:
            self._live.pop(run_id, None)
        return True

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_url": self.model_url, "runs_live": self.live_runs()}

    def stop(self) -> None:
        """Stop accepting on both listeners, drain in-flight control requests and model
        exchanges (each is settled), end every run this instance started, and remove the
        control socket and its directory."""
        self._stopping.set()
        try:
            self._close_listeners()
        finally:
            try:
                with self._lock:
                    self._closed = True
                    live = list(self._live)
                for run_id in live:
                    try:
                        self.end_run(run_id, RunEndReason.SHUTDOWN)
                    except Exception:
                        log.exception("could not end a run at shutdown")
            finally:
                self._remove_socket()
                if self._sweeper is not None:
                    self._sweeper.join(timeout=5)

    def _close_listeners(self) -> None:
        control, self._control = self._control, None
        server, self._server = self._server, None
        try:
            if control is not None:
                if self._control_thread is not None:
                    control.shutdown()
                    self._control_thread.join(timeout=5)
                control.server_close()
                control.drain(CONTROL_DRAIN_SECONDS)
        finally:
            if server is not None:
                server.stop()

    def _end(
        self, ctx: RunContext, reason: RunEndReason, *, instance_id: str | None = None
    ) -> None:
        gate = self.world.gate

        def revoked(token: TokenRecord) -> None:
            gate.record_token_event(
                OperatorAction.TOKEN_REVOKED, token, customer_id=ctx.customer_id, actor=None
            )

        def ended() -> None:
            gate.record_operator_action(
                OperatorAction.RUN_ENDED,
                customer_id=ctx.customer_id,
                actor=None,
                target=ctx.run_id,
                details={
                    "reason": reason.value,
                    "run_id": ctx.run_id,
                    "instance_id": instance_id or self.instance_id,
                },
            )

        self.tokens.end_run(ctx.run_id, record=revoked, ended=ended)

    def _end_dead_instances(self) -> int:
        """End every live run that a dead instance started. Returns how many."""
        ended = 0
        alive: dict[int, bool] = {}
        for served in self.tokens.live_served_runs():
            if served.instance_id == self.instance_id:
                continue
            if alive.setdefault(served.pid, _process_alive(served.pid)):
                continue
            ctx = self.tokens.run(served.run_id)
            if ctx is None:
                continue
            self._end(ctx, RunEndReason.RECOVERED, instance_id=served.instance_id)
            ended += 1
        return ended

    def _make_socket_dir(self) -> Path:
        """Create the control socket's directory, new and owner-only, and return the socket's
        path. An existing directory (or anything else at its path) is refused, not reused."""
        wanted = self._config.control_socket
        if wanted is None:
            self._socket_dir = Path(tempfile.mkdtemp(prefix="agentgate-control-"))
            return self._socket_dir / SOCKET_NAME
        directory = wanted.parent
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            raise ProxySetupError(
                f"{directory} already exists: the control socket's directory must be new"
            ) from None
        except OSError as exc:
            raise ProxySetupError(f"cannot create {directory}: {exc.strerror}") from None
        self._socket_dir = directory
        # mkdir's mode is narrowed by the umask; the directory must still be usable by us.
        os.chmod(directory, 0o700)
        return wanted

    def _remove_socket(self) -> None:
        directory, self._socket_dir = self._socket_dir, None
        socket_path, self.socket_path = self.socket_path, None
        if directory is None:
            return
        with contextlib.suppress(OSError):
            (socket_path or directory / SOCKET_NAME).unlink()
        with contextlib.suppress(OSError):
            directory.rmdir()

    def _sweep(self) -> None:
        """Reclaim reservations whose exchange never ended (a crash, an exception)."""
        interval = self._config.sweep_interval.total_seconds()
        while not self._stopping.wait(interval):
            try:
                self.world.gate.release_stale(self._config.lease, record=True)
            except Exception:
                log.exception("stale reservation sweep failed")


def _process_alive(pid: int) -> bool:
    """Whether a process with this id exists. A reused id reads as alive, which leaves a
    dead instance's runs to their tokens' lifetime rather than ending a live one's."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _secure_socket(path: Path, directory: Path | None) -> None:
    """Make the bound socket owner-only, then check that the directory is still the private
    one created for it and the socket is the one bound in it."""
    os.chmod(path, 0o600)
    uid = os.getuid()
    if directory is None:
        raise ProxySetupError("the control socket has no directory")
    held = os.lstat(directory)
    if not stat.S_ISDIR(held.st_mode) or held.st_uid != uid or stat.S_IMODE(held.st_mode) != 0o700:
        raise ProxySetupError(f"{directory} is not a private directory")
    bound = os.lstat(path)
    if not stat.S_ISSOCK(bound.st_mode) or bound.st_uid != uid:
        raise ProxySetupError(f"{path} is not the control socket")


# The control listener


class ControlServer(socketserver.UnixStreamServer):
    """The control socket: HTTP/1.1 requests, one per connection, each handled in a thread
    of its own, at most `workers` at once. Beyond that a request is answered 503 at once."""

    def __init__(self, path: Path, proxy: MultiRunProxy, *, workers: int) -> None:
        self.proxy = proxy
        self._slots = threading.BoundedSemaphore(workers)
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()
        try:
            super().__init__(str(path), _ControlHandler)
        except OSError as exc:
            raise ProxySetupError(f"cannot bind the control socket {path}: {exc}") from None

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            _answer_busy(request)
            self.shutdown_request(request)
            return
        worker = threading.Thread(
            target=self._work, args=(request, client_address), name="model-proxy-control-request"
        )
        worker.daemon = True
        with self._threads_lock:
            self._threads.add(worker)
        worker.start()

    def _work(self, request: Any, client_address: Any) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            log.exception("control request failed")
        finally:
            self.shutdown_request(request)
            self._slots.release()
            with self._threads_lock:
                self._threads.discard(threading.current_thread())

    def drain(self, timeout: float) -> None:
        """Wait for in-flight control requests to finish."""
        with self._threads_lock:
            pending = list(self._threads)
        for worker in pending:
            worker.join(timeout)


def _answer_busy(request: socket.socket) -> None:
    body = json.dumps({"error": "busy"}).encode()
    head = (
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    )
    with contextlib.suppress(OSError):
        request.settimeout(1.0)
        request.sendall(head.encode() + body)


_END_ROUTE = re.compile(r"/runs/(?P<run_id>[^/]+)/end")
_START_FIELDS = frozenset({"customer_id", "user_id", "ttl_seconds", "max_usd", "max_calls"})


class _BadRequest(Exception):
    def __init__(self, reason: str, status: int = 400) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


class _ControlHandler(BaseHTTPRequestHandler):
    """Parses and validates a control request and calls the proxy. Answers are JSON with a
    status and, for anything refused, a reason code: never a traceback, never a key."""

    server: ControlServer
    protocol_version = "HTTP/1.1"
    timeout = CONTROL_TIMEOUT_SECONDS
    server_version = "agentgate-control"
    sys_version = ""

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def do_PUT(self) -> None:
        self._route("PUT")

    def do_DELETE(self) -> None:
        self._route("DELETE")

    def do_PATCH(self) -> None:
        self._route("PATCH")

    def _route(self, method: str) -> None:
        self.close_connection = True
        path = self.path.partition("?")[0]
        try:
            if path == "/health":
                self._only(method, "GET")
                self._send(200, self.server.proxy.health())
            elif path == "/runs":
                self._only(method, "POST")
                self._start(self._body())
            elif (found := _END_ROUTE.fullmatch(path)) is not None:
                self._only(method, "POST")
                self._body()
                self._end(found["run_id"])
            else:
                raise _BadRequest("not_found", 404)
        except _BadRequest as bad:
            self._send(bad.status, {"error": bad.reason})
        except ServerStopping:
            self._send(503, {"error": "stopping"})
        except Exception:
            log.exception("control request failed")
            self._send(503, {"error": "unavailable"})

    def _only(self, method: str, allowed: str) -> None:
        if method != allowed:
            raise _BadRequest("method_not_allowed", 405)

    def _body(self) -> bytes:
        if self.headers.get("Transfer-Encoding") is not None:
            raise _BadRequest("length_required")
        raw = self.headers.get("Content-Length") or "0"
        if not raw.isdigit():
            raise _BadRequest("length_required")
        length = int(raw)
        if length > MAX_CONTROL_BODY:
            raise _BadRequest("body_too_large", 413)
        return self.rfile.read(length) if length else b""

    def _start(self, body: bytes) -> None:
        customer_id, user_id, ttl, caps = _start_request(body)
        try:
            started = self.server.proxy.start_run(customer_id, user_id, ttl, caps)
        except IdentityError as refused:
            self._send(403, {"error": refused.reason})
            return
        limits = started.limits
        self._send(
            201,
            {
                "run_id": started.run.run_id,
                "customer_id": started.run.customer_id,
                "user_id": started.run.user_id,
                "token": started.token,
                "expires_at": started.expires_at.isoformat(),
                "max_usd": plain_decimal(limits.per_run_usd),
                "max_calls": limits.max_calls_per_run,
                "per_customer_day_usd": plain_decimal(limits.per_customer_day_usd),
            },
        )

    def _end(self, run_id: str) -> None:
        try:
            self.server.proxy.end_run(run_id, RunEndReason.REQUESTED)
        except UnknownRun:
            raise _BadRequest("unknown_run", 404) from None
        self._send(200, {"run_id": run_id, "status": "ended"})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # The stdlib answers a malformed request line or an unsupported method with an HTML
        # page naming the server; answer with a reason code like every other refusal.
        self.close_connection = True
        # A request line too malformed to name a version is answered as HTTP/1.1, not as
        # HTTP/0.9 without a status line.
        if self.request_version == "HTTP/0.9":
            self.request_version = "HTTP/1.1"
        reason = "method_not_allowed" if code == 501 else "bad_request"
        with contextlib.suppress(OSError):
            self._send(405 if code == 501 else code, {"error": reason})

    def log_message(self, format: str, *args: Any) -> None:
        # Requests are on the audit chain; a Unix socket peer has no address to log.
        pass


def _start_request(body: bytes) -> tuple[str, str, timedelta | None, Caps | None]:
    """The fields of a `POST /runs` body, read strictly, or _BadRequest with a reason."""
    try:
        data = json.loads(body.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    except (UnicodeDecodeError, ValueError):
        raise _BadRequest("invalid_json") from None
    if not isinstance(data, dict):
        raise _BadRequest("invalid_json")
    if set(data) - _START_FIELDS:
        raise _BadRequest("unknown_field")
    if "customer_id" not in data or "user_id" not in data:
        raise _BadRequest("missing_field")
    customer_id, user_id = data["customer_id"], data["user_id"]
    for value in (customer_id, user_id):
        if not isinstance(value, str) or not 0 < len(value) <= _MAX_ID_LENGTH:
            raise _BadRequest("invalid_field")
    ttl = None
    if (seconds := data.get("ttl_seconds")) is not None:
        if not _positive_int(seconds):
            raise _BadRequest("invalid_field")
        ttl = timedelta(seconds=min(seconds, int(MAX_TTL.total_seconds())))
    max_usd = None
    if (amount := data.get("max_usd")) is not None:
        # A decimal string, as in the policy, so no float rounding reaches the ledger.
        if not isinstance(amount, str) or len(amount) > 32:
            raise _BadRequest("invalid_field")
        try:
            max_usd = Decimal(amount)
        except InvalidOperation:
            raise _BadRequest("invalid_field") from None
    max_calls = data.get("max_calls")
    if max_calls is not None and not _positive_int(max_calls):
        raise _BadRequest("invalid_field")
    try:
        caps = Caps(max_usd=max_usd, max_calls=max_calls)
    except (TypeError, ValueError):
        raise _BadRequest("invalid_field") from None
    return customer_id, user_id, ttl, caps


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def plain_decimal(value: Decimal) -> str:
    return "0" if value.is_zero() else format(value.normalize(), "f")
