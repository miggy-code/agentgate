"""The model proxy: Panoptes' reverse proxy with agentgate's decisions plugged into its hooks.

Panoptes observes and forwards. `ModelGatewayHooks.before_forward` decides: it admits a
request only if it is a POST to a generation endpoint, carries a live run token, and passes
`Gate.admit_model` (policy, worst-case reservation, caps). It then swaps the client's
credential for the provider's key. `after_exchange` ends the reservation by what happened to
the call (`Gate.finish_model`). `ModelProxy` is the process around it: it starts the run,
issues the one token, serves, sweeps stale reservations, and ends the run on shutdown.

A client sees a status and a reason code, never a policy detail or a key.
"""

import contextlib
import logging
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from panoptes.db import AuditDB
from panoptes.hooks import (
    Exchange,
    Identity,
    OutboundHeaders,
    ProxyHooks,
    Reject,
    Upstream,
)
from panoptes.hooks import User as ExchangeUser
from panoptes.pricing import PricingTable
from panoptes.proxy import ProxyServer
from panoptes.schema import ValidationError

from agentgate.audit.base import OperatorAction
from agentgate.audit.panoptes import PanoptesSink, storable_identifier
from agentgate.context import RunContext
from agentgate.gate import CapReached, Denied, Gate, IdentityError
from agentgate.model.admission import Caps, ModelExchange, parse_model_call
from agentgate.model.keys import ProviderKeys, upstream_headers
from agentgate.model.pricing import RATE_FIELDS, RateTable
from agentgate.model.tokens import (
    DEFAULT_TTL,
    TokenRecord,
    TokenRejected,
    TokenStore,
    presented_token,
    write_token_file,
)
from agentgate.models import User
from agentgate.policy.schema import ModelProvider
from agentgate.spend import Reservation
from agentgate.world import DEFAULT_RESERVATION_LEASE, World

log = logging.getLogger(__name__)

# The agent type a proxied exchange is recorded under. Not "agentgate": that is the type of
# the gate's own decision events, which `read_events` reads.
CLIENT_AGENT_TYPE = "model_client"
ASSURANCE = "agentgate_run_token"

# The wire protocol each policy provider speaks. A proxy serves one upstream per wire
# protocol, so one policy provider per wire protocol.
WIRE_OF: Mapping[ModelProvider, str] = {
    ModelProvider.ANTHROPIC: "anthropic",
    ModelProvider.OPENAI: "openai",
    ModelProvider.OPENAI_COMPATIBLE: "openai",
    ModelProvider.GEMINI: "gemini",
}

# The one path each generation endpoint is served at, matched on the path as received. A
# Gemini model is read from this raw path and never percent-decoded, so `%2e` and friends
# cannot name a model the policy lists.
GENERATION_ROUTES: Mapping[str, re.Pattern[str]] = {
    "openai": re.compile(r"/v1/(?:chat/completions|responses)"),
    "anthropic": re.compile(r"/v1/messages"),
    "gemini": re.compile(
        r"/v1(?:beta|alpha)?/models/(?P<model>[^/:?#]+):(?:generateContent|streamGenerateContent)"
    ),
}

# Reason codes the client sees.
NOT_GENERATION = "route_not_allowed"
UNKNOWN_PROVIDER = "provider_not_served"
TOKEN_REFUSED = "token_refused"
IDENTITY_REFUSED = "identity_refused"
UNAVAILABLE = "unavailable"

SWEEP_INTERVAL = timedelta(seconds=60)


class ProxySetupError(ValueError):
    """The proxy cannot be started as configured. Messages never hold a key."""


@dataclass(frozen=True)
class Route:
    wire: str
    path_model: str | None


def generation_route(method: str, target: str) -> Route | None:
    """The generation endpoint a request is for, or None for anything else: another method,
    a listing, a token count, a fetch of an earlier response. `target` is the path as
    received, with its query."""
    if method != "POST":
        return None
    path = target.partition("?")[0]
    for wire, pattern in GENERATION_ROUTES.items():
        found = pattern.fullmatch(path)
        if found is not None:
            groups = found.groupdict()
            return Route(wire, groups.get("model"))
    return None


@dataclass
class _Admitted:
    """What `before_forward` hands `after_exchange` through `exchange.state`."""

    ctx: RunContext
    reservation: Reservation


class ModelGatewayHooks(ProxyHooks):
    def __init__(
        self,
        gate: Gate,
        tokens: TokenStore,
        run: RunContext,
        keys: ProviderKeys,
        rates: RateTable,
        providers: Mapping[str, ModelProvider],
        caps: Caps | None = None,
    ) -> None:
        self._gate = gate
        self._tokens = tokens
        self._run = run
        self._keys = keys
        self._rates = rates
        # The policy provider each wire protocol is served as.
        self._providers = dict(providers)
        self._caps = caps

    def before_forward(self, exchange: Exchange) -> Reject | None:
        ex = exchange
        # 1. Only a POST to a generation endpoint.
        route = generation_route(ex.method, ex.path)
        if route is None:
            return _reject(403, NOT_GENERATION)
        # 2. The provider: Panoptes says which wire protocol; the operator's upstreams say
        # which policy provider speaks it.
        provider = self._providers.get(ex.provider)
        if ex.provider != route.wire or provider is None:
            return _reject(403, UNKNOWN_PROVIDER)
        # 3. The raw body, with agentgate's strict parser (never `request_json`).
        call = parse_model_call(
            provider,
            ex.request_body,
            path_model=route.path_model,
            correlation_id=ex.correlation_id,
        )
        call = replace(
            call,
            exchange_id=ex.exchange_id,
            correlation_source=ex.correlation_source,
        )
        # 4. The token, then the run and its user.
        try:
            token = presented_token(ex.request_headers.items(), ex.query_credentials)
            ctx = self._tokens.resolve(token, run_id=self._run.run_id)
        except TokenRejected as refused:
            return self._token_refused(refused, ex)
        # 5. Everything that can raise happens before the reservation exists.
        try:
            identity = _identity(ctx)
            headers = upstream_headers(self._keys, provider, ex.outbound_headers.as_dict().items())
        except (ValidationError, ValueError, TypeError, KeyError):
            return _reject(403, IDENTITY_REFUSED)
        ex.identity = identity
        # 6. The decision, and the reservation it makes.
        try:
            admitted = self._gate.admit_model(ctx, call, rates=self._rates, caps=self._caps)
        except Exception:
            log.exception("admit_model failed; the request is not forwarded")
            return _reject(503, UNAVAILABLE)
        if isinstance(admitted, Denied):
            return _reject(403, admitted.reason)
        if isinstance(admitted, CapReached):
            return _reject(429, admitted.cap.value)
        ex.state["agentgate"] = _Admitted(ctx, admitted)
        # 7. The provider's credential replaces every one the client sent. A failure here is
        # a hook error: Panoptes answers 502 and does not forward, and `after_exchange`,
        # owed for every entered exchange, finds request_sent False and releases the
        # reservation.
        ex.drop_query_credentials()
        ex.outbound_headers = OutboundHeaders(dict(headers))
        return None

    def after_exchange(self, exchange: Exchange) -> None:
        self._finish(exchange)

    def _finish(self, exchange: Exchange) -> None:
        """End the exchange's reservation, once. A rejected exchange has none."""
        admitted = exchange.state.pop("agentgate", None)
        if not isinstance(admitted, _Admitted):
            return
        status = exchange.upstream_status
        micros = exchange.estimated_usd_micros
        ended = ModelExchange(
            request_sent=bool(exchange.request_sent),
            provider_error=status is not None and status >= 400,
            complete=bool(exchange.complete),
            produced_output=bool(exchange.produced_output),
            cost_usd=None if micros is None else Decimal(micros).scaleb(-6),
            exchange_id=exchange.exchange_id,
        )
        try:
            self._gate.finish_model(admitted.ctx, admitted.reservation, ended)
        except Exception:
            # The reservation stays open for the stale sweep, which reclaims it.
            log.exception("could not end a model call's reservation")

    def _token_refused(self, refused: TokenRejected, exchange: Exchange) -> Reject:
        # A token that names a run is recorded on that run; a missing or unknown one
        # names none, and is on Panoptes' record of the rejected exchange.
        named = None if refused.run_id is None else self._tokens.run(refused.run_id)
        if named is not None:
            try:
                self._gate.refuse_token(
                    named, refused.reason.value, exchange_id=exchange.exchange_id
                )
            except Exception:
                log.exception("could not record a refused token")
                return _reject(503, UNAVAILABLE)
        return _reject(401, TOKEN_REFUSED)


def _reject(status: int, reason: str) -> Reject:
    return Reject(status, {"error": reason}, reason=reason)


def _identity(ctx: RunContext) -> Identity:
    """The identity an exchange is recorded under: the run the token belongs to."""
    for value in (ctx.run_id, ctx.user_id):
        if not storable_identifier(value):
            raise ValueError("run or user id cannot be stored")
    return Identity(
        agent_type=CLIENT_AGENT_TYPE,
        session_id=ctx.run_id,
        run_id=ctx.run_id,
        user=ExchangeUser(id=ctx.user_id),
        assurance=ASSURANCE,
    )


def check_rates(rates: RateTable, pricing: PricingTable, models: set[str]) -> None:
    """Fail unless the reservation (RateTable) and the settlement (Panoptes' PricingTable)
    come from one table: the same version and source, and the same rates for every model
    the policy lists."""
    if (rates.version, rates.source) != (pricing.version, pricing.source):
        raise ProxySetupError("the two rate tables differ in version or source")
    for name in sorted(models):
        reserve, settle = rates.models.get(name), pricing.models.get(name)
        if reserve is None and settle is None:
            continue  # unpriced: reserved at the policy's fixed amount, settled in full
        if reserve is None or settle is None:
            raise ProxySetupError(f"model {name!r} is priced by only one of the two tables")
        for field in RATE_FIELDS:
            if getattr(reserve, field) != settle[field]:
                raise ProxySetupError(f"model {name!r} has different {field} in the two tables")


@dataclass(frozen=True)
class ProxyConfig:
    # Origin of the upstream serving each policy provider (at most one per wire protocol).
    upstreams: Mapping[ModelProvider, str]
    keys: ProviderKeys
    rate_table: Path
    host: str = "127.0.0.1"
    port: int = 0
    caps: Caps | None = None
    token_ttl: timedelta = DEFAULT_TTL
    token_file: Path | None = None
    lease: timedelta = DEFAULT_RESERVATION_LEASE
    sweep_interval: timedelta = SWEEP_INTERVAL


class ModelProxy:
    """One run's model proxy over a world. `start()` returns the port; `stop()` drains."""

    def __init__(self, world: World, user_id: str, config: ProxyConfig) -> None:
        self._world = world
        self._user_id = user_id
        self._config = config
        self._server: ProxyServer | None = None
        self._stopping = threading.Event()
        self._sweeper: threading.Thread | None = None
        self._tokens = TokenStore(world.store, world.clock)
        self._run: RunContext | None = None
        self._token_id: str | None = None
        # The run's token when it is not written to a file; the process that started the
        # proxy hands it to its client.
        self.token: str | None = None
        self.port = 0

    @property
    def run(self) -> RunContext:
        if self._run is None:
            raise RuntimeError("the proxy has not started")
        return self._run

    def validate(self) -> tuple[RateTable, PricingTable, AuditDB, User]:
        """Everything that must hold before the proxy serves, or ProxySetupError: one
        upstream per wire protocol, a key for each, a user and a policy with a model for
        them, one rate table read both ways, and a Panoptes audit log to share."""
        config, world = self._config, self._world
        wires = [WIRE_OF[provider] for provider in config.upstreams]
        if not config.upstreams or len(wires) != len(set(wires)):
            raise ProxySetupError("serve one upstream per wire protocol, and at least one")
        for provider in config.upstreams:
            if not config.keys.has(provider):
                raise ProxySetupError(f"no provider key for {provider.value}")
        user = world.store.get_user(self._user_id)
        if user is None:
            raise ProxySetupError("unknown user")
        engine = world.engines.get(user.customer_id)
        if engine is None:
            raise ProxySetupError("the user's customer has no policy")
        listed = {
            model for provider, model in engine.listed_models() if provider in config.upstreams
        }
        if not listed:
            raise ProxySetupError("the policy lists no model for the served providers")
        try:
            rates = RateTable.load(config.rate_table)
            pricing = PricingTable.load(str(config.rate_table))
            check_rates(rates, pricing, listed)
        except ValueError as exc:
            raise ProxySetupError(str(exc)) from None
        sink_db = world.sink.db if isinstance(world.sink, PanoptesSink) else None
        if sink_db is None:
            raise ProxySetupError("the model proxy needs a Panoptes audit log")
        return rates, pricing, sink_db, user

    def start(self, run: RunContext | None = None) -> int:
        """Validate, start (or join) the run, issue its token and serve."""
        config, world = self._config, self._world
        if self._server is not None:
            raise RuntimeError("the proxy is already started")
        rates, pricing, sink_db, user = self.validate()

        try:
            self._run = run if run is not None else world.gate.start_run(user.customer_id, user.id)
        except IdentityError as err:
            raise ProxySetupError(f"{err.reason}: {err}") from None
        ctx = self._run
        issued = self._tokens.issue(
            ctx.run_id,
            config.token_ttl,
            record=lambda made: self._record(OperatorAction.TOKEN_ISSUED, made, actor=user),
        )
        self._token_id = issued.record.token_id
        try:
            if config.token_file is not None:
                write_token_file(config.token_file, issued.token)
            hooks = ModelGatewayHooks(
                world.gate,
                self._tokens,
                ctx,
                config.keys,
                rates,
                {WIRE_OF[provider]: provider for provider in config.upstreams},
                config.caps,
            )
            self._server = ProxyServer(
                host=config.host,
                port=config.port,
                hooks=hooks,
                db=sink_db,
                chain=True,
                upstreams={
                    WIRE_OF[provider]: Upstream(url, name=provider.value)
                    for provider, url in config.upstreams.items()
                },
                pricing=pricing,
            )
            self.port = self._server.start()
        except BaseException:
            self._end_run()
            raise
        self.token = issued.token if config.token_file is None else None
        self._sweeper = threading.Thread(target=self._sweep, name="model-proxy-sweep", daemon=True)
        self._sweeper.start()
        return self.port

    def stop(self) -> None:
        """Stop accepting, finish in-flight exchanges (each is settled), then end the run:
        its token is revoked and recorded."""
        self._stopping.set()
        try:
            if self._server is not None:
                self._server.stop()
        finally:
            if self._sweeper is not None:
                self._sweeper.join(timeout=5)
            self._end_run()

    def _end_run(self) -> None:
        if self._run is None:
            return
        self._tokens.end_run(
            self._run.run_id,
            record=lambda gone: self._record(OperatorAction.TOKEN_REVOKED, gone, actor=None),
        )
        token_file = self._config.token_file
        if token_file is not None:
            with contextlib.suppress(OSError):
                token_file.unlink()

    def _record(self, action: OperatorAction, token: TokenRecord, *, actor: object) -> None:
        run = self.run
        user = self._world.store.get_user(self._user_id) if actor is not None else None
        self._world.gate.record_token_event(action, token, customer_id=run.customer_id, actor=user)

    def _sweep(self) -> None:
        """Reclaim reservations whose exchange never ended (a crash, an exception)."""
        interval = self._config.sweep_interval.total_seconds()
        while not self._stopping.wait(interval):
            try:
                self._world.gate.release_stale(self._config.lease, record=True)
            except Exception:
                log.exception("stale reservation sweep failed")
