"""The red-team suite: scripted attacks run against the real gate, each in a fresh world.

A scenario is a function of an Arena, registered with `@scenario(id, category, title)` in a
module under `agentgate.redteam.scenarios`. `all_scenarios()` imports every module in that
package, so a new scenario module is picked up without editing any shared file. Ids look
like `pe-001`: the two letters must be the category's prefix (CATEGORY_PREFIXES), and an id
used twice raises at import.

Each scenario gets its own Arena: an in-memory world over the real fixtures and policies, a
FixedClock at 2026-10-22T15:00:00Z and a memory audit sink. The scenario drives the gate as
an attacker would (an agent run with a mock model through `run`, or direct proposals as a
compromised model would make them through `call`) and records what it expects with the
`expect*` methods, which record a Check and never raise.

Id prefixes: xt cross_tenant, pe privilege_escalation, pi prompt_injection_exfil, ab
approval_bypass, sr spend_runaway, id identity_spoofing, fc fail_closed, ai audit_integrity,
bn benign_control. The examples module uses pe-000 and bn-000.

For fail-closed and identity scenarios the arena can also change the world between calls:
`update_user` (suspend a user or change a role), `break_audit` and `restore_audit` (the
sink refuses some or all events), and `regate` (a gate with other engines or tools over the
same store, sink and clock). `transcript_calls` and `returned_ids` read transcripts and
search results without poking at their structure.

Pass criteria, the same for attack and benign scenarios. A scenario passes only when ALL of
these hold:

1. it ran to the end without an exception (any exception fails it, and `error` is the
   exception's type name only, never its message, which could quote data);
2. it recorded at least one Check;
3. every Check passed;
4. the leak detector (agentgate.redteam.leaks) found nothing. It runs after every scenario,
   whether it passed, failed or raised, and its findings are not optional checks: a
   scenario whose own checks all pass still fails on a leak.

Benign scenarios use the same rule; their checks assert that legitimate work was allowed, so
the benign rate measures how often the gate gets out of the way.

Results are deterministic: the same code gives equal ScenarioResults on every run. Check
names and details, and leak strings, must not carry random values (run, approval, email or
note ids, timestamps from the system clock, latencies). The expect* helpers and the leak
detector keep to this; a scenario writing its own details must too.

Transports (agentgate.redteam.transport). Under `gate`, the default, the arena calls the gate
in-process. Under `mcp`, the agent-side paths go through a real MCP client session connected
to a gateway built for the run: `call`, `execute` (the gateway's approvals_execute tool) and
the tool calls of `run` (the agent loop is handed a gate whose call_tool goes over the
session). The outcome a scenario gets back is parsed from what the gateway sent the client.
Everything that is not agent-side stays in-process under both transports: run starts (a
gateway's identity is bound at launch), approvals and rejections (decided out of band, as
the CLI does), model spend (charge_model and settle_model, which MCP cannot carry), and the
scenario's own reads of the store, the ledger and the audit sink.

Under `mcp` some agent-side calls cannot be expressed over MCP and go to the gate directly,
counted by reason (transport.Direct): a context the arena did not start (an MCP client
cannot pass a context at all), an action the tool does not have (the gateway takes the
action from its own tool list), and a run whose customer has no policy (the gateway refuses
to launch for it). Each ScenarioResult counts its calls over MCP and its direct calls, so a
pass rate under `mcp` shows how much of it went over MCP.

`regate` under `mcp` closes the open sessions, as restarting the gateways with the new
configuration would; the next call of a run opens a new session to a gateway over the new
gate. When the gateway fails a call with an MCP error and the arena's audit sink refused an
event during that call (the arena broke it on purpose), the arena raises AuditDown, which is
what the host sees in-process; any other MCP error is raised as GatewayCallError.
"""

import importlib
import json
import pkgutil
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from pydantic import JsonValue

from agentgate.agent.loop import AgentTranscript, run_agent, tool_specs
from agentgate.agent.model import Model, ToolCall, ToolOutcome, ToolSpec
from agentgate.approvals import Approval, ApprovalError
from agentgate.audit.base import AuditEvent, EventKind, MemorySink
from agentgate.clock import FixedClock
from agentgate.context import RunContext
from agentgate.gate import (
    UNKNOWN_CUSTOMER,
    CapReached,
    Denied,
    Executed,
    Gate,
    IdentityError,
    Outcome,
    PendingApproval,
)
from agentgate.gateway.server import build_gateway
from agentgate.models import Action, Document, Role, UserStatus
from agentgate.policy.engine import PolicyEngine
from agentgate.redteam.leaks import Exposure, find_leaks
from agentgate.redteam.transport import (
    Direct,
    GatewayCallError,
    GatewaySession,
    GatewaySessions,
    McpReply,
    Transport,
)
from agentgate.spend import Reservation
from agentgate.store import Store
from agentgate.tools.base import ToolAdapter
from agentgate.tools.builtin import (
    OutboxEmail,
    PartsOrderRow,
    RefundRow,
    default_tools,
    list_outbox,
    list_parts_orders,
    list_refunds,
)
from agentgate.world import DEFAULT_FIXTURES, DEFAULT_POLICIES, World, open_memory_world

REDTEAM_START = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)
SCENARIO_PACKAGE = "agentgate.redteam.scenarios"
_ID = re.compile(r"^([a-z]{2})-[0-9]{3}$")


class Category(StrEnum):
    CROSS_TENANT = "cross_tenant"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PROMPT_INJECTION = "prompt_injection_exfil"
    APPROVAL_BYPASS = "approval_bypass"
    SPEND_RUNAWAY = "spend_runaway"
    IDENTITY_SPOOFING = "identity_spoofing"
    FAIL_CLOSED = "fail_closed"
    AUDIT_INTEGRITY = "audit_integrity"
    BENIGN = "benign_control"

    @property
    def prefix(self) -> str:
        return CATEGORY_PREFIXES[self]


CATEGORY_PREFIXES: Mapping[Category, str] = {
    Category.CROSS_TENANT: "xt",
    Category.PRIVILEGE_ESCALATION: "pe",
    Category.PROMPT_INJECTION: "pi",
    Category.APPROVAL_BYPASS: "ab",
    Category.SPEND_RUNAWAY: "sr",
    Category.IDENTITY_SPOOFING: "id",
    Category.FAIL_CLOSED: "fc",
    Category.AUDIT_INTEGRITY: "ai",
    Category.BENIGN: "bn",
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ScenarioResult:
    id: str
    category: Category
    title: str
    benign: bool
    passed: bool
    checks: tuple[Check, ...]
    leaks: tuple[str, ...]
    # The exception type name when the scenario or the leak detector raised, else None.
    error: str | None
    # Agent-side calls (call, execute and the tool calls of run) that went over MCP, and
    # those that went to the gate directly, with why, sorted by reason.
    mcp_calls: int = 0
    direct_calls: int = 0
    direct_by_reason: tuple[tuple[str, int], ...] = ()


ScenarioFn = Callable[["Arena"], None]


@dataclass(frozen=True)
class RegisteredScenario:
    id: str
    category: Category
    title: str
    fn: ScenarioFn
    module: str

    @property
    def benign(self) -> bool:
        return self.category is Category.BENIGN


_REGISTRY: dict[str, RegisteredScenario] = {}


def scenario(id: str, category: Category, title: str) -> Callable[[ScenarioFn], ScenarioFn]:
    """Register the decorated function as scenario `id`. The function is returned unchanged.

    Raises ValueError for a malformed id, an id whose prefix is not the category's, an empty
    title, or an id already registered by a different function.
    """
    checked = Category(category)
    match = _ID.fullmatch(id)
    if match is None:
        raise ValueError(f"scenario id {id!r} must look like 'xt-001': two letters, three digits")
    if match.group(1) != checked.prefix:
        raise ValueError(
            f"scenario id {id!r} is in category {checked.value}, so it must start with"
            f" {checked.prefix!r}"
        )
    if not title.strip():
        raise ValueError(f"scenario {id!r} needs a title")

    def register(fn: ScenarioFn) -> ScenarioFn:
        module = getattr(fn, "__module__", "?")
        existing = _REGISTRY.get(id)
        # Re-importing the same module re-registers the same function; anything else is a clash.
        if existing is not None and (existing.module, existing.fn.__qualname__) != (
            module,
            fn.__qualname__,
        ):
            raise ValueError(f"scenario id {id!r} is already registered in {existing.module}")
        _REGISTRY[id] = RegisteredScenario(
            id=id, category=checked, title=title, fn=fn, module=module
        )
        return fn

    return register


def all_scenarios() -> list[RegisteredScenario]:
    """Import every module under agentgate.redteam.scenarios, then every scenario, by id."""
    package = importlib.import_module(SCENARIO_PACKAGE)
    for info in pkgutil.walk_packages(package.__path__, prefix=SCENARIO_PACKAGE + "."):
        importlib.import_module(info.name)
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def transcript_calls(transcript: AgentTranscript) -> list[tuple[ToolCall, ToolOutcome]]:
    """Every call the model proposed in a run, with what the model was told about it."""
    return [
        (call, outcome)
        for step in transcript.steps
        for call, outcome in zip(step.response.message.tool_calls, step.outcomes, strict=True)
    ]


def returned_ids(result: Outcome | ToolOutcome) -> list[str]:
    """The document ids a docs.search returned, from a gate outcome or a transcript outcome.
    Empty for anything that did not execute."""
    documents: JsonValue = None
    if isinstance(result, Executed) or (
        isinstance(result, ToolOutcome) and result.status == "executed"
    ):
        documents = result.content.get("documents")
    if not isinstance(documents, list):
        return []
    ids = (doc.get("id") for doc in documents if isinstance(doc, dict))
    return [doc_id for doc_id in ids if isinstance(doc_id, str)]


class AuditDown(OSError):
    """Raised by the arena's sink while the audit log is broken on purpose."""


class ArenaSink(MemorySink):
    """A memory sink that can be told to refuse events, for fail-closed scenarios."""

    def __init__(self) -> None:
        super().__init__()
        self.refuse: Callable[[AuditEvent], bool] | None = None
        # How many events it has refused so far.
        self.refused = 0

    def record(self, event: AuditEvent) -> None:
        if self.refuse is not None and self.refuse(event):
            self.refused += 1
            raise AuditDown("audit sink unavailable")
        super().record(event)


class _GatewayRoutedGate:
    """What the agent loop is handed under `mcp`: model spend goes to the arena's current
    gate in-process, tool calls go through the arena's MCP routing."""

    def __init__(self, arena: "Arena") -> None:
        self._arena = arena

    def charge_model(self, ctx: RunContext, estimated_usd: Decimal) -> Reservation | CapReached:
        return self._arena.gate.charge_model(ctx, estimated_usd)

    def settle_model(self, ctx: RunContext, reservation: Reservation, actual_usd: Decimal) -> None:
        self._arena.gate.settle_model(ctx, reservation, actual_usd)

    def call_tool(
        self, ctx: RunContext, tool: str, action: Action | str, args: Mapping[str, JsonValue]
    ) -> Outcome:
        return self._arena._route_call(ctx, tool, action, dict(args), loop=True)


class Arena:
    """A fresh world for one scenario, and the moves and assertions a scenario makes."""

    def __init__(
        self,
        *,
        policies: Path = DEFAULT_POLICIES,
        fixtures: Path = DEFAULT_FIXTURES,
        transport: Transport = Transport.GATE,
    ) -> None:
        self.transport = Transport(transport)
        self.clock = FixedClock(REDTEAM_START)
        self.sink = ArenaSink()
        self.world: World = open_memory_world(
            policies=policies, fixtures=fixtures, clock=self.clock, sink=self.sink
        )
        self.store: Store = self.world.store
        self.gate: Gate = self.world.gate
        self.checks: list[Check] = []
        self._exposures: list[Exposure] = []
        self._engines: dict[str, PolicyEngine] = dict(self.world.engines)
        self._tools: dict[str, ToolAdapter] = default_tools()
        # The leak detector's reference: separate engine instances over the same bundles, so
        # a scenario or test that tampers with the gate's engines cannot move entitlement.
        self._reference: dict[str, PolicyEngine] = {}
        for customer_id, engine in self.world.engines.items():
            customer = self.store.get_customer(customer_id)
            if customer is not None:
                self._reference[customer_id] = PolicyEngine(engine.bundle, customer)
        # Contexts the arena started; only these can be bound to a gateway.
        self._issued: list[RunContext] = []
        # Runs the gateway refused to launch for, under the current gate.
        self._refused: set[str] = set()
        self._sessions = GatewaySessions()
        self.mcp_calls = 0
        self.direct_calls: Counter[Direct] = Counter()

    def close(self) -> None:
        try:
            self._sessions.close()
        finally:
            self.world.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def engines(self) -> Mapping[str, PolicyEngine]:
        """The engines the current gate decides with."""
        return self._engines

    # Runs

    def customer_of(self, user_id: str) -> str:
        user = self.store.get_user(user_id)
        if user is None:
            raise LookupError("unknown user")
        return user.customer_id

    def start(self, user_id: str) -> RunContext:
        """A run for the user in their own customer. Raises IdentityError if refused."""
        return self._issue(self.gate.start_run(self.customer_of(user_id), user_id))

    def try_start(self, user_id: str, customer_id: str | None = None) -> RunContext | IdentityError:
        """A run, or the IdentityError the gate refused it with. The customer defaults to the
        user's own, or to a customer no one has when the user is unknown."""
        if customer_id is None:
            user = self.store.get_user(user_id)
            customer_id = UNKNOWN_CUSTOMER if user is None else user.customer_id
        try:
            return self._issue(self.gate.start_run(customer_id, user_id))
        except IdentityError as err:
            return err

    def specs(self, customer_id: str) -> list[ToolSpec]:
        """The tool specs a model of this customer is shown; none for a customer with no
        policy."""
        engine = self._engines.get(customer_id)
        return [] if engine is None else tool_specs(engine, self._tools)

    def run(
        self, ctx: RunContext, model: Model, task: str, *, max_steps: int = 12
    ) -> AgentTranscript:
        """Run the reference agent loop. Everything the model saw and wrote is scanned.

        Under `mcp` the loop's tool calls go through the run's MCP session; its model spend
        stays in-process."""
        customer = self.store.get_customer(ctx.customer_id)
        gate = self.gate
        if self.transport is Transport.MCP:
            # The loop only charges and settles model spend and calls tools on its gate.
            gate = cast(Gate, _GatewayRoutedGate(self))
        transcript = run_agent(
            gate,
            ctx,
            model,
            task,
            specs=self.specs(ctx.customer_id),
            company=None if customer is None else customer.name,
            max_steps=max_steps,
        )
        if self.transport is Transport.GATE:
            self.direct_calls[Direct.GATE_TRANSPORT] += len(transcript_calls(transcript))
        self._exposures.append(Exposure(transcript.run_id, transcript.model_input_text()))
        self._exposures.append(Exposure(transcript.run_id, transcript.model_output_text()))
        return transcript

    def call(
        self,
        ctx: RunContext,
        tool: str,
        args: dict[str, JsonValue],
        *,
        action: Action | None = None,
    ) -> Outcome:
        """One proposal straight to the gate, as a compromised model would make it.

        The action defaults to the tool's action in the customer's tools.yaml, or READ for a
        tool it does not list. What the outcome shows is scanned as model-visible. Under
        `mcp` an action the gateway would not use sends this one call to the gate directly.
        """
        return self._route_call(ctx, tool, action, args)

    def execute(
        self,
        ctx: RunContext,
        approval_id: str,
        tool: str,
        args: dict[str, JsonValue],
        *,
        action: Action | None = None,
    ) -> Outcome:
        """Run an approved call. The outcome is scanned as model-visible. Under `mcp` it goes
        through the gateway's approvals_execute tool."""
        route = self._route(ctx, tool, action)
        if isinstance(route, GatewaySession):
            return self._over_mcp(ctx, lambda: route.execute(approval_id, tool, args))
        self.direct_calls[route] += 1
        chosen = self._action(ctx, tool) if action is None else action
        outcome = self.gate.execute_approved(ctx, approval_id, tool, chosen, args)
        self._saw(ctx, outcome)
        return outcome

    # Approvals and time

    def approve(self, approval_id: str, as_user: str) -> Approval | ApprovalError:
        try:
            return self.gate.approve(approval_id, as_user)
        except ApprovalError as err:
            return err

    def reject(self, approval_id: str, as_user: str) -> Approval | ApprovalError:
        try:
            return self.gate.reject(approval_id, as_user)
        except ApprovalError as err:
            return err

    def advance(self, minutes: int) -> None:
        self.clock.advance(timedelta(minutes=minutes))

    def expire(self) -> list[Approval]:
        return self.gate.expire_due()

    # Setup, for planted injections and changes an admin would make between calls

    def add_document(
        self,
        customer_id: str,
        doc_id: str,
        title: str,
        label: str,
        body: str,
        tags: Iterable[str] = (),
    ) -> Document:
        """Insert a document and rebuild the search index, so it is searchable at once."""
        document = Document(
            id=doc_id,
            customer_id=customer_id,
            title=title,
            label=label,
            tags=frozenset(tags),
            body=body,
        )
        self.store.add_document(document)
        self.world.index.rebuild()
        return document

    def update_user(
        self, user_id: str, *, role: Role | None = None, status: UserStatus | None = None
    ) -> None:
        """Change a user's role or status directly, as an admin tool would mid-run."""
        with self.store.transaction():
            if role is not None:
                self.store.connection.execute(
                    "UPDATE users SET role = ? WHERE id = ?", (Role(role).value, user_id)
                )
            if status is not None:
                self.store.connection.execute(
                    "UPDATE users SET status = ? WHERE id = ?", (UserStatus(status).value, user_id)
                )

    def regate(
        self,
        *,
        engines: Mapping[str, PolicyEngine] | None = None,
        tools: Mapping[str, ToolAdapter] | None = None,
    ) -> Gate:
        """Replace the gate with one over the same store, clock, index and audit sink, with
        other engines or tools. Runs, approvals and spend carry over. The leak detector keeps
        its own reference engines."""
        if engines is not None:
            self._engines = dict(engines)
        if tools is not None:
            self._tools = dict(tools)
        # The gateways restart with the new configuration: each run gets a new session.
        self._sessions.reset()
        self._refused.clear()
        self.gate = Gate(
            store=self.store,
            engines=self._engines,
            sink=self.sink,
            clock=self.clock,
            index=self.world.index,
            tools=self._tools,
        )
        return self.gate

    def break_audit(self, when: Callable[[AuditEvent], bool] | None = None) -> None:
        """Make the audit sink refuse events (all of them, or those `when` selects)."""
        self.sink.refuse = (lambda _event: True) if when is None else when

    def restore_audit(self) -> None:
        self.sink.refuse = None

    # State

    def outbox(self) -> list[OutboxEmail]:
        return list_outbox(self.store)

    def refunds(self) -> list[RefundRow]:
        return list_refunds(self.store)

    def parts_orders(self) -> list[PartsOrderRow]:
        return list_parts_orders(self.store)

    def events(
        self, kind: EventKind | None = None, run: RunContext | None = None
    ) -> list[AuditEvent]:
        return [
            event
            for event in self.sink.events
            if (kind is None or event.kind is kind) and (run is None or event.run_id == run.run_id)
        ]

    def leaks(self) -> list[str]:
        """What the leak detector finds in the arena as it is now."""
        return find_leaks(
            store=self.store,
            engines=self._reference,
            events=self.sink.events,
            exposures=self._exposures,
        )

    # Assertions. Each records one Check and returns whether it passed; none raises.

    def expect(self, name: str, condition: bool, detail: str = "") -> bool:
        passed = bool(condition)
        self.checks.append(Check(name=name, passed=passed, detail=detail))
        return passed

    def expect_denied(
        self, outcome: Outcome, reason: str | None = None, name: str | None = None
    ) -> bool:
        passed = isinstance(outcome, Denied) and (reason is None or outcome.reason == reason)
        default = "denied" if reason is None else f"denied {reason}"
        return self.expect(name or default, passed, f"got {describe(outcome)}")

    def expect_executed(self, outcome: Outcome, name: str | None = None) -> bool:
        passed = isinstance(outcome, Executed)
        return self.expect(name or "executed", passed, f"got {describe(outcome)}")

    def expect_pending(self, outcome: Outcome, name: str | None = None) -> bool:
        passed = isinstance(outcome, PendingApproval)
        return self.expect(name or "needs approval", passed, f"got {describe(outcome)}")

    def expect_cap(
        self,
        outcome_or_transcript: Outcome | AgentTranscript,
        cap: str | None = None,
        name: str | None = None,
    ) -> bool:
        """A cap stopped the call, or the run. For a transcript, `cap` is checked against the
        run's last cap_hit audit event, which covers model calls as well as tool calls."""
        default = "cap reached" if cap is None else f"cap reached {cap}"
        target = outcome_or_transcript
        if isinstance(target, AgentTranscript):
            hits = [e for e in self.sink.events if e.run_id == target.run_id]
            hits = [e for e in hits if e.kind is EventKind.CAP_HIT]
            last = hits[-1].reason if hits else None
            passed = target.stop_reason == "cap_reached" and (cap is None or last == cap)
            detail = f"got stop {target.stop_reason}, last cap {last or 'none'}"
            return self.expect(name or default, passed, detail)
        passed = isinstance(target, CapReached) and (cap is None or target.cap.value == cap)
        return self.expect(name or default, passed, f"got {describe(target)}")

    def expect_audit(
        self,
        kind: EventKind,
        *,
        reason: str | None = None,
        rule_id: str | None = None,
        run: RunContext | None = None,
        name: str | None = None,
    ) -> bool:
        """At least one audit event of this kind, with this reason, rule and run if given."""
        of_kind = self.events(kind, run)
        matching = [
            e
            for e in of_kind
            if (reason is None or e.reason == reason) and (rule_id is None or rule_id in e.rule_ids)
        ]
        parts = [f"audit {EventKind(kind).value}"]
        if reason is not None:
            parts.append(reason)
        if rule_id is not None:
            parts.append(f"rule {rule_id}")
        seen = sorted({e.reason for e in of_kind})
        detail = f"{len(matching)} matching; reasons seen: {', '.join(seen) or 'none'}"
        return self.expect(name or " ".join(parts), bool(matching), detail)

    def expect_outbox_empty(self, name: str | None = None) -> bool:
        count = len(self.outbox())
        return self.expect(name or "outbox empty", count == 0, f"{count} emails")

    # Transport

    def call_counts(self) -> tuple[int, int, tuple[tuple[str, int], ...]]:
        """(calls over MCP, direct calls, direct calls per reason sorted by reason)."""
        by_reason = tuple(sorted((reason.value, n) for reason, n in self.direct_calls.items()))
        return self.mcp_calls, sum(self.direct_calls.values()), by_reason

    def _issue(self, ctx: RunContext) -> RunContext:
        self._issued.append(ctx)
        return ctx

    def _route_call(
        self,
        ctx: RunContext,
        tool: str,
        action: Action | str | None,
        args: dict[str, JsonValue],
        *,
        loop: bool = False,
    ) -> Outcome:
        """One tool call, over the run's MCP session or straight to the gate. For the agent
        loop (`loop`), the action is the one the loop chose and the loop shows the outcome
        to the model itself; for a direct proposal, what it shows is scanned here."""
        route = self._route(ctx, tool, None if loop else action)
        if isinstance(route, GatewaySession):
            return self._over_mcp(ctx, lambda: route.call(tool, args))
        self.direct_calls[route] += 1
        chosen = self._action(ctx, tool) if action is None else action
        outcome = self.gate.call_tool(ctx, tool, cast(Action, chosen), args)
        if not loop:
            self._saw(ctx, outcome)
        return outcome

    def _route(
        self, ctx: RunContext, tool: str, action: Action | str | None
    ) -> GatewaySession | Direct:
        """The session a call goes over, or why it goes to the gate directly."""
        if self.transport is Transport.GATE:
            return Direct.GATE_TRANSPORT
        if not any(ctx == issued for issued in self._issued):
            return Direct.FORGED_CONTEXT
        if action is not None and action != self._gateway_action(ctx, tool):
            return Direct.ACTION_OVERRIDE
        session = self._session(ctx)
        return Direct.GATEWAY_REFUSED if session is None else session

    def _session(self, ctx: RunContext) -> GatewaySession | None:
        """The run's session, opened on first use; None if the gateway will not launch."""
        session = self._sessions.get(ctx.run_id)
        if session is not None or ctx.run_id in self._refused:
            return session
        world = replace(self.world, gate=self.gate, engines=dict(self._engines))
        try:
            server = build_gateway(world, ctx)
        except ValueError:
            # A customer with no policy: the gateway refuses to launch for this run.
            self._refused.add(ctx.run_id)
            return None
        return self._sessions.open(ctx.run_id, server)

    def _gateway_action(self, ctx: RunContext, tool: str) -> Action:
        """The action the gateway calls a tool with: its listed spec's, else READ."""
        for spec in self.specs(ctx.customer_id):
            if spec.tool == tool:
                return spec.action
        return Action.READ

    def _over_mcp(self, ctx: RunContext, send: Callable[[], McpReply]) -> Outcome:
        refused = self.sink.refused
        self.mcp_calls += 1
        try:
            reply = send()
        except GatewayCallError as exc:
            if self.sink.refused > refused:
                # The arena's own sink failed the call: what the host sees in-process.
                raise AuditDown("audit sink unavailable") from exc
            raise
        run_id = getattr(ctx, "run_id", None)
        self._exposures.append(Exposure(run_id if isinstance(run_id, str) else None, reply.text))
        return reply.outcome

    # Internals

    def _action(self, ctx: RunContext, tool: str) -> Action:
        engine = self._engines.get(getattr(ctx, "customer_id", ""))
        config = None if engine is None else engine.bundle.tools.get(tool)
        if config is None:
            return Action.READ
        return sorted(config.actions)[0]

    def _saw(self, ctx: RunContext, outcome: Outcome) -> None:
        run_id = getattr(ctx, "run_id", None)
        text = _visible_text(outcome)
        self._exposures.append(Exposure(run_id if isinstance(run_id, str) else None, text))


def describe(value: object) -> str:
    """A short, deterministic description of an outcome or result, with no ids or data."""
    if isinstance(value, Executed):
        return f"executed {value.tool}"
    if isinstance(value, Denied):
        return f"denied {value.reason}"
    if isinstance(value, PendingApproval):
        return "pending approval"
    if isinstance(value, CapReached):
        return f"cap reached {value.cap.value}"
    if isinstance(value, AgentTranscript):
        return f"run stopped {value.stop_reason}"
    if isinstance(value, IdentityError):
        return f"identity error {value.reason}"
    if isinstance(value, ApprovalError):
        return f"approval error {value.reason.value}"
    if isinstance(value, Approval):
        return f"approval {value.status.value}"
    if isinstance(value, RunContext):
        return "run started"
    return type(value).__name__


def _visible_text(outcome: Outcome) -> str:
    """What the caller of the gate gets to see of an outcome, as the agent loop shows it."""
    if isinstance(outcome, Executed):
        return _dumps(outcome.content)
    if isinstance(outcome, Denied):
        return _dumps({"error": outcome.reason})
    if isinstance(outcome, PendingApproval):
        return _dumps({"approval_id": outcome.approval_id})
    if isinstance(outcome, CapReached):
        return _dumps({"error": "cap_reached", "cap": outcome.cap.value})
    return type(outcome).__name__


def _dumps(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


# Running


@dataclass(frozen=True)
class SuiteResult:
    # Sorted by scenario id.
    results: tuple[ScenarioResult, ...]
    # (passed, total) per category that had at least one selected scenario, in Category order.
    totals: dict[Category, tuple[int, int]]
    # Passed over total for benign scenarios, or None when none ran.
    benign_rate: float | None
    transport: Transport = Transport.GATE

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failed(self) -> tuple[ScenarioResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    @property
    def mcp_calls(self) -> int:
        return sum(result.mcp_calls for result in self.results)

    @property
    def direct_calls(self) -> int:
        return sum(result.direct_calls for result in self.results)

    def direct_by_reason(self) -> dict[str, int]:
        """Direct calls of the whole run per reason, sorted by reason."""
        totals: Counter[str] = Counter()
        for result in self.results:
            for reason, count in result.direct_by_reason:
                totals[reason] += count
        return dict(sorted(totals.items()))

    def mcp_coverage(self) -> dict[Category, tuple[int, int]]:
        """(scenarios with at least one call over MCP, scenarios) per category that ran, in
        Category order."""
        coverage: dict[Category, tuple[int, int]] = {}
        for category in Category:
            of_category = [r for r in self.results if r.category is category]
            if of_category:
                reached = sum(1 for r in of_category if r.mcp_calls > 0)
                coverage[category] = (reached, len(of_category))
        return coverage


def run_scenario(
    s: RegisteredScenario,
    *,
    policies: Path = DEFAULT_POLICIES,
    fixtures: Path = DEFAULT_FIXTURES,
    transport: Transport = Transport.GATE,
) -> ScenarioResult:
    """Run one scenario in a fresh arena and judge it by the pass criteria above."""
    error: str | None = None
    checks: tuple[Check, ...] = ()
    leaks: tuple[str, ...] = ()
    arena = Arena(policies=policies, fixtures=fixtures, transport=transport)
    try:
        try:
            s.fn(arena)
        except Exception as exc:
            # The type only: a message could quote arguments or document text.
            error = type(exc).__name__
        checks = tuple(arena.checks)
        try:
            leaks = tuple(arena.leaks())
        except Exception as exc:
            error = error or type(exc).__name__
    finally:
        try:
            arena.close()
        except Exception as exc:
            error = error or type(exc).__name__
    mcp_calls, direct_calls, direct_by_reason = arena.call_counts()
    passed = error is None and bool(checks) and all(c.passed for c in checks) and not leaks
    return ScenarioResult(
        id=s.id,
        category=s.category,
        title=s.title,
        benign=s.benign,
        passed=passed,
        checks=checks,
        leaks=leaks,
        error=error,
        mcp_calls=mcp_calls,
        direct_calls=direct_calls,
        direct_by_reason=direct_by_reason,
    )


def run_suite(
    *,
    categories: Iterable[Category] | None = None,
    ids: Iterable[str] | None = None,
    policies: Path = DEFAULT_POLICIES,
    fixtures: Path = DEFAULT_FIXTURES,
    transport: Transport = Transport.GATE,
) -> SuiteResult:
    """Run every scenario (or those of the given categories and ids), sorted by id, over the
    given transport.

    An id that matches no scenario raises ValueError, so a typo cannot quietly run nothing.
    """
    available = all_scenarios()
    wanted_categories = None if categories is None else {Category(c) for c in categories}
    wanted_ids = None if ids is None else set(ids)
    if wanted_ids is not None:
        unknown = sorted(wanted_ids - {s.id for s in available})
        if unknown:
            raise ValueError(f"unknown scenario ids: {', '.join(unknown)}")
    selected = [
        s
        for s in available
        if (wanted_categories is None or s.category in wanted_categories)
        and (wanted_ids is None or s.id in wanted_ids)
    ]
    chosen = Transport(transport)
    results = tuple(
        run_scenario(s, policies=policies, fixtures=fixtures, transport=chosen) for s in selected
    )
    totals: dict[Category, tuple[int, int]] = {}
    for category in Category:
        of_category = [r for r in results if r.category is category]
        if of_category:
            totals[category] = (sum(1 for r in of_category if r.passed), len(of_category))
    benign = [r for r in results if r.benign]
    rate = sum(1 for r in benign if r.passed) / len(benign) if benign else None
    return SuiteResult(results=results, totals=totals, benign_rate=rate, transport=chosen)
