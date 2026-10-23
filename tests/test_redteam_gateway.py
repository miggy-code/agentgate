"""The red-team suite through the MCP gateway: the whole suite over `--transport mcp`, how much
of it really went over MCP, gateway mutations that only the mcp transport catches, outcome
parsing for every outcome kind, the routing rules for calls MCP cannot express, and
determinism.
"""

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.types import CallToolResult, ImageContent, TextContent

import agentgate.gateway.server as gateway_server
from agentgate.agent.mock import ScriptedModel
from agentgate.audit.base import EventKind
from agentgate.context import RunContext
from agentgate.gate import CapReached, Denied, Executed, Gate, Outcome, PendingApproval
from agentgate.gateway import EXECUTE_TOOL, build_gateway
from agentgate.models import Action, Role
from agentgate.redteam import core
from agentgate.redteam.core import (
    Arena,
    AuditDown,
    Category,
    SuiteResult,
    describe,
    run_suite,
    transcript_calls,
)
from agentgate.redteam.transport import (
    COST_NOT_SENT,
    Direct,
    GatewayCallError,
    GatewayProtocolError,
    Transport,
    parse_result,
)
from agentgate.spend import CapKind
from agentgate.world import World
from conftest import FIXTURES, JR, KC, POLICIES

MCP = Transport.MCP
GATE = Transport.GATE
SERVICE_MENU = "kc-service-menu"
PAYROLL = "kc-payroll-summary"
CUSTOMER = "beatrice.l@mail-sample.example"

# Failures under mcp that are real gateway findings, pinned with the checks that fail, so a
# fix (or a new failure) shows up here. None at present. ab-001 was one until the gateway
# stopped refusing approvals_execute wrappers on its own, unaudited (an empty approval id
# was invalid_args with no audit event, where the gate says not_found and records it).
KNOWN_MCP_FAILURES: dict[str, tuple[str, ...]] = {}
# Reasons a scenario may make no call over MCP at all although it makes agent-side calls:
# every one of them is a call MCP cannot express.
INEXPRESSIBLE = {Direct.FORGED_CONTEXT.value, Direct.ACTION_OVERRIDE.value}


def suite(transport: Transport, **kwargs: Any) -> SuiteResult:
    return run_suite(policies=POLICIES, fixtures=FIXTURES, transport=transport, **kwargs)


def failed(result: SuiteResult) -> list[str]:
    return [r.id for r in result.failed]


@pytest.fixture(scope="module")
def mcp_suite() -> SuiteResult:
    return suite(MCP)


@pytest.fixture
def arena() -> Iterator[Arena]:
    with Arena(policies=POLICIES, fixtures=FIXTURES, transport=MCP) as built:
        yield built


# The whole suite over MCP


def test_the_suite_passes_over_mcp_except_the_pinned_gateway_findings(
    mcp_suite: SuiteResult,
) -> None:
    assert mcp_suite.transport is MCP
    assert mcp_suite.total == len(core.all_scenarios())
    failing = {
        r.id: tuple(c.name for c in r.checks if not c.passed)
        for r in mcp_suite.failed
        if r.error is None and not r.leaks
    }
    assert [r.id for r in mcp_suite.failed] == sorted(failing), "errors or leaks under mcp"
    assert failing == KNOWN_MCP_FAILURES


def test_made_up_approval_ids_are_audited_over_mcp(mcp_suite: SuiteResult) -> None:
    [result] = [r for r in mcp_suite.results if r.id == "ab-001"]
    assert result.passed
    assert result.mcp_calls > 0 and result.direct_calls == 0


def test_every_attack_category_reaches_the_gateway(mcp_suite: SuiteResult) -> None:
    coverage = mcp_suite.mcp_coverage()
    assert list(coverage) == list(Category)
    for category in Category:
        of_category = [r for r in mcp_suite.results if r.category is category]
        agent_side = sum(r.mcp_calls + r.direct_calls for r in of_category)
        if agent_side:
            reached, total = coverage[category]
            assert reached > 0, category
            assert sum(r.mcp_calls for r in of_category) > 0, category
            assert total == len(of_category)
    assert mcp_suite.mcp_calls > 10 * mcp_suite.direct_calls


def test_a_scenario_without_mcp_calls_only_made_calls_mcp_cannot_express(
    mcp_suite: SuiteResult,
) -> None:
    untouched = [r for r in mcp_suite.results if r.mcp_calls == 0]
    for result in untouched:
        reasons = {reason for reason, _ in result.direct_by_reason}
        assert reasons <= INEXPRESSIBLE, (result.id, reasons)
    # Benign work and the injection and approval attacks are all agent-side.
    for category in (Category.BENIGN, Category.PROMPT_INJECTION, Category.APPROVAL_BYPASS):
        reached, total = mcp_suite.mcp_coverage()[category]
        assert reached == total, category


def test_direct_calls_are_counted_by_reason(mcp_suite: SuiteResult) -> None:
    by_reason = mcp_suite.direct_by_reason()
    assert set(by_reason) <= {d.value for d in Direct} - {Direct.GATE_TRANSPORT.value}
    assert sum(by_reason.values()) == mcp_suite.direct_calls
    for result in mcp_suite.results:
        assert sum(n for _, n in result.direct_by_reason) == result.direct_calls


def test_the_gate_transport_makes_no_mcp_calls() -> None:
    result = suite(GATE, categories=[Category.APPROVAL_BYPASS, Category.BENIGN])
    assert result.transport is GATE
    assert result.mcp_calls == 0
    assert result.direct_calls > 0
    assert result.direct_by_reason() == {Direct.GATE_TRANSPORT.value: result.direct_calls}


def test_the_mcp_suite_is_deterministic() -> None:
    categories = [Category.APPROVAL_BYPASS, Category.FAIL_CLOSED, Category.IDENTITY_SPOOFING]
    first = suite(MCP, categories=categories)
    second = suite(MCP, categories=categories)
    assert first.results
    assert first == second


# Mutations: a broken gateway fails scenarios under mcp and nothing under gate


def _swapped_names(real: Any) -> Any:
    swap = {"docs.read": "docs_search", "docs.search": "docs_read"}
    return lambda tool: swap.get(tool) or real(tool)


def test_a_broken_name_mapping_is_caught_only_over_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["bn-001", "pe-001", "pi-001", "xt-001"]
    assert suite(MCP, ids=ids).all_passed
    monkeypatch.setattr(gateway_server, "mcp_name", _swapped_names(gateway_server.mcp_name))
    assert failed(suite(MCP, ids=ids)) == ids
    assert suite(GATE, ids=ids).all_passed


def test_a_gateway_that_misreports_denials_is_caught_only_over_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = ["ab-002", "fc-009", "sr-010", "xt-001"]
    real = gateway_server._result

    def every_denial_is_default_deny(outcome: Outcome) -> CallToolResult:
        result = real(outcome)
        if not result.is_error:
            return result
        text = json.dumps({"error": "default_deny"})
        return CallToolResult(content=[TextContent(text=text)], is_error=True)

    monkeypatch.setattr(gateway_server, "_result", every_denial_is_default_deny)
    assert failed(suite(MCP, ids=ids)) == ids
    assert suite(GATE, ids=ids).all_passed


class _LenientGate:
    """A gate behind the gateway that drops identity keys instead of refusing them."""

    def __init__(self, gate: Gate) -> None:
        self._gate = gate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gate, name)

    def call_tool(self, ctx: RunContext, tool: str, action: Action, args: Any) -> Outcome:
        dropped = {"user_id", "role", "customer_id", "run_id"}
        kept = {key: value for key, value in args.items() if key not in dropped}
        return self._gate.call_tool(ctx, tool, action, kept)


def test_a_gateway_without_the_strict_args_check_is_caught_only_over_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = ["id-009", "xt-009"]
    assert suite(MCP, ids=ids).all_passed

    def lenient(world: World, ctx: RunContext) -> Any:
        return build_gateway(replace(world, gate=_LenientGate(world.gate)), ctx)

    monkeypatch.setattr(core, "build_gateway", lenient)
    assert failed(suite(MCP, ids=ids)) == ids
    assert suite(GATE, ids=ids).all_passed


# Outcome parsing

EXPIRES = datetime(2026, 10, 22, 17, 0, tzinfo=UTC)
OUTCOMES: list[Outcome] = [
    Executed(
        "docs.read",
        {"id": "kc-x", "body": "café → ok", "nested": {"a": [1, "b", None, True, 2.5]}},
        Decimal("0.01"),
    ),
    Executed("email.send", {"status": "queued"}, Decimal(0)),
    # Content that only looks like a pending approval is still executed content.
    Executed("notes.write", {"status": "pending_approval"}, Decimal(0)),
    Denied("not_found"),
    Denied("invalid_args"),
    PendingApproval("apr_" + "a" * 32, EXPIRES),
    *(CapReached(cap) for cap in CapKind),
]


@pytest.mark.parametrize("outcome", OUTCOMES, ids=describe)
def test_parse_round_trips_every_outcome_kind(outcome: Outcome) -> None:
    tool = outcome.tool if isinstance(outcome, Executed) else "docs.read"
    reply = parse_result(tool, gateway_server._result(outcome))
    if isinstance(outcome, Executed):
        assert isinstance(reply.outcome, Executed)
        assert (reply.outcome.tool, reply.outcome.content) == (outcome.tool, outcome.content)
        # The gateway never sends a call's cost.
        assert reply.outcome.cost_usd.is_nan() and COST_NOT_SENT.is_nan()
    else:
        assert reply.outcome == outcome
    assert json.loads(reply.text) is not None


def _text(value: Any, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(text=json.dumps(value))], is_error=is_error)


@pytest.mark.parametrize(
    "result",
    [
        CallToolResult(content=[], is_error=False),
        CallToolResult(content=[TextContent(text="{}"), TextContent(text="{}")], is_error=False),
        CallToolResult(content=[ImageContent(data="AA==", mime_type="image/png")]),
        CallToolResult(content=[TextContent(text="not json")], is_error=False),
        _text(["a list"]),
        _text({"reason": "default_deny"}, is_error=True),
        _text({"error": "default_deny", "detail": "x"}, is_error=True),
        _text({"error": "cap_reached", "cap": "no_such_cap"}, is_error=True),
        _text({"status": "pending_approval", "approval_id": 7, "expires_at": "x"}),
        _text({"status": "pending_approval", "approval_id": "apr_x", "expires_at": "soon"}),
    ],
)
def test_parse_refuses_what_the_gateway_never_sends(result: CallToolResult) -> None:
    with pytest.raises(GatewayProtocolError):
        parse_result("docs.read", result)


def _sequence(arena: Arena) -> list[Outcome]:
    """One outcome of every kind, through the arena's transport."""
    staff = arena.start("kc-staff-nadine")
    manager = arena.start("kc-mgr-marisol")
    email = {"to": [CUSTOMER], "subject": "Your visit", "body": "See you Monday."}
    outcomes = [
        arena.call(staff, "docs.read", {"document_id": SERVICE_MENU}),
        arena.call(staff, "docs.read", {"document_id": PAYROLL}),
        arena.call(manager, "email.send", email),
    ]
    limit = arena.engines[KC].bundle.policy.spend.max_calls_per_run
    for _ in range(limit):
        outcomes.append(arena.call(staff, "docs.search", {"query": "furnace"}))
    return outcomes


def test_every_outcome_kind_arrives_over_mcp_as_it_does_in_process() -> None:
    with Arena(policies=POLICIES, fixtures=FIXTURES, transport=GATE) as gate_arena:
        in_process = _sequence(gate_arena)
    with Arena(policies=POLICIES, fixtures=FIXTURES, transport=MCP) as mcp_arena:
        over_mcp = _sequence(mcp_arena)
        assert mcp_arena.call_counts() == (len(over_mcp), 0, ())
    assert [describe(o) for o in over_mcp] == [describe(o) for o in in_process]
    kinds = {type(o) for o in over_mcp}
    assert kinds == {Executed, Denied, PendingApproval, CapReached}
    for mcp_outcome, gate_outcome in zip(over_mcp, in_process, strict=True):
        if isinstance(gate_outcome, Executed):
            assert isinstance(mcp_outcome, Executed)
            assert mcp_outcome.content == gate_outcome.content
        if isinstance(gate_outcome, PendingApproval):
            assert isinstance(mcp_outcome, PendingApproval)
            assert mcp_outcome.expires_at == gate_outcome.expires_at


# Routing


def test_calls_go_over_one_session_per_run(arena: Arena) -> None:
    staff = arena.start("kc-staff-nadine")
    arena.expect_executed(arena.call(staff, "docs.read", {"document_id": SERVICE_MENU}))
    session = arena._sessions.get(staff.run_id)
    assert session is not None
    assert EXECUTE_TOOL in session.names and "docs_read" in session.names
    arena.expect_denied(arena.call(staff, "docs.read", {"document_id": PAYROLL}), "not_found")
    assert arena._sessions.get(staff.run_id) is session
    other = arena.start("kc-staff-nadine")
    arena.call(other, "docs.read", {"document_id": SERVICE_MENU})
    assert arena._sessions.get(other.run_id) not in {None, session}
    assert arena.call_counts() == (3, 0, ())
    assert all(check.passed for check in arena.checks)


def test_a_forged_context_goes_to_the_gate_directly(arena: Arena) -> None:
    staff = arena.start("kc-staff-nadine")
    forged = replace(staff, role=Role.OWNER)
    outcome = arena.call(forged, "docs.read", {"document_id": PAYROLL})
    assert outcome == Denied("invalid_run")
    assert arena.call_counts() == (0, 1, ((Direct.FORGED_CONTEXT.value, 1),))
    # An equal context the arena started is not forged.
    same = replace(staff)
    assert isinstance(arena.call(same, "docs.read", {"document_id": SERVICE_MENU}), Executed)
    assert arena.call_counts()[0] == 1


def test_an_action_the_tool_does_not_have_goes_to_the_gate_directly(arena: Arena) -> None:
    staff = arena.start("kc-staff-nadine")
    read = {"document_id": SERVICE_MENU}
    assert isinstance(arena.call(staff, "docs.read", read, action=Action.SEND), Denied)
    # The tool's own action is expressible, so it goes over MCP.
    assert isinstance(arena.call(staff, "docs.read", read, action=Action.READ), Executed)
    assert arena.call_counts() == (1, 1, ((Direct.ACTION_OVERRIDE.value, 1),))


def test_a_run_the_gateway_will_not_launch_for_goes_directly(arena: Arena) -> None:
    staff = arena.start("kc-staff-nadine")
    jr_staff = arena.start("jr-staff-priya")
    arena.call(staff, "docs.read", {"document_id": SERVICE_MENU})
    before = arena._sessions.get(staff.run_id)
    arena.regate(engines={JR: arena.engines[JR]})
    # regate restarts the gateways: the old session is closed and gone.
    assert arena._sessions.get(staff.run_id) is None and before is not None
    assert arena.call(staff, "docs.read", {"document_id": SERVICE_MENU}) == Denied("no_policy")
    jr_read = arena.call(jr_staff, "docs.read", {"document_id": "jr-return-policy"})
    assert isinstance(jr_read, Executed)
    assert arena.call_counts() == (2, 1, ((Direct.GATEWAY_REFUSED.value, 1),))


def test_execute_goes_through_approvals_execute(arena: Arena) -> None:
    manager = arena.start("kc-mgr-marisol")
    email: dict[str, Any] = {"to": [CUSTOMER], "subject": "Your visit", "body": "Monday."}
    pending = arena.call(manager, "email.send", email)
    assert isinstance(pending, PendingApproval)
    assert arena.execute(manager, pending.approval_id, "email.send", email) == Denied(
        "not_approved"
    )
    arena.approve(pending.approval_id, "kc-mgr-terrence")
    executed = arena.execute(manager, pending.approval_id, "email.send", email)
    assert isinstance(executed, Executed) and executed.tool == "email.send"
    assert len(arena.outbox()) == 1
    assert arena.call_counts() == (3, 0, ())


def test_the_agent_loop_calls_tools_over_mcp_and_spends_in_process(
    arena: Arena, monkeypatch: pytest.MonkeyPatch
) -> None:
    staff = arena.start("kc-staff-nadine")
    charged: list[Decimal] = []
    settled: list[Decimal] = []
    charge, settle = arena.gate.charge_model, arena.gate.settle_model

    def charge_model(ctx: RunContext, estimate: Decimal) -> Any:
        charged.append(estimate)
        return charge(ctx, estimate)

    def settle_model(ctx: RunContext, reservation: Any, actual: Decimal) -> None:
        settled.append(actual)
        settle(ctx, reservation, actual)

    monkeypatch.setattr(arena.gate, "charge_model", charge_model)
    monkeypatch.setattr(arena.gate, "settle_model", settle_model)
    model = ScriptedModel(
        "scripted",
        [
            [("docs.search", {"query": "furnace"})],
            [("docs.read", {"document_id": PAYROLL}), ("shell.exec", {"cmd": "ls"})],
        ],
    )
    transcript = arena.run(staff, model, "Look around.")
    statuses = [(c.tool, o.status, o.content.get("error")) for c, o in transcript_calls(transcript)]
    assert statuses == [
        ("docs.search", "executed", None),
        ("docs.read", "denied", "not_found"),
        ("shell.exec", "denied", "unknown_tool"),
    ]
    assert arena.call_counts() == (3, 0, ())
    # Model spend was charged and settled on the gate in-process, once per model call.
    assert transcript.stop_reason == "final"
    assert len(charged) == len(settled) == len(transcript.steps) == 3


def test_an_audit_failure_inside_the_gateway_reads_as_audit_down(arena: Arena) -> None:
    manager = arena.start("kc-mgr-marisol")
    email = {"to": [CUSTOMER], "subject": "Your visit", "body": "See you Monday."}
    arena.break_audit(lambda event: event.kind is EventKind.APPROVAL_REQUIRED)
    with pytest.raises(AuditDown):
        arena.call(manager, "email.send", email)
    arena.restore_audit()
    assert arena.gate.approvals.list(KC) == []
    assert arena.call_counts() == (1, 0, ())


def test_any_other_gateway_error_is_a_gateway_call_error(
    arena: Arena, monkeypatch: pytest.MonkeyPatch
) -> None:
    staff = arena.start("kc-staff-nadine")

    def boom(*_args: Any) -> Outcome:
        raise RuntimeError("not an audit failure")

    monkeypatch.setattr(arena.gate, "call_tool", boom)
    with pytest.raises(GatewayCallError) as caught:
        arena.call(staff, "docs.read", {"document_id": SERVICE_MENU})
    assert "not an audit failure" not in str(caught.value)


# The gateway finding behind the former ab-001 failure, as its own test.


def test_the_gateway_audits_a_malformed_execute_request(arena: Arena) -> None:
    staff = arena.start("kc-staff-nadine")
    world = replace(arena.world, gate=arena.gate)
    request = {"approval_id": "", "tool": "email_send", "args": {}}

    async def send() -> None:
        async with Client(build_gateway(world, staff), mode="legacy") as client:
            await client.call_tool(EXECUTE_TOOL, request)

    before = len(arena.sink.events)
    anyio.run(send)
    [event] = arena.sink.events[before:]
    assert (event.kind, event.reason, event.user_id, event.run_id) == (
        EventKind.DENY,
        "approval_not_found",
        "kc-staff-nadine",
        staff.run_id,
    )
