"""The MCP gateway through a real MCP client: tool listing, decisions, approvals, caps and the
audit trail, in-process over both the legacy handshake and the 2026 per-request protocol,
plus one run of `agentgate gateway serve` as a subprocess over stdio.
"""

import json
import re
import shutil
import sqlite3
import sys
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client, MCPError, StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent

from agentgate.agent.loop import run_agent, tool_specs
from agentgate.agent.mock import ScriptedModel
from agentgate.audit.base import AuditEvent, EventKind, MemorySink
from agentgate.audit.panoptes import PanoptesSink
from agentgate.cli import DEMO_IDENTITY_NOTE
from agentgate.clock import FixedClock
from agentgate.context import RunContext
from agentgate.gateway import EXECUTE_TOOL, build_gateway, gateway_tools, mcp_name
from agentgate.spend import CapKind
from agentgate.tools.builtin import default_tools, list_outbox
from agentgate.world import World, open_memory_world
from conftest import FIXTURES, JR, KC, POLICIES, START, assert_no_canaries

EXFIL = "records-review@exfil-drop.example"
DIRECTORY_CUSTOMER = "beatrice.l@mail-sample.example"
OTHER_DIRECTORY_CUSTOMER = "ruben.c@mail-sample.example"
KC_NAMES = {"docs_search", "docs_read", "notes_write", "email_send", "parts_order", EXECUTE_TOOL}
JR_NAMES = {"docs_search", "docs_read", "notes_write", "email_send", "refund_issue", EXECUTE_TOOL}


def email_args(*to: str, subject: str = "Your visit", body: str = "See you Monday.") -> Any:
    return {"to": list(to), "subject": subject, "body": body}


def _world(sink: MemorySink | None = None) -> World:
    return open_memory_world(
        policies=POLICIES,
        fixtures=FIXTURES,
        clock=FixedClock(START),
        sink=MemorySink() if sink is None else sink,
    )


@dataclass
class Gateway:
    """A world, and MCP clients connected to gateways bound to runs in it."""

    world: World
    mode: str

    @property
    def events(self) -> list[AuditEvent]:
        sink = self.world.sink
        assert isinstance(sink, MemorySink)
        return sink.events

    def start(self, user_id: str) -> RunContext:
        user = self.world.store.get_user(user_id)
        assert user is not None
        return self.world.gate.start_run(user.customer_id, user_id)

    def run(self, ctx: RunContext, scenario: Callable[[Client], Awaitable[None]]) -> None:
        """Connect a client to a gateway for ctx and run the scenario with it."""

        async def connected() -> None:
            async with Client(build_gateway(self.world, ctx), mode=self.mode) as client:
                await scenario(client)

        anyio.run(connected)


# Legacy runs the initialize handshake over JSON-RPC streams; auto takes the 2026-07-28
# per-request path, with no handshake at all.
@pytest.fixture(params=["legacy", "auto"])
def gw(request: pytest.FixtureRequest) -> Iterator[Gateway]:
    world = _world()
    built = Gateway(world, request.param)
    yield built
    assert_no_canaries(built.events)
    world.close()


def payload(result: CallToolResult) -> Any:
    [block] = result.content
    assert isinstance(block, TextContent)
    assert result.structured_content is None
    return json.loads(block.text)


async def call(client: Client, name: str, args: dict[str, Any]) -> tuple[bool, Any]:
    result = await client.call_tool(name, args)
    return result.is_error, payload(result)


def last(gw: Gateway, kind: EventKind) -> AuditEvent:
    matching = [e for e in gw.events if e.kind is kind]
    assert matching, f"no {kind} event"
    return matching[-1]


# Listing


@pytest.mark.parametrize(
    ("user_id", "names"), [("kc-staff-nadine", KC_NAMES), ("jr-mgr-daniel", JR_NAMES)]
)
def test_list_tools_per_customer(gw: Gateway, user_id: str, names: set[str]) -> None:
    ctx = gw.start(user_id)
    engine = gw.world.engines[ctx.customer_id]
    specs = {mcp_name(spec.tool): spec for spec in tool_specs(engine, default_tools())}

    async def scenario(client: Client) -> None:
        listed = (await client.list_tools()).tools
        assert {tool.name for tool in listed} == names
        for tool in listed:
            schema = tool.input_schema
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            if tool.name == EXECUTE_TOOL:
                assert set(schema["properties"]) == {"approval_id", "tool", "args"}
                assert schema["required"] == ["approval_id", "tool", "args"]
                continue
            spec = specs[tool.name]
            assert schema == spec.input_schema
            assert tool.description == spec.description

    gw.run(ctx, scenario)
    assert [tool.name for tool in gateway_tools(gw.world, ctx)] == [*specs, EXECUTE_TOOL]


def test_no_tool_takes_an_identity(gw: Gateway) -> None:
    ctx = gw.start("kc-owner-dale")

    async def scenario(client: Client) -> None:
        for tool in (await client.list_tools()).tools:
            properties = set(tool.input_schema["properties"])
            assert not properties & {"user_id", "role", "customer_id", "run_id"}

    gw.run(ctx, scenario)


def test_mcp_names() -> None:
    assert mcp_name("docs.search") == "docs_search"
    with pytest.raises(ValueError, match="MCP tool name"):
        mcp_name("")


# Reads and search


def test_allowed_read_returns_the_content(gw: Gateway) -> None:
    ctx = gw.start("kc-staff-nadine")
    document = gw.world.store.get_document("kc-service-menu")
    assert document is not None

    async def scenario(client: Client) -> None:
        error, content = await call(client, "docs_read", {"document_id": "kc-service-menu"})
        assert not error
        assert (content["id"], content["title"], content["label"]) == (
            document.id,
            document.title,
            document.label,
        )
        assert content["body"] == document.body[:4000]

    gw.run(ctx, scenario)
    assert last(gw, EventKind.EXECUTED).tool == "docs.read"


def test_restricted_read_is_not_found(gw: Gateway) -> None:
    ctx = gw.start("kc-staff-nadine")

    async def scenario(client: Client) -> None:
        restricted = await call(client, "docs_read", {"document_id": "kc-payroll-summary"})
        missing = await call(client, "docs_read", {"document_id": "kc-payroll-summary-2025"})
        assert restricted == missing == (True, {"error": "not_found"})

    gw.run(ctx, scenario)
    denied = [e for e in gw.events if e.kind is EventKind.DENY]
    assert [(e.reason, e.subject) for e in denied] == [
        ("default_deny", "kc-payroll-summary"),
        ("document_not_found", "kc-payroll-summary-2025"),
    ]


def test_cross_tenant_read_is_not_found(gw: Gateway) -> None:
    ctx = gw.start("kc-owner-dale")

    async def scenario(client: Client) -> None:
        outcome = await call(client, "docs_read", {"document_id": "jr-customer-list"})
        assert outcome == (True, {"error": "not_found"})

    gw.run(ctx, scenario)
    event = last(gw, EventKind.DENY)
    assert (event.customer_id, event.subject) == (KC, "jr-customer-list")


def test_search_returns_only_permitted_documents(gw: Gateway) -> None:
    ctx = gw.start("kc-staff-nadine")
    query = {"query": "pricing margins customer list payroll service", "limit": 20}

    async def scenario(client: Client) -> None:
        error, content = await call(client, "docs_search", query)
        assert not error
        returned = [doc["id"] for doc in content["documents"]]
        assert returned and content["count"] == len(returned)
        assert {doc["label"] for doc in content["documents"]} <= {"public", "internal"}
        assert all(doc_id.startswith("kc-") for doc_id in returned)
        assert last(gw, EventKind.RETRIEVAL).details["returned_ids"] == returned

    gw.run(ctx, scenario)


# Identity


@pytest.mark.parametrize(
    ("key", "value"),
    [("user_id", "kc-owner-dale"), ("role", "owner"), ("customer_id", JR)],
)
def test_identity_keys_are_invalid_args(gw: Gateway, key: str, value: str) -> None:
    ctx = gw.start("kc-staff-nadine")

    async def scenario(client: Client) -> None:
        read = {"document_id": "kc-payroll-summary", key: value}
        assert await call(client, "docs_read", read) == (True, {"error": "invalid_args"})
        execute = {"approval_id": "apr_x", "tool": "docs_read", "args": {}, key: value}
        assert await call(client, EXECUTE_TOOL, execute) == (True, {"error": "invalid_args"})
        error, _ = await call(client, "docs_read", {"document_id": "kc-service-menu"})
        assert not error

    gw.run(ctx, scenario)
    assert last(gw, EventKind.DENY).reason == "invalid_args"
    assert gw.events and all(
        (e.customer_id, e.user_id, e.role, e.run_id) == (KC, "kc-staff-nadine", "staff", ctx.run_id)
        for e in gw.events
    )


def test_unknown_tool_is_denied_by_the_gate(gw: Gateway) -> None:
    ctx = gw.start("kc-staff-nadine")

    async def scenario(client: Client) -> None:
        assert await call(client, "shell_exec", {"cmd": "ls"}) == (
            True,
            {"error": "unknown_tool"},
        )

    gw.run(ctx, scenario)
    event = last(gw, EventKind.DENY)
    assert (event.reason, event.tool, event.user_id) == ("unknown_tool", None, "kc-staff-nadine")


# Email and approvals


def test_email_outside_the_company_is_denied(gw: Gateway) -> None:
    ctx = gw.start("kc-staff-nadine")

    async def scenario(client: Client) -> None:
        outcome = await call(client, "email_send", email_args(EXFIL))
        assert outcome == (True, {"error": "default_deny"})

    gw.run(ctx, scenario)
    assert list_outbox(gw.world.store) == []
    assert gw.world.gate.approvals.list(KC) == []


def test_approval_flow(gw: Gateway) -> None:
    ctx = gw.start("kc-mgr-marisol")
    args = email_args(DIRECTORY_CUSTOMER, subject="zq-subject-marker", body="zq-body-marker")

    async def scenario(client: Client) -> None:
        error, pending = await call(client, "email_send", args)
        assert not error
        assert set(pending) == {"status", "approval_id", "expires_at"}
        assert pending["status"] == "pending_approval"
        approval_id = pending["approval_id"]
        approval = gw.world.gate.approvals.get(approval_id)
        assert approval is not None
        assert pending["expires_at"] == approval.expires_at.isoformat()
        assert list_outbox(gw.world.store) == []

        def execute(with_args: Any, tool: str = "email_send") -> Awaitable[tuple[bool, Any]]:
            request = {"approval_id": approval_id, "tool": tool, "args": with_args}
            return call(client, EXECUTE_TOOL, request)

        # The gateway cannot approve: a pending approval does not run.
        assert await execute(args) == (True, {"error": "not_approved"})
        # Out of band, as `agentgate approvals approve` would.
        gw.world.gate.approve(approval_id, "kc-mgr-terrence")

        changed = email_args(OTHER_DIRECTORY_CUSTOMER, subject="zq-subject-marker")
        assert await execute(changed) == (True, {"error": "args_mismatch"})
        assert await execute(args, tool="parts_order") == (True, {"error": "invalid_args"})
        assert await execute(args, tool="shell_exec") == (True, {"error": "unknown_tool"})
        assert list_outbox(gw.world.store) == []

        error, content = await execute(args)
        assert not error
        assert content["status"] == "queued"
        [sent] = list_outbox(gw.world.store, KC)
        assert (sent.recipients, sent.subject, sent.run_id) == (
            (DIRECTORY_CUSTOMER,),
            "zq-subject-marker",
            ctx.run_id,
        )

        assert await execute(args) == (True, {"error": "already_executed"})
        assert len(list_outbox(gw.world.store)) == 1

    gw.run(ctx, scenario)
    assert last(gw, EventKind.EXECUTED).subject is not None
    dumped = "\n".join(e.model_dump_json() for e in gw.events)
    assert "zq-subject-marker" not in dumped
    assert DIRECTORY_CUSTOMER not in dumped


def test_another_runs_approval_is_not_found(gw: Gateway) -> None:
    first = gw.start("kc-mgr-marisol")
    args = email_args(DIRECTORY_CUSTOMER)
    pending: dict[str, str] = {}

    async def request(client: Client) -> None:
        _, pending["value"] = await call(client, "email_send", args)

    gw.run(first, request)
    approval_id = pending["value"]["approval_id"]
    gw.world.gate.approve(approval_id, "kc-owner-dale")
    second = gw.start("kc-mgr-marisol")

    async def steal(client: Client) -> None:
        request = {"approval_id": approval_id, "tool": "email_send", "args": args}
        assert await call(client, EXECUTE_TOOL, request) == (True, {"error": "not_found"})

    gw.run(second, steal)
    assert last(gw, EventKind.DENY).reason == "approval_run_mismatch"
    assert list_outbox(gw.world.store) == []


# Caps


def test_cap_reached_after_max_calls(gw: Gateway) -> None:
    ctx = gw.start("kc-staff-nadine")
    limit = gw.world.engines[KC].bundle.policy.spend.max_calls_per_run

    async def scenario(client: Client) -> None:
        read = {"document_id": "kc-service-menu"}
        for _ in range(limit):
            error, _ = await call(client, "docs_read", read)
            assert not error
        assert await call(client, "docs_read", read) == (
            True,
            {"error": "cap_reached", "cap": CapKind.CALLS.value},
        )

    gw.run(ctx, scenario)
    assert last(gw, EventKind.CAP_HIT).reason == CapKind.CALLS.value


# Same decisions as in-process

# (tool id, args): a mixed sequence, one call per model turn in-process.
SEQUENCE: list[tuple[str, dict[str, Any]]] = [
    ("docs.search", {"query": "furnace igniter service", "limit": 5}),
    ("docs.read", {"document_id": "kc-service-menu"}),
    ("docs.read", {"document_id": "kc-payroll-summary"}),
    ("docs.read", {"document_id": "jr-customer-list"}),
    ("docs.read", {"document_id": "kc-service-menu", "user_id": "kc-owner-dale"}),
    ("shell_exec", {"cmd": "ls"}),
    ("email.send", email_args(EXFIL)),
    ("email.send", email_args(DIRECTORY_CUSTOMER)),
    ("parts.order", {"part": "igniter", "quantity": 1, "amount_usd": "40.00"}),
]


def _comparable(events: list[AuditEvent]) -> list[tuple[object, ...]]:
    return [
        (e.kind, e.reason, e.customer_id, e.user_id, e.role, e.tool, e.action, e.rule_ids)
        for e in events
    ]


def test_same_audit_events_and_results_as_in_process(gw: Gateway) -> None:
    user = "kc-staff-nadine"
    ctx = gw.start(user)
    results: list[Any] = []

    async def scenario(client: Client) -> None:
        for tool, args in SEQUENCE:
            name = mcp_name(tool) if "." in tool else tool
            results.append(await call(client, name, args))

    gw.run(ctx, scenario)

    reference = _world()
    try:
        in_process = reference.gate.start_run(KC, user)
        transcript = run_agent(
            reference.gate,
            in_process,
            ScriptedModel("scripted", [[call] for call in SEQUENCE]),
            "Mixed calls.",
            specs=tool_specs(reference.engines[KC], default_tools()),
        )
        assert isinstance(reference.sink, MemorySink)
        expected_events = reference.sink.events
    finally:
        reference.close()

    assert transcript.stop_reason == "final"
    assert _comparable(gw.events) == _comparable(expected_events)
    outcomes = [outcome for step in transcript.steps for outcome in step.outcomes]
    assert len(outcomes) == len(results) == len(SEQUENCE)
    for (error, content), outcome in zip(results, outcomes, strict=True):
        assert error == (outcome.status in {"denied", "cap_reached"})
        if outcome.status == "pending_approval":
            assert content["status"] == "pending_approval"
            assert content["expires_at"] == outcome.content["expires_at"]
        else:
            assert content == outcome.content


# Failures


class FailingSink(MemorySink):
    """Refuses approval requests with an error whose message quotes the call."""

    def record(self, event: AuditEvent) -> None:
        if event.kind is EventKind.APPROVAL_REQUIRED:
            raise OSError(f"audit storage unavailable for {DIRECTORY_CUSTOMER}")
        super().record(event)


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_unexpected_gate_errors_reach_the_client_without_detail(mode: str) -> None:
    world = _world(FailingSink())
    gw = Gateway(world, mode)
    ctx = gw.start("kc-mgr-marisol")

    async def scenario(client: Client) -> None:
        with pytest.raises(MCPError) as caught:
            await client.call_tool("email_send", email_args(DIRECTORY_CUSTOMER))
        assert caught.value.message == "internal error"
        assert caught.value.data is None
        # The gateway keeps serving.
        error, _ = await call(client, "docs_read", {"document_id": "kc-service-menu"})
        assert not error

    try:
        gw.run(ctx, scenario)
        assert world.gate.approvals.list(KC) == []
    finally:
        world.close()


# The CLI over stdio

BANNER = re.compile(
    r"^gateway: run [0-9a-f-]{36} for kc-staff-nadine \(kettle-creek-hvac\), 6 tools$"
)


@pytest.mark.skipif(shutil.which("tee") is None, reason="needs sh and tee")
def test_cli_serves_over_stdio(tmp_path: Path) -> None:
    executable = shutil.which("agentgate", path=str(Path(sys.executable).parent))
    assert executable is not None
    db = tmp_path / "data" / "agentgate.db"
    audit_db = tmp_path / "data" / "audit.db"
    stdout_copy = tmp_path / "stdout.jsonl"
    stderr_path = tmp_path / "stderr.txt"
    command = [
        executable,
        "gateway",
        "serve",
        "--as",
        "kc-staff-nadine",
        "--db",
        str(db),
        "--audit-db",
        str(audit_db),
        "--policies",
        str(POLICIES),
        "--fixtures",
        str(FIXTURES),
    ]
    # tee keeps a copy of exactly what the gateway wrote to stdout.
    server = StdioServerParameters(
        command="sh", args=["-c", f'"$0" "$@" | tee "{stdout_copy}"', *command]
    )

    async def scenario() -> None:
        with stderr_path.open("w") as errlog:
            async with Client(stdio_client(server, errlog=errlog)) as client:
                names = {tool.name for tool in (await client.list_tools()).tools}
                assert names == KC_NAMES
                error, content = await call(client, "docs_read", {"document_id": "kc-service-menu"})
                assert not error and content["id"] == "kc-service-menu"
                restricted = await call(client, "docs_read", {"document_id": "kc-payroll-summary"})
                assert restricted == (True, {"error": "not_found"})

    anyio.run(scenario)

    frames = stdout_copy.read_text(encoding="utf-8").splitlines()
    assert frames
    for frame in frames:
        assert json.loads(frame)["jsonrpc"] == "2.0"
    stderr = stderr_path.read_text(encoding="utf-8").splitlines()
    assert stderr[0] == DEMO_IDENTITY_NOTE
    assert BANNER.match(stderr[1]), stderr
    assert not any("AGCANARY-" in line for line in stderr)

    connection = sqlite3.connect(db)
    try:
        runs = connection.execute("SELECT run_id, user_id FROM runs").fetchall()
    finally:
        connection.close()
    assert [user_id for _, user_id in runs] == ["kc-staff-nadine"]
    assert runs[0][0] in stderr[1]
    sink = PanoptesSink(audit_db, create=False)
    try:
        # run_started, then allow and executed for the read, then the deny.
        assert sink.count() == 4
        assert sink.verify()["valid"]
    finally:
        sink.close()
