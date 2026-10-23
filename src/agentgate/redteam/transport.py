"""How the red-team arena reaches the gate: in-process (`gate`) or through the MCP gateway
(`mcp`).

Under `mcp` every agent-side tool call a scenario makes goes through a real MCP client
session connected to a gateway (`agentgate.gateway.build_gateway`) bound to the run, over the
SDK's in-memory transport with the legacy initialize handshake. One gateway and one session
per run, opened on the run's first call over MCP and closed with the arena. The SDK is async
and the arena is not, so the sessions live on one event loop thread (an anyio blocking
portal) and every call waits for its answer, which keeps the order of calls, and so the
suite, deterministic.

The client only knows what the gateway sends it. `parse_result` turns a CallToolResult back
into the Outcome shape the scenarios assert on from the JSON text the gateway sent and
nothing else; it never looks at the gate. Two things the gateway does not send are filled in
from the client's own side: an Executed outcome's `tool` is the tool the client asked for,
and its `cost_usd` is `COST_NOT_SENT` (a quiet NaN), since a call's cost never reaches an MCP
client. Scenarios read spend from the ledger, not from outcomes.

Tool names: the client lists the gateway's tools once per session and calls a tool by the
gateway's naming rule (`docs.search` is `docs_search`) when that name is listed. A tool the
gateway does not list (another customer's tool, or one a compromised model made up) is sent
under the id the scenario used, as an attacker's client would send any name it likes; the
gateway passes an unlisted name to the gate unchanged.
"""

import json
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import Client, MCPError
from mcp.server import Server
from mcp.types import CallToolResult, TextContent
from pydantic import JsonValue

from agentgate.gate import CapReached, Denied, Executed, Outcome, PendingApproval
from agentgate.gateway.server import EXECUTE_TOOL, mcp_name
from agentgate.spend import CapKind

# What an Executed outcome parsed from an MCP result carries as its cost: never sent.
COST_NOT_SENT = Decimal("NaN")
# The connect mode of every session: the initialize handshake, as most MCP clients do it.
CLIENT_MODE = "legacy"
_PENDING_KEYS = frozenset({"status", "approval_id", "expires_at"})


class Transport(StrEnum):
    """How agent-side calls reach the gate."""

    GATE = "gate"
    MCP = "mcp"

    @property
    def description(self) -> str:
        if self is Transport.GATE:
            return "every call goes to the gate in-process"
        return "agent-side tool calls go through an MCP client session to the gateway"


class Direct(StrEnum):
    """Why an agent-side call did not go over MCP."""

    # The run used the gate transport: nothing goes over MCP.
    GATE_TRANSPORT = "gate_transport"
    # The context is not one the arena started: an MCP client cannot pass a context at all.
    FORGED_CONTEXT = "forged_context"
    # The scenario chose an action the tool does not have; over MCP the action comes from
    # the gateway's own tool list, so the call cannot be expressed.
    ACTION_OVERRIDE = "action_override"
    # The gateway refuses to launch for this run (its customer has no policy).
    GATEWAY_REFUSED = "gateway_refused"


class GatewayProtocolError(RuntimeError):
    """The gateway answered with something that is not one of its outcomes."""


class GatewayCallError(RuntimeError):
    """The gateway answered a call with an MCP error instead of a result.

    Only the error code is kept: the gateway sends a generic message, and an exception
    message here must not quote arguments either.
    """

    def __init__(self, code: int) -> None:
        super().__init__(f"gateway returned MCP error {code}")
        self.code = code


@dataclass(frozen=True)
class McpReply:
    """One answer from the gateway: the outcome it means, and the text the client got."""

    outcome: Outcome
    text: str


def parse_result(tool: str, result: CallToolResult) -> McpReply:
    """The Outcome a gateway result stands for, from what the client received only.

    Raises GatewayProtocolError for anything the gateway would not send: other content
    than one text block, text that is not a JSON object, or an error without a reason.
    """
    if len(result.content) != 1 or not isinstance(result.content[0], TextContent):
        raise GatewayProtocolError("expected exactly one text block")
    text = result.content[0].text
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        raise GatewayProtocolError("result text is not JSON") from None
    if not isinstance(payload, dict):
        raise GatewayProtocolError("result is not a JSON object")
    if result.is_error:
        return McpReply(_error_outcome(payload), text)
    if payload.get("status") == "pending_approval" and set(payload) == _PENDING_KEYS:
        approval_id = payload["approval_id"]
        expires_at = payload["expires_at"]
        if not isinstance(approval_id, str) or not isinstance(expires_at, str):
            raise GatewayProtocolError("malformed pending approval")
        try:
            expires = datetime.fromisoformat(expires_at)
        except ValueError:
            raise GatewayProtocolError("malformed approval expiry") from None
        return McpReply(PendingApproval(approval_id=approval_id, expires_at=expires), text)
    content: dict[str, JsonValue] = payload
    return McpReply(Executed(tool=tool, content=content, cost_usd=COST_NOT_SENT), text)


def _error_outcome(payload: dict[str, Any]) -> Outcome:
    error = payload.get("error")
    if not isinstance(error, str):
        raise GatewayProtocolError("error result without a reason")
    if error == "cap_reached" and set(payload) == {"error", "cap"}:
        try:
            return CapReached(CapKind(payload["cap"]))
        except ValueError:
            raise GatewayProtocolError("unknown cap") from None
    if set(payload) != {"error"}:
        raise GatewayProtocolError("error result with unexpected keys")
    return Denied(error)


class GatewaySession:
    """A client connected to one gateway, driven from synchronous code through a portal."""

    def __init__(self, portal: BlockingPortal, server: Server[Any]) -> None:
        self._portal = portal
        self._context: AbstractContextManager[Client] = portal.wrap_async_context_manager(
            Client(server, mode=CLIENT_MODE)
        )
        self._client = self._context.__enter__()
        self._open = True
        try:
            listed = portal.call(self._client.list_tools)
        except BaseException:
            self.close()
            raise
        self.names = frozenset(tool.name for tool in listed.tools)

    def name_for(self, tool: str) -> str:
        """The name the client calls a tool by: the listed MCP name, else the id as given."""
        try:
            name = mcp_name(tool)
        except ValueError:
            return tool
        return name if name in self.names and name != EXECUTE_TOOL else tool

    def call(self, tool: str, args: dict[str, JsonValue]) -> McpReply:
        return parse_result(tool, self._call(self.name_for(tool), args))

    def execute(self, approval_id: str, tool: str, args: dict[str, JsonValue]) -> McpReply:
        request: dict[str, JsonValue] = {
            "approval_id": approval_id,
            "tool": self.name_for(tool),
            "args": args,
        }
        return parse_result(tool, self._call(EXECUTE_TOOL, request))

    def _call(self, name: str, args: dict[str, JsonValue]) -> CallToolResult:
        try:
            result: CallToolResult = self._portal.call(self._client.call_tool, name, args)
        except MCPError as exc:
            raise GatewayCallError(exc.code) from None
        return result

    def close(self) -> None:
        if self._open:
            self._open = False
            self._context.__exit__(None, None, None)


class GatewaySessions:
    """The MCP sessions of one arena, one per run, on one event loop thread."""

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._portal: BlockingPortal | None = None
        self._sessions: dict[str, GatewaySession] = {}

    def get(self, run_id: str) -> GatewaySession | None:
        return self._sessions.get(run_id)

    def open(self, run_id: str, server: Server[Any]) -> GatewaySession:
        if self._portal is None:
            self._portal = self._stack.enter_context(start_blocking_portal())
        session = GatewaySession(self._portal, server)
        self._sessions[run_id] = session
        return session

    def reset(self) -> None:
        """Close every session, as restarting the gateways would; the portal stays."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        _close_all(sessions)

    def close(self) -> None:
        try:
            self.reset()
        finally:
            self._portal = None
            self._stack.close()


def _close_all(sessions: list[GatewaySession]) -> None:
    error: BaseException | None = None
    for session in sessions:
        try:
            session.close()
        except BaseException as exc:
            error = error or exc
    if error is not None:
        raise error
