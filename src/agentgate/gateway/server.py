"""The MCP gateway: one process that owns the tools and offers them to any MCP client.

In-process middleware only protects tools that are reachable solely through the gate. The
gateway moves the tools, the database and their side effects into their own process, and the
agent gets nothing but an MCP connection. Every tool call it makes goes through the same
`Gate.call_tool` the reference loop uses, and nothing else in the gateway can reach a tool.

One gateway is bound to one run. The gateway starts that run itself, for an identity given
at launch, before it serves anything; the MCP client never names a user, role or customer.
No tool takes an identity argument, and the strict args models already refuse unknown keys
such as `user_id` or `role` as `invalid_args`, so a client cannot claim to be someone else.

`list_tools` offers the run customer's tools, as the reference loop's `tool_specs` builds
them, under MCP names from the same rule the Anthropic adapter uses (`docs.search` becomes
`docs_search`): dots are legal MCP names, but most model providers a client forwards them to
refuse them. One gateway tool, `approvals_execute`, runs an approved call through
`gate.execute_approved`, which checks the run, the args hash, the expiry and the current
policy again. It cannot approve anything: approvals are decided out of band, with
`agentgate approvals approve`.

Every refusal a client gets is the gate's, on record with the bound run's identity. The
gateway refuses nothing itself: an approval id or tool name of any length or shape goes to
`execute_approved`, which looks it up and audits `approval_not_found` and the rest, and an
approvals_execute request whose wrapper cannot be read at all (a missing field, a wrong type,
an extra key such as `user_id`) goes to `gate.refuse`, which checks the run and user like any
call and records the `invalid_args` denial, naming only the wrapper's own fields. Two
failures reach a client without a gate record: an exception inside the gate (the client gets
a generic internal error; when it is the audit sink that failed, nothing can be recorded),
and requests the MCP SDK rejects before this server's handlers run (a malformed JSON-RPC
request or params that are not an object).

`call_tool` maps the name back to a tool id and takes the action from that tool's spec,
never from the client. A name that maps to no listed tool goes to the gate unchanged, so the
gate denies it as an unknown tool and the denial is audited. The result is the same
model-facing outcome the reference loop shows its model, as JSON text: the tool's content,
`{"error": reason}`, a pending approval's id and expiry, or the cap that was reached.

The gate is synchronous and the SDK is async, so each call runs in a worker thread. Calls are
serialized, one at a time per gateway, so a client that sends calls concurrently gets the
ordering the reference loop has: a call is decided only after the run's max label reflects
every earlier result.

Model spend cannot be metered here. The gateway never sees the model's calls, only tool
calls, so `charge_model` is never made and the per-run and per-customer-day USD caps count
tool costs only. Tool costs and `max_calls_per_run` still apply to every tool call.
"""

import json
import logging
import re
from collections.abc import Callable, Mapping
from typing import Any

import anyio
import anyio.to_thread
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INTERNAL_ERROR,
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

import agentgate
from agentgate.agent.loop import tool_specs
from agentgate.agent.model import ToolSpec
from agentgate.context import RunContext
from agentgate.gate import Denied, Executed, Outcome, PendingApproval
from agentgate.models import Action
from agentgate.tools.builtin import default_tools
from agentgate.world import World

SERVER_NAME = "agentgate"
EXECUTE_TOOL = "approvals_execute"
EXECUTE_DESCRIPTION = (
    "Run a call a person has already approved, with exactly the tool and args that were"
    " approved. Approvals are decided outside this connection; this cannot approve anything."
)

# The Anthropic adapter's rule for provider tool names, so a client that forwards these
# names to a model provider needs no mapping of its own.
_MCP_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

logger = logging.getLogger(__name__)


class _ExecuteArgs(BaseModel):
    """The args of approvals_execute: strict, like every built-in tool's."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    # No length or shape limits here: the gate checks the id and the tool, and audits them.
    approval_id: str
    tool: str
    args: dict[str, JsonValue]


_EXECUTE_FIELDS = frozenset(_ExecuteArgs.model_fields)


_EXECUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approval_id": {
            "type": "string",
            "description": "The approval id a pending call returned.",
        },
        "tool": {
            "type": "string",
            "description": "The name of the approved tool, as this server lists it.",
        },
        "args": {
            "type": "object",
            "description": "Exactly the args of the approved call.",
        },
    },
    "required": ["approval_id", "tool", "args"],
    "additionalProperties": False,
}


def mcp_name(tool: str) -> str:
    """The MCP name for a tool id: `docs.search` becomes `docs_search`."""
    name = _INVALID_NAME_CHARS.sub("_", tool)[:64]
    if not _MCP_NAME.fullmatch(name):
        raise ValueError(f"tool {tool!r} cannot be mapped to an MCP tool name")
    return name


def gateway_tools(world: World, ctx: RunContext) -> list[Tool]:
    """The tools the gateway lists for this run: its customer's tools and approvals_execute."""
    return [*(_listed(spec) for spec in _specs(world, ctx)), _execute_tool()]


def build_gateway(world: World, ctx: RunContext) -> Server[Any]:
    """An MCP server bound to one run the gateway started; every tool call goes to the gate."""
    specs = _specs(world, ctx)
    by_tool = {spec.tool: spec for spec in specs}
    by_name: dict[str, str] = {}
    for spec in specs:
        name = mcp_name(spec.tool)
        if name == EXECUTE_TOOL or by_name.setdefault(name, spec.tool) != spec.tool:
            raise ValueError(f"tool {spec.tool!r} maps to an MCP name already in use")
    gate = world.gate
    # One gate call at a time, so concurrent client calls are decided in order.
    limiter = anyio.CapacityLimiter(1)

    def call(tool_name: str, args: Mapping[str, JsonValue]) -> Outcome:
        tool = by_name.get(tool_name, tool_name)
        spec = by_tool.get(tool)
        # An unknown tool still goes to the gate, so the deny is decided and recorded there.
        action = Action.READ if spec is None else spec.action
        return gate.call_tool(ctx, tool, action, args)

    def execute(args: Mapping[str, JsonValue]) -> Outcome:
        try:
            checked = _ExecuteArgs.model_validate(args, strict=True)
        except ValidationError as exc:
            # The wrapper cannot be read, so there is no call to hand execute_approved; the
            # gate still refuses it, checking the run and recording the denial.
            named = args.get("tool")
            tool = by_name.get(named, named) if isinstance(named, str) else None
            return gate.refuse(ctx, tool=tool, details=_wrapper_errors(exc))
        tool = by_name.get(checked.tool, checked.tool)
        spec = by_tool.get(tool)
        action = Action.READ if spec is None else spec.action
        return gate.execute_approved(ctx, checked.approval_id, tool, action, checked.args)

    async def run(work: Callable[[], Outcome]) -> CallToolResult:
        try:
            outcome = await anyio.to_thread.run_sync(work, limiter=limiter)
        except Exception as exc:
            # The exception type only, on stderr, since messages can quote arguments. The
            # client gets a generic error: in-process, this would reach the host, not the model.
            logger.error("gateway call failed: %s", type(exc).__name__)
            raise MCPError(code=INTERNAL_ERROR, message="internal error") from None
        return _result(outcome)

    async def on_list_tools(
        _request: ServerRequestContext[Any], _params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        # Built afresh per request, so nothing the SDK does to a result changes the next one.
        return ListToolsResult(tools=[*(_listed(spec) for spec in specs), _execute_tool()])

    async def on_call_tool(
        _request: ServerRequestContext[Any], params: CallToolRequestParams
    ) -> CallToolResult:
        args: dict[str, Any] = {} if params.arguments is None else params.arguments
        if params.name == EXECUTE_TOOL:
            return await run(lambda: execute(args))
        name = params.name
        return await run(lambda: call(name, args))

    return Server(
        SERVER_NAME,
        version=agentgate.__version__,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def serve_stdio(world: World, ctx: RunContext) -> None:
    """Serve the gateway for this run over stdin and stdout until the client disconnects.

    While it serves, the SDK points file descriptor 1 at stderr, so stray output from any
    library cannot corrupt the MCP stream on stdout.
    """
    server = build_gateway(world, ctx)

    async def serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(serve)


def _specs(world: World, ctx: RunContext) -> list[ToolSpec]:
    engine = world.engines.get(ctx.customer_id)
    if engine is None:
        raise ValueError(f"customer {ctx.customer_id!r} has no policy")
    # The same adapters open_world's gate runs, so the list matches what the gate accepts.
    return tool_specs(engine, default_tools())


def _wrapper_errors(exc: ValidationError) -> dict[str, JsonValue]:
    """Why an approvals_execute wrapper was refused: error types and the wrapper's own field
    names only, never a key the client chose or any value, as the gate reports tool args."""
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    fields = sorted(
        {str(e["loc"][0]) for e in errors if e["loc"] and e["loc"][0] in _EXECUTE_FIELDS}
    )
    types = sorted({e["type"] for e in errors})
    return {"error_types": list(types), "fields": list(fields)}


def _listed(spec: ToolSpec) -> Tool:
    return Tool(
        name=mcp_name(spec.tool),
        description=spec.description,
        input_schema=json.loads(json.dumps(spec.input_schema)),
    )


def _execute_tool() -> Tool:
    return Tool(
        name=EXECUTE_TOOL,
        description=EXECUTE_DESCRIPTION,
        input_schema=json.loads(json.dumps(_EXECUTE_SCHEMA)),
    )


def _result(outcome: Outcome) -> CallToolResult:
    """The model-facing outcome as JSON text: what the reference loop shows its model."""
    payload: dict[str, JsonValue]
    is_error = False
    if isinstance(outcome, Executed):
        payload = outcome.content
    elif isinstance(outcome, Denied):
        payload = {"error": outcome.reason}
        is_error = True
    elif isinstance(outcome, PendingApproval):
        payload = {
            "status": "pending_approval",
            "approval_id": outcome.approval_id,
            "expires_at": outcome.expires_at.isoformat(),
        }
    else:
        payload = {"error": "cap_reached", "cap": outcome.cap.value}
        is_error = True
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return CallToolResult(content=[TextContent(text=text)], is_error=is_error)
