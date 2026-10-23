"""The MCP gateway: the customer's tools behind an MCP server, every call through the gate."""

from agentgate.gateway.server import (
    EXECUTE_TOOL,
    build_gateway,
    gateway_tools,
    mcp_name,
    serve_stdio,
)

__all__ = ["EXECUTE_TOOL", "build_gateway", "gateway_tools", "mcp_name", "serve_stdio"]
