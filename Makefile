.PHONY: install lint format type test policy redteam bench check demo

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

type:
	uv run mypy

test:
	uv run pytest

policy:
	uv run agentgate policy lint
	uv run agentgate policy test

# Both transports: in-process (reports/redteam.*) and through the MCP gateway
# (reports/redteam-mcp.*).
redteam:
	uv run agentgate redteam
	uv run agentgate redteam --transport mcp

bench:
	uv run agentgate bench

check: lint type test policy redteam

demo:
	uv run agentgate demo
