.PHONY: install lint format type test policy check

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

check: lint type test policy
