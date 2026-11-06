"""A minimal customer and models-only policy for one experiment, as `init-experiment` writes.

An experiment needs one thing from agentgate: "this user may call these models, under these
spend and call caps". This writes exactly that, a one-user fixture and a policy with only
`spend` and `models`, so every tool call is denied and every model not listed is denied.
Every file is validated with the loaders' own models before anything is written.
"""

import contextlib
import os
import stat
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from agentgate.models import Customer, User
from agentgate.policy.schema import ModelProvider, Policy

DEFAULT_CUSTOMER = "lab"
DEFAULT_USER = "lab-runner"
DEFAULT_MAX_OUTPUT_TOKENS = 8000
DEFAULT_MAX_REQUEST_BYTES = 4_000_000


class ExperimentError(ValueError):
    """The experiment cannot be written as asked."""


@dataclass(frozen=True)
class ExperimentModel:
    provider: ModelProvider
    model: str

    @classmethod
    def parse(cls, text: str) -> "ExperimentModel":
        """PROVIDER:NAME. Only the first colon splits, since model names may contain one."""
        name, _, model = text.partition(":")
        try:
            provider = ModelProvider(name)
        except ValueError:
            known = ", ".join(p.value for p in ModelProvider)
            raise ExperimentError(f"unknown provider {name!r} (one of {known})") from None
        if not model:
            raise ExperimentError(f"--model is PROVIDER:NAME, got {text!r}")
        return cls(provider, model)

    @property
    def entry_id(self) -> str:
        """The policy entry id: the model name as a slug ("models/gemini-2.5" ->
        "models-gemini-2-5")."""
        slug = "".join(c if c.isalnum() else "-" for c in self.model.lower()).strip("-")
        return "-".join(part for part in slug.split("-") if part)[:64] or "model"


@dataclass(frozen=True)
class ExperimentFiles:
    fixtures: Path
    policies: Path
    customer: str
    user: str


def write_experiment(
    root: Path,
    models: list[ExperimentModel],
    *,
    max_usd: Decimal,
    max_calls: int,
    day_usd: Decimal | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    customer: str = DEFAULT_CUSTOMER,
    user: str = DEFAULT_USER,
    allow_server_features: bool = False,
) -> ExperimentFiles:
    """Write root/fixtures/<customer>/ and root/policies/<customer>/, never over a file."""
    if not models:
        raise ExperimentError("name at least one --model")
    ids = [model.entry_id for model in models]
    if len(set(ids)) != len(ids):
        raise ExperimentError("two --model names map to the same policy entry id")
    fixtures, policies = root / "fixtures", root / "policies"
    customer_dir, policy_dir = fixtures / customer, policies / customer
    files: dict[Path, Any] = {
        customer_dir / "customer.yaml": {
            "id": customer,
            "name": "Experiment",
            "domains": [f"{customer}.example"],
            "timezone": "UTC",
        },
        customer_dir / "users.yaml": [
            {
                "id": user,
                "name": "Experiment Runner",
                "email": f"runner@{customer}.example",
                "role": "owner",
                "status": "active",
            }
        ],
        policy_dir / "policy.yaml": {
            "apiVersion": "agentgate/v1",
            "customer": customer,
            "labels": ["public"],
            "default": "deny",
            "spend": {
                "per_run_usd": _plain(max_usd),
                "per_customer_day_usd": _plain(max_usd if day_usd is None else day_usd),
                "max_calls_per_run": max_calls,
            },
            "models": {
                "allow": [
                    {
                        "id": entry_id,
                        "provider": model.provider.value,
                        "model": model.model,
                        "roles": ["owner"],
                        "max_output_tokens": max_output_tokens,
                        "max_request_bytes": max_request_bytes,
                    }
                    | ({"allow_server_features": True} if allow_server_features else {})
                    for entry_id, model in zip(ids, models, strict=True)
                ],
                "reserve": "computed",
            },
        },
    }
    _check(Customer, files[customer_dir / "customer.yaml"])
    _check(User, files[customer_dir / "users.yaml"][0] | {"customer_id": customer})
    _check(Policy, files[policy_dir / "policy.yaml"])
    existing = [str(path) for path in files if path.exists() or path.is_symlink()]
    if existing:
        raise ExperimentError(f"already exists: {', '.join(existing)}")
    # Each created file stays open until the end: an open file's inode cannot be reused, so
    # the identity check in a rollback cannot mistake a replacement for ours.
    created: list[tuple[Path, int]] = []
    try:
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            created.append((path, _create(path, yaml.safe_dump(data, sort_keys=False))))
    except FileExistsError as exc:
        # Something appeared after the check above: undo only what this call created,
        # never touch the newcomer, nor a file that replaced one of ours meanwhile.
        for path, fd in created:
            unlink_if_same(path, fd)
        raise ExperimentError(f"already exists: {exc.filename}") from None
    finally:
        for _, fd in created:
            os.close(fd)
    return ExperimentFiles(fixtures=fixtures, policies=policies, customer=customer, user=user)


def _create(path: Path, text: str) -> int:
    """Create path exclusively and never through a symlink; return it still open.

    No existing file is ever truncated or redirected. The caller closes the descriptor."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o644)
    try:
        data = text.encode("utf-8")
        while data:
            data = data[os.write(fd, data) :]
    except BaseException:
        os.close(fd)
        raise
    return fd


def unlink_if_same(path: Path, fd: int) -> bool:
    """Remove path only if it is still the file open as fd.

    The caller must hold fd open from creation until this call: an open file's inode cannot
    be freed, so a file that replaced it always has a different (device, inode). A file that
    replaced it, or a symlink, is left alone. POSIX has no unlink-by-identity, so a
    replacement in the instant between the check and the unlink is still possible; this
    narrows that window to a few microseconds instead of the whole run."""
    with contextlib.suppress(OSError):
        ours, info = os.fstat(fd), os.lstat(path)
        if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == (ours.st_dev, ours.st_ino):
            path.unlink()
            return True
    return False


def _check(model: type[BaseModel], data: object) -> None:
    try:
        model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            ".".join(str(part) for part in err["loc"])
            + ": "
            + err["msg"].removeprefix("Value error, ")
            for err in exc.errors(include_url=False, include_input=False)
        )
        raise ExperimentError(problems) from None


def _plain(value: Decimal) -> str:
    return "0" if value.is_zero() else format(value.normalize(), "f")
