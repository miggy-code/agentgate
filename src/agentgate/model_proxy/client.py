"""A client for the multi-run model proxy's control socket (`agentgate model-proxy server`).

A host starts a run for a user with `start_run`, hands the returned token to the agent's
model client (in place of a provider key, sent to the server's `model_url`), and ends the run
with `end_run`. The host needs no HTTP code of its own.

Before each request the client checks that the socket path is the server's: a socket, not a
symlink, owned by this user, in a directory owned by this user that no one else can enter. A
path planted by someone else is refused rather than followed, so a host cannot be pointed at
a server that is not the one it started.
"""

import os
import stat
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from urllib.parse import quote

import httpx


class ControlError(Exception):
    """The server refused a request, or could not be reached. `reason` is its reason code."""

    def __init__(self, reason: str, status: int | None = None) -> None:
        super().__init__(reason if status is None else f"{status} {reason}")
        self.reason = reason
        self.status = status


@dataclass(frozen=True)
class RunStart:
    """A started run, as the server reported it."""

    run_id: str
    customer_id: str
    user_id: str
    # Shown once: the server keeps only its hash.
    token: str
    expires_at: datetime
    # The run's caps in effect.
    max_usd: Decimal
    max_calls: int
    per_customer_day_usd: Decimal

    def __repr__(self) -> str:
        return (
            f"RunStart(run_id={self.run_id!r}, customer_id={self.customer_id!r},"
            f" user_id={self.user_id!r}, expires_at={self.expires_at!r})"
        )


class ControlClient:
    def __init__(self, socket_path: str | Path, *, timeout: float = 30.0) -> None:
        self.socket_path = Path(socket_path)
        self._client = httpx.Client(
            transport=httpx.HTTPTransport(uds=str(self.socket_path)),
            base_url="http://agentgate-control",
            timeout=timeout,
        )

    def start_run(
        self,
        customer_id: str,
        user_id: str,
        *,
        ttl_seconds: int | None = None,
        max_usd: Decimal | str | None = None,
        max_calls: int | None = None,
    ) -> RunStart:
        body: dict[str, Any] = {"customer_id": customer_id, "user_id": user_id}
        return self._start(body, ttl_seconds, max_usd, max_calls)

    def start_verified_run(
        self,
        assertion: str,
        *,
        ttl_seconds: int | None = None,
        max_usd: Decimal | str | None = None,
        max_calls: int | None = None,
    ) -> RunStart:
        """Start a run for the person a signed identity assertion names (a server started
        with `--identity-config`). The server verifies it and starts the run in the issuer's
        customer. A refusal is a ControlError with the reason code (`expired`, `replayed`,
        ...; `issuer_unavailable` with status 503). The assertion is never in an error."""
        return self._start({"assertion": assertion}, ttl_seconds, max_usd, max_calls)

    def _start(
        self,
        body: dict[str, Any],
        ttl_seconds: int | None,
        max_usd: Decimal | str | None,
        max_calls: int | None,
    ) -> RunStart:
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        if max_usd is not None:
            body["max_usd"] = str(max_usd)
        if max_calls is not None:
            body["max_calls"] = max_calls
        data = self._request("POST", "/runs", body, expect=201)
        try:
            return RunStart(
                run_id=str(data["run_id"]),
                customer_id=str(data["customer_id"]),
                user_id=str(data["user_id"]),
                token=str(data["token"]),
                expires_at=datetime.fromisoformat(str(data["expires_at"])),
                max_usd=Decimal(str(data["max_usd"])),
                max_calls=int(data["max_calls"]),
                per_customer_day_usd=Decimal(str(data["per_customer_day_usd"])),
            )
        except (KeyError, TypeError, ValueError, ArithmeticError):
            raise ControlError("invalid_response") from None

    def end_run(self, run_id: str) -> None:
        """End a run the server started. Ending a run already ended is not an error."""
        self._request("POST", f"/runs/{quote(run_id, safe='')}/end", None, expect=200)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", None, expect=200)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None, *, expect: int
    ) -> dict[str, Any]:
        check_control_socket(self.socket_path)
        try:
            response = self._client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise ControlError("unreachable") from exc
        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise ControlError("invalid_response", response.status_code)
        if response.status_code != expect:
            raise ControlError(str(data.get("error", "refused")), response.status_code)
        return data


def check_control_socket(path: Path) -> None:
    """Refuse a socket path that is not a private control socket: the path must be a socket
    (not a symlink) owned by this user, in a directory (not a symlink) owned by this user
    with no access for group or others."""
    uid = os.getuid()
    try:
        directory = os.lstat(path.parent)
        sock = os.lstat(path)
    except OSError:
        raise ControlError("unsafe_socket") from None
    if (
        not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != uid
        or directory.st_mode & 0o077
        or not stat.S_ISSOCK(sock.st_mode)
        or sock.st_uid != uid
    ):
        raise ControlError("unsafe_socket")
