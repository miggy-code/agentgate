"""A model proxy over a real world, in front of a fake provider, for tests that speak HTTP.

`Provider` is a local HTTP server that records what it is sent and answers from a script.
`Rig` starts a ModelProxy over a file world whose upstream is that provider, and `post`
sends the proxy a request the way a client SDK would.
"""

import http.client
import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentgate.audit.base import AuditEvent, EventKind
from agentgate.audit.panoptes import read_events
from agentgate.model.admission import Caps
from agentgate.model.keys import ProviderKeys
from agentgate.model.proxy import ModelProxy, ProxyConfig
from agentgate.policy.schema import ModelProvider
from conftest import open_file_world

PROVIDER_KEY = "sk-provider-secret-0123456789"
STAFF = "kc-staff-nadine"

RATES: dict[str, Any] = {
    "version": "proxy-rates-1",
    "source": "unit test",
    "currency": "USD",
    "models": {
        "deepseek-flash": {
            "input_per_million": "0.14",
            "cached_input_per_million": "0.014",
            "output_per_million": "0.28",
        }
    },
}
BODY = {
    "model": "deepseek-flash",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "hi"}],
}
USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
OK_REPLY = {"id": "c1", "model": "deepseek-flash", "choices": [], "usage": USAGE}
# 100 * 0.14 + 50 * 0.28 = 28 per million tokens.
OK_COST = Decimal("0.000028")


@dataclass
class Seen:
    method: str
    target: str
    headers: list[tuple[str, str]]
    body: bytes


@dataclass
class Script:
    status: int = 200
    body: bytes = field(default_factory=lambda: json.dumps(OK_REPLY).encode())
    # Promise more bytes than are sent, then close: a reply that is cut short.
    truncate: bool = False


class Provider:
    def __init__(self) -> None:
        self.seen: list[Seen] = []
        self.script = Script()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.seen.append(Seen(self.command, self.path, list(self.headers.items()), body))
                script = outer.script
                self.send_response(script.status)
                self.send_header("Content-Type", "application/json")
                size = len(script.body) + (1000 if script.truncate else 0)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                self.wfile.write(script.body)
                self.wfile.flush()
                if script.truncate:
                    self.close_connection = True

            do_POST = do_GET = _answer

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@dataclass
class Reply:
    status: int
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)

    @property
    def error(self) -> str:
        return str(self.json().get("error"))


class Rig:
    def __init__(
        self,
        tmp_path: Path,
        provider: Provider,
        *,
        user: str = STAFF,
        caps: Caps | None = None,
        upstream_provider: ModelProvider = ModelProvider.OPENAI_COMPATIBLE,
    ) -> None:
        self.provider = provider
        self.world = open_file_world(tmp_path)
        self.audit_path = tmp_path / "data" / "audit.db"
        self.rates = tmp_path / "rates.json"
        self.rates.write_text(json.dumps(RATES))
        self.config = ProxyConfig(
            upstreams={upstream_provider: provider.origin},
            keys=ProviderKeys({upstream_provider: PROVIDER_KEY}),
            rate_table=self.rates,
            caps=caps,
            token_ttl=timedelta(hours=1),
            sweep_interval=timedelta(hours=1),
        )
        self.proxy = ModelProxy(self.world, user, self.config)
        self.port = self.proxy.start()
        assert self.proxy.token is not None
        self.token: str = self.proxy.token
        self.run = self.proxy.run

    def post(
        self,
        path: str = "/v1/chat/completions",
        body: dict[str, Any] | bytes | None = None,
        *,
        headers: dict[str, str] | None = None,
        token: str | bool | None = True,
        method: str = "POST",
    ) -> Reply:
        data = (
            json.dumps(BODY if body is None else body).encode()
            if not isinstance(body, bytes)
            else body
        )
        sent = {"Content-Type": "application/json"}
        if token is True:
            sent["Authorization"] = f"Bearer {self.token}"
        elif isinstance(token, str):
            sent["Authorization"] = f"Bearer {token}"
        sent.update(headers or {})
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request(method, path, body=data if method == "POST" else None, headers=sent)
            response = conn.getresponse()
            return Reply(response.status, response.read())
        finally:
            conn.close()

    def spent(self) -> Decimal:
        """What the run's settled reservations charged."""
        return sum(
            (
                Decimal(row["actual_usd"])
                for row in self.reservations()
                if row["status"] == "settled"
            ),
            Decimal(0),
        )

    def reservations(self) -> list[Any]:
        """The run's reservations once none is open: after_exchange runs after the client has
        its reply, so a test waits for it. A reservation still open after 5 seconds is
        returned open."""
        deadline = time.monotonic() + 5
        while True:
            rows = self._rows()
            if all(row["status"] != "open" for row in rows) or time.monotonic() > deadline:
                return rows
            time.sleep(0.02)

    def _rows(self) -> list[Any]:
        return self.world.store.connection.execute(
            "SELECT * FROM spend_reservations WHERE run_id = ?", (self.run.run_id,)
        ).fetchall()

    def events(self) -> list[AuditEvent]:
        """agentgate's own events, as the log stores them. The event that ends a call is
        recorded just after its reservation closes, so this waits for one per closed
        reservation (5 seconds at most)."""
        closed = sum(1 for row in self.reservations() if row["status"] != "open")
        deadline = time.monotonic() + 5
        while True:
            found = read_events(self.audit_path)
            ended = sum(1 for e in found if e.kind is EventKind.EXECUTED)
            if ended >= closed or time.monotonic() > deadline:
                return found
            time.sleep(0.02)

    def exchange_events(self) -> list[dict[str, Any]]:
        """Panoptes' events for the exchanges it observed, as stored."""
        db = self.world.sink.db  # type: ignore[attr-defined]
        return [dict(row) for row in db.iter_events(agent_type="model_client", order="rowid")]

    def close(self) -> None:
        try:
            self.proxy.stop()
        finally:
            self.world.close()


@contextmanager
def rig(tmp_path: Path, **kwargs: Any) -> Iterator[tuple[Rig, Provider]]:
    provider = Provider()
    made = Rig(tmp_path, provider, **kwargs)
    try:
        yield made, provider
    finally:
        made.close()
        provider.close()


Check = Callable[[Rig, Provider], None]
