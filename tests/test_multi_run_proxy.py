"""The multi-run model proxy (`agentgate model-proxy server`) over real HTTP and a real Unix
control socket, in front of a fake provider. Each test is one boundary of serving many runs
from one process; the mutation tests at the end remove the protection behind the two the
spec names (the served-runs check and per-run caps) and require their tests to fail.
"""

import contextlib
import http.client
import json
import os
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from agentgate.audit.base import AuditEvent, EventKind, OperatorAction
from agentgate.audit.panoptes import read_events, verify_log
from agentgate.model.admission import Caps
from agentgate.model.keys import ProviderKeys
from agentgate.model.tokens import TokenStore
from agentgate.model_proxy.client import ControlClient, ControlError
from agentgate.model_proxy.proxy import ProxyConfig, ProxySetupError
from agentgate.model_proxy.server import MultiRunProxy, ServedRuns, ServerConfig
from agentgate.policy.schema import ModelProvider
from agentgate.store import Store
from agentgate.world import World, open_world
from conftest import FIXTURES, KC, POLICIES, mutant, open_file_world
from proxyharness import BODY, PROVIDER_KEY, RATES, Provider, Reply
from test_cli_ops import _cli, _error

OWNER = "kc-owner-dale"
STAFF = "kc-staff-nadine"
COLBY = "kc-staff-colby"
JR = "juniper-row-apothecary"
JR_OWNER = "jr-owner-marisol"
UPSTREAM = ModelProvider.OPENAI_COMPATIBLE


def _short_dir() -> Path:
    """A new directory with a short path: a Unix socket path is limited to about 100 bytes,
    and pytest's tmp_path on macOS is longer than that."""
    return Path(tempfile.mkdtemp(prefix="ag-"))


class Served:
    """A MultiRunProxy over a file world, its fake provider and a control client."""

    def __init__(
        self,
        tmp_path: Path,
        provider: Provider,
        *,
        fixtures: Path = FIXTURES,
        control_socket: Path | None = None,
        workers: int = 16,
    ) -> None:
        self.tmp_path = tmp_path
        self.fixtures = fixtures
        self.provider = provider
        self.audit_path = tmp_path / "data" / "audit.db"
        self.rates = tmp_path / "rates.json"
        self.rates.write_text(json.dumps(RATES))
        self.world = self.open_world()
        self.config = ServerConfig(
            upstreams={UPSTREAM: provider.origin},
            keys=ProviderKeys({UPSTREAM: PROVIDER_KEY}),
            rate_table=self.rates,
            control_socket=control_socket,
            control_workers=workers,
            sweep_interval=timedelta(hours=1),
        )
        self.proxy = MultiRunProxy(self.world, self.config)
        self.port, self.socket = self.proxy.start()
        self.client = ControlClient(self.socket)

    def open_world(self) -> World:
        return open_world(
            db_path=self.tmp_path / "data" / "agentgate.db",
            audit_path=self.audit_path,
            policies=POLICIES,
            fixtures=self.fixtures,
        )

    def post(
        self,
        token: str,
        body: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        conn: http.client.HTTPConnection | None = None,
    ) -> Reply:
        sent = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        sent.update(headers or {})
        conn = conn or self.connect()
        try:
            conn.request(
                "POST", "/v1/chat/completions", body=json.dumps(body or BODY), headers=sent
            )
            response = conn.getresponse()
            return Reply(response.status, response.read())
        finally:
            conn.close()

    def connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)

    def reservations(self, run_id: str) -> list[Any]:
        """The run's reservations once none is open (after_exchange runs after the reply)."""
        deadline = time.monotonic() + 30
        while True:
            rows = self.world.store.connection.execute(
                "SELECT * FROM spend_reservations WHERE run_id = ?", (run_id,)
            ).fetchall()
            if all(row["status"] != "open" for row in rows) or time.monotonic() > deadline:
                return rows
            time.sleep(0.02)

    def events(self) -> list[AuditEvent]:
        return read_events(self.audit_path)

    def close(self) -> None:
        try:
            self.client.close()
            self.proxy.stop()
        finally:
            self.world.close()


@contextmanager
def served(tmp_path: Path, **kwargs: Any) -> Iterator[Served]:
    provider = Provider()
    made = Served(tmp_path, provider, **kwargs)
    try:
        yield made
    finally:
        made.close()
        provider.close()


def _token_refusals(events: list[AuditEvent], run_id: str) -> list[str]:
    return [
        str(e.details.get("refused_because"))
        for e in events
        if e.kind is EventKind.DENY and e.reason == "token_refused" and e.run_id == run_id
    ]


def _run_ended(events: list[AuditEvent]) -> dict[str, str]:
    return {
        str(e.details["run_id"]): str(e.details["reason"])
        for e in events
        if e.kind is EventKind.OPERATOR_ACTION and e.reason == OperatorAction.RUN_ENDED.value
    }


# Serving many runs


def test_two_users_runs_are_served_at_once_each_token_as_its_own_run(tmp_path: Path) -> None:
    with served(tmp_path) as made:
        owner = made.client.start_run(KC, OWNER)
        staff = made.client.start_run(KC, STAFF)
        assert (owner.user_id, staff.user_id) == (OWNER, STAFF)
        assert owner.run_id != staff.run_id and owner.token != staff.token

        # The correlation id is the client's claim; the token alone says which run it is.
        assert made.post(owner.token, headers={"X-Correlation-Id": staff.run_id}).status == 200
        assert made.post(staff.token).status == 200
        assert made.post(staff.token).status == 200

        assert len(made.reservations(owner.run_id)) == 1
        assert len(made.reservations(staff.run_id)) == 2
        assert made.client.health()["runs_live"] == 2
        sent = {name.lower(): value for name, value in made.provider.seen[0].headers}
        assert sent["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_a_token_from_a_serve_process_on_the_same_world_is_refused_not_served(
    tmp_path: Path,
) -> None:
    with served(tmp_path) as made:
        single = made.open_world()
        serve = _serve_proxy(single, made)
        try:
            serve_token = serve.token
            assert serve_token is not None
            reply = made.post(serve_token)
            assert reply.status == 401 and reply.error == "token_refused"
            assert made.provider.seen == []
            assert _token_refusals(made.events(), serve.run.run_id) == ["not_served"]
        finally:
            serve.stop()
            single.close()


def _serve_proxy(world: World, made: Served) -> Any:
    from agentgate.model_proxy.proxy import ModelProxy

    proxy = ModelProxy(
        world,
        STAFF,
        ProxyConfig(
            upstreams={UPSTREAM: made.provider.origin},
            keys=ProviderKeys({UPSTREAM: PROVIDER_KEY}),
            rate_table=made.rates,
            sweep_interval=timedelta(hours=1),
        ),
    )
    proxy.start()
    return proxy


def test_an_ended_runs_token_is_refused_and_ending_is_idempotent(tmp_path: Path) -> None:
    with served(tmp_path) as made:
        run = made.client.start_run(KC, STAFF)
        made.client.end_run(run.run_id)
        made.client.end_run(run.run_id)

        reply = made.post(run.token)
        assert reply.status == 401 and reply.error == "token_refused"
        assert made.provider.seen == []
        events = made.events()
        assert _run_ended(events) == {run.run_id: "requested"}
        assert _token_refusals(events, run.run_id) == ["revoked"]
        assert made.client.health()["runs_live"] == 0


def test_a_run_this_server_did_not_start_cannot_be_ended_through_it(tmp_path: Path) -> None:
    with served(tmp_path) as made:
        single = made.open_world()
        serve = _serve_proxy(single, made)
        try:
            for run_id in (serve.run.run_id, "no-such-run"):
                with pytest.raises(ControlError) as refused:
                    made.client.end_run(run_id)
                assert (refused.value.status, refused.value.reason) == (404, "unknown_run")
            serve_token = serve.token
            assert serve_token is not None
            assert TokenStore(single.store, single.clock).resolve(serve_token)
        finally:
            serve.stop()
            single.close()


def test_a_suspended_users_live_token_is_refused(tmp_path: Path) -> None:
    fixtures = Path(shutil.copytree(FIXTURES, tmp_path / "fixtures"))
    with served(tmp_path, fixtures=fixtures) as made:
        run = made.client.start_run(KC, COLBY)
        assert made.post(run.token).status == 200

        users = fixtures / KC / "users.yaml"
        text = users.read_text()
        colby = text.index(f"id: {COLBY}")
        status = text.index("status: active", colby)
        users.write_text(text[:status] + "status: suspended" + text[status + 14 :])
        made.open_world().close()  # another process syncs the fixtures

        reply = made.post(run.token)
        assert reply.status == 401 and reply.error == "token_refused"
        assert len(made.provider.seen) == 1


def test_run_start_refuses_another_customers_user_unknown_users_and_suspended_ones(
    tmp_path: Path,
) -> None:
    with served(tmp_path) as made:
        made.world.store.suspend_user(COLBY)
        for customer, user, reason in (
            (KC, JR_OWNER, "wrong_customer"),
            (KC, "kc-nobody", "unknown_user"),
            (KC, COLBY, "suspended_user"),
        ):
            with pytest.raises(ControlError) as refused:
                made.client.start_run(customer, user)
            assert (refused.value.status, refused.value.reason) == (403, reason)
        denied = [e.reason for e in made.events() if e.kind is EventKind.DENY]
        assert denied == ["wrong_customer", "unknown_user", "suspended_user"]
        assert made.client.health()["runs_live"] == 0


# Caps


def test_per_run_caps_tighten_the_policy_and_never_widen_it(tmp_path: Path) -> None:
    with served(tmp_path) as made:
        tight = made.client.start_run(KC, STAFF, max_calls=2, max_usd="0.25")
        wide = made.client.start_run(KC, OWNER, max_calls=1000, max_usd="999")

        assert (tight.max_calls, tight.max_usd) == (2, Decimal("0.25"))
        assert (wide.max_calls, wide.max_usd) == (40, Decimal("0.5"))
        assert [made.post(tight.token).status for _ in range(3)] == [200, 200, 429]
        statuses = [made.post(wide.token).status for _ in range(41)]
        assert statuses == [200] * 40 + [429]


def test_concurrent_last_dollar_reservations_across_runs_stay_under_each_cap(
    tmp_path: Path,
) -> None:
    with served(tmp_path) as made:
        from agentgate.model.pricing import RateTable

        size = len(json.dumps(BODY).encode())
        worst = RateTable.load(made.rates).worst_case_usd("deepseek-flash", 1000, size)
        assert worst is not None
        runs = [made.client.start_run(KC, user, max_usd=worst * 2) for user in (OWNER, STAFF)]
        runs.append(made.client.start_run(KC, "kc-mgr-marisol", max_usd=worst * 2))
        made.provider.hold = threading.Event()
        statuses: dict[str, list[int]] = {run.run_id: [] for run in runs}
        start = threading.Barrier(len(runs) * 5)

        def send(run_id: str, token: str, conn: http.client.HTTPConnection) -> None:
            start.wait()
            statuses[run_id].append(made.post(token, conn=conn).status)

        # Connect one at a time before the barrier: the listener's backlog is 5, and 15
        # connects at once overflow it, which Linux recovers from only after seconds of
        # SYN-ACK retries. The race under test is the reservation, not the TCP accept.
        conns = []
        for _ in range(len(runs) * 5):
            conn = made.connect()
            conn.connect()
            conns.append(conn)

        workers = [
            threading.Thread(target=send, args=(run.run_id, run.token, conns.pop()))
            for run in runs
            for _ in range(5)
        ]
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 30
        while sum(len(s) for s in statuses.values()) < len(runs) * 3:
            assert time.monotonic() < deadline, statuses
            time.sleep(0.02)
        made.provider.hold.set()
        for worker in workers:
            worker.join(30)

        assert {run_id: sorted(s) for run_id, s in statuses.items()} == {
            run.run_id: [200, 200, 429, 429, 429] for run in runs
        }


# The control socket


def test_the_socket_is_0600_in_a_new_0700_directory_and_both_go_on_stop(tmp_path: Path) -> None:
    base = _short_dir()
    try:
        wanted = base / "ctl" / "control.sock"
        with served(tmp_path, control_socket=wanted) as made:
            assert made.socket == wanted
            assert stat.S_IMODE(os.lstat(wanted).st_mode) == 0o600
            assert stat.S_ISSOCK(os.lstat(wanted).st_mode)
            assert stat.S_IMODE(os.lstat(wanted.parent).st_mode) == 0o700
        assert not wanted.parent.exists()
    finally:
        shutil.rmtree(base)


def test_a_default_socket_gets_a_new_private_directory(tmp_path: Path) -> None:
    with served(tmp_path) as made:
        directory = made.socket.parent
        assert stat.S_IMODE(os.lstat(directory).st_mode) == 0o700
        assert stat.S_IMODE(os.lstat(made.socket).st_mode) == 0o600
    assert not directory.exists()


@pytest.mark.parametrize("existing", ["directory", "symlink", "socket"])
def test_a_pre_existing_socket_path_is_refused_not_reused(tmp_path: Path, existing: str) -> None:
    base = _short_dir()
    try:
        directory = base / "ctl"
        if existing == "directory":
            directory.mkdir(mode=0o700)
        elif existing == "symlink":
            (base / "elsewhere").mkdir(mode=0o700)
            directory.symlink_to(base / "elsewhere")
        else:
            directory.mkdir(mode=0o700)
            left = socket.socket(socket.AF_UNIX)
            left.bind(str(directory / "control.sock"))
            left.close()
        provider = Provider()
        try:
            with pytest.raises(ProxySetupError, match="must be new"):
                Served(tmp_path, provider, control_socket=directory / "control.sock")
        finally:
            provider.close()
        assert not (base / "elsewhere" / "control.sock").exists()
    finally:
        shutil.rmtree(base)


def test_the_client_refuses_a_symlinked_socket_path(tmp_path: Path) -> None:
    with served(tmp_path) as made:
        base = _short_dir()
        try:
            link = base / "control.sock"
            link.symlink_to(made.socket)
            with ControlClient(link) as client, pytest.raises(ControlError) as refused:
                client.start_run(KC, STAFF)
            assert refused.value.reason == "unsafe_socket"
            assert made.client.health()["runs_live"] == 0
        finally:
            shutil.rmtree(base)


MALFORMED = [
    (b"", "invalid_json"),
    (b"{", "invalid_json"),
    (b"[]", "invalid_json"),
    (b"\xff\xfe", "invalid_json"),
    (b'{"customer_id": "a", "customer_id": "b", "user_id": "u"}', "invalid_json"),
    (b'{"user_id": "kc-staff-nadine"}', "missing_field"),
    (b'{"customer_id": "kc-creek-hvac", "user_id": "u", "role": "owner"}', "unknown_field"),
    (b'{"customer_id": 1, "user_id": "u"}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": ""}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": "u", "ttl_seconds": true}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": "u", "ttl_seconds": 0}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": "u", "max_usd": 0.5}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": "u", "max_usd": "NaN"}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": "u", "max_usd": "-1"}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": "u", "max_usd": "lots"}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": "u", "max_calls": 1.5}', "invalid_field"),
    (b'{"customer_id": "k", "user_id": "u", "max_calls": 0}', "invalid_field"),
]


def _raw(path: Path, request: bytes) -> tuple[int, dict[str, str], bytes]:
    """Send one raw HTTP request on the control socket; the status, headers and body."""
    with socket.socket(socket.AF_UNIX) as conn:
        conn.settimeout(15)
        conn.connect(str(path))
        # A busy server answers and closes without reading the request; its answer is
        # still there to read. Linux resets a socket closed with unread data, so the send
        # or the read after the answer can fail with a reset instead of a clean end.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            conn.sendall(request)
        data = b""
        with contextlib.suppress(ConnectionResetError):
            while chunk := conn.recv(65536):
                data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    headers = dict(line.split(": ", 1) for line in lines[1:])
    return int(lines[0].split()[1]), {k.lower(): v for k, v in headers.items()}, body


def _post(path: Path, target: str, body: bytes) -> bytes:
    return (
        f"POST {target} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode() + body


def test_a_malformed_control_body_gets_400_with_a_reason_and_no_traceback(
    tmp_path: Path,
) -> None:
    with served(tmp_path) as made:
        for body, reason in MALFORMED:
            status, headers, answer = _raw(made.socket, _post(made.socket, "/runs", body))
            assert (status, json.loads(answer)) == (400, {"error": reason}), body
            assert headers["content-type"] == "application/json"
            assert b"Traceback" not in answer
        assert made.client.health()["runs_live"] == 0
        assert [e for e in made.events() if e.kind is EventKind.RUN_STARTED] == []


def test_other_routes_methods_and_oversized_bodies_are_refused(tmp_path: Path) -> None:
    with served(tmp_path) as made:
        cases = [
            (b"GET /runs HTTP/1.1\r\nHost: x\r\n\r\n", 405, "method_not_allowed"),
            (b"DELETE /health HTTP/1.1\r\nHost: x\r\n\r\n", 405, "method_not_allowed"),
            (b"BREW /health HTTP/1.1\r\nHost: x\r\n\r\n", 405, "method_not_allowed"),
            (b"GET /admin HTTP/1.1\r\nHost: x\r\n\r\n", 404, "not_found"),
            (b"GET /runs/a/b/end HTTP/1.1\r\nHost: x\r\n\r\n", 404, "not_found"),
            (_post(made.socket, "/runs", b"{" * 5000), 413, "body_too_large"),
            (b"GET /health HTTP/x\r\n\r\n", 400, "bad_request"),
            (b"garbage\r\n\r\n", 400, "bad_request"),
        ]
        for request, status, reason in cases:
            got, _, answer = _raw(made.socket, request)
            assert (got, json.loads(answer)) == (status, {"error": reason}), request
            assert b"Traceback" not in answer


def test_a_start_response_holds_the_token_and_the_caps_in_effect(tmp_path: Path) -> None:
    with served(tmp_path) as made:
        body = json.dumps({"customer_id": KC, "user_id": STAFF, "ttl_seconds": 999_999})
        status, headers, answer = _raw(made.socket, _post(made.socket, "/runs", body.encode()))
        data = json.loads(answer)
        assert status == 201 and headers["cache-control"] == "no-store"
        assert set(data) == {
            "run_id",
            "customer_id",
            "user_id",
            "token",
            "expires_at",
            "max_usd",
            "max_calls",
            "per_customer_day_usd",
        }
        assert (data["max_usd"], data["max_calls"], data["per_customer_day_usd"]) == (
            "0.5",
            40,
            "20",
        )
        # The lifetime asked for is cut to the server's (one hour by default).
        (token,) = TokenStore(made.world.store, made.world.clock).list_tokens(data["run_id"])
        assert token.expires_at - token.issued_at == timedelta(hours=1)
        assert PROVIDER_KEY not in answer.decode()


def test_requests_beyond_the_control_bound_are_answered_503(tmp_path: Path) -> None:
    with served(tmp_path, workers=1) as made:
        idle = socket.socket(socket.AF_UNIX)
        idle.connect(str(made.socket))  # holds the one worker: it never sends a request
        try:
            time.sleep(0.2)
            status, _, answer = _raw(made.socket, b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            assert (status, json.loads(answer)) == (503, {"error": "busy"})
        finally:
            idle.close()
        deadline = time.monotonic() + 30
        while True:
            status, _, _ = _raw(made.socket, b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            if status == 200 or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        assert status == 200


# Lifecycle


def test_stop_ends_every_run_and_revokes_its_tokens(tmp_path: Path) -> None:
    provider = Provider()
    made = Served(tmp_path, provider)
    try:
        runs = [made.client.start_run(KC, user) for user in (OWNER, STAFF)]
        made.client.end_run(runs[0].run_id)
        made.client.close()
        made.proxy.stop()
        tokens = TokenStore(made.world.store, made.world.clock)
        for run in runs:
            (token,) = tokens.list_tokens(run.run_id)
            assert token.revoked_reason == "run_ended"
        assert _run_ended(made.events()) == {
            runs[0].run_id: "requested",
            runs[1].run_id: "shutdown",
        }
        result, _ = verify_log(made.audit_path)
        assert result["valid"], result
        assert not made.socket.exists()
    finally:
        made.world.close()
        provider.close()


def test_a_crashed_instances_runs_are_ended_on_the_next_start(tmp_path: Path) -> None:
    executable = shutil.which("agentgate", path=str(Path(sys.executable).parent))
    assert executable is not None
    provider = Provider()
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({UPSTREAM.value: PROVIDER_KEY}))
    keys.chmod(0o600)
    ready = tmp_path / "ready.json"
    command = [
        executable,
        "model-proxy",
        "server",
        "--upstream",
        f"openai_compatible={provider.origin}",
        "--rate-table",
        str(rates),
        "--keys-file",
        str(keys),
        "--ready-file",
        str(ready),
        "--db",
        str(tmp_path / "data" / "agentgate.db"),
        "--audit-db",
        str(tmp_path / "data" / "audit.db"),
        "--policies",
        str(POLICIES),
        "--fixtures",
        str(FIXTURES),
    ]
    log = tmp_path / "stderr.txt"
    with log.open("w") as errlog:
        crashed = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errlog)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert crashed.poll() is None, log.read_text()
            assert time.monotonic() < deadline, log.read_text()
            time.sleep(0.05)
        info = json.loads(ready.read_text())
        with ControlClient(info["control_socket"]) as client:
            runs = [client.start_run(KC, user) for user in (OWNER, STAFF)]
        crashed.send_signal(signal.SIGKILL)
        crashed.wait(10)
    finally:
        if crashed.poll() is None:
            crashed.kill()
            crashed.wait(10)
    try:
        with served(tmp_path) as made:
            assert made.proxy.recovered == 2
            for run in runs:
                assert made.post(run.token).status == 401
            ended = _run_ended(made.events())
            assert ended == {run.run_id: "recovered" for run in runs}
            assert made.provider.seen == []
    finally:
        shutil.rmtree(Path(info["control_socket"]).parent, ignore_errors=True)
        provider.close()


def test_a_live_instances_runs_are_left_alone_by_a_second_server(tmp_path: Path) -> None:
    with served(tmp_path) as first:
        run = first.client.start_run(KC, STAFF)
        provider = Provider()
        second = Served(tmp_path, provider)
        try:
            assert second.proxy.recovered == 0
            assert first.post(run.token).status == 200
            # The second server does not serve the first one's run.
            assert second.post(run.token).status == 401
        finally:
            second.close()
            provider.close()


def test_the_server_needs_a_policy_with_a_served_model_and_a_key(tmp_path: Path) -> None:
    world = open_file_world(tmp_path)
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    try:
        for upstreams, keys, message in (
            (
                {ModelProvider.ANTHROPIC: "http://127.0.0.1:9"},
                {ModelProvider.ANTHROPIC: "k"},
                "no customer's policy lists a model",
            ),
            ({UPSTREAM: "http://127.0.0.1:9"}, {ModelProvider.ANTHROPIC: "k"}, "no provider key"),
        ):
            config = ServerConfig(upstreams=upstreams, keys=ProviderKeys(keys), rate_table=rates)
            with pytest.raises(ProxySetupError, match=message):
                MultiRunProxy(world, config).start()
    finally:
        world.close()


# The command line


def test_a_non_loopback_host_is_one_error_line_not_a_traceback(tmp_path: Path) -> None:
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({UPSTREAM.value: PROVIDER_KEY}))
    keys.chmod(0o600)
    result = _cli(
        tmp_path,
        "model-proxy",
        "server",
        "--upstream",
        "openai_compatible=http://127.0.0.1:9",
        "--rate-table",
        str(rates),
        "--keys-file",
        str(keys),
        "--host",
        "0.0.0.0",
    )
    assert "non-loopback" in _error(result)
    assert PROVIDER_KEY not in result.output


# Storage


def test_a_run_whose_limits_cannot_be_stored_does_not_exist(tmp_path: Path) -> None:
    world = open_file_world(tmp_path)
    try:

        def fail(ctx: Any) -> None:
            raise RuntimeError("disk full")

        with pytest.raises(RuntimeError, match="disk full"):
            world.gate.start_run(KC, STAFF, record=fail)
        runs = world.store.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert runs == 0
        assert [
            e
            for e in read_events(tmp_path / "data" / "audit.db")
            if e.kind is EventKind.RUN_STARTED
        ] == []
    finally:
        world.close()


def test_a_store_from_before_served_runs_reads_read_only(tmp_path: Path) -> None:
    # Read-only opens never migrate, so a store without the new tables reads as one in which
    # no run has stored caps and no server started a run.
    path = tmp_path / "old.db"
    Store(path).close()
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE run_limits")
    conn.execute("DROP TABLE served_runs")
    conn.commit()
    conn.close()

    with Store(path, readonly=True) as reader:
        assert reader.get_run_limits("any-run") is None
        assert reader.get_served_run("any-run") is None
        assert reader.list_served_runs() == []
    with Store(path, readonly=True) as reader:
        tables = {
            row[0] for row in reader.connection.execute("SELECT name FROM sqlite_master").fetchall()
        }
    assert "run_limits" not in tables and "served_runs" not in tables


# Mutation proofs: each protection removed, its test must fail.


def _fails(test: Any, tmp_path: Path) -> bool:
    try:
        test(tmp_path)
    except (AssertionError, pytest.fail.Exception):
        return True
    return False


def test_without_the_served_runs_check_a_serve_token_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edits = (("if not self._proxy.serves(ctx.run_id):", "if False:"),)
    monkeypatch.setattr(ServedRuns, "resolve", mutant(ServedRuns, "resolve", edits, {}))
    assert _fails(
        test_a_token_from_a_serve_process_on_the_same_world_is_refused_not_served, tmp_path
    )


def test_without_per_run_caps_the_caps_test_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_caps(self: ServedRuns, run: Any) -> Caps | None:
        return None

    monkeypatch.setattr(ServedRuns, "caps", no_caps)
    assert _fails(test_per_run_caps_tighten_the_policy_and_never_widen_it, tmp_path)
