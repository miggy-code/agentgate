"""`agentgate model-proxy serve` and `agentgate doctor model-proxy`: the command line around
the proxy. The serve test runs the real command as a subprocess against a fake provider."""

import http.client
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import Result

from agentgate.audit.panoptes import read_events
from agentgate.cli import app
from conftest import FIXTURES, POLICIES
from proxyharness import BODY, PROVIDER_KEY, RATES, STAFF, Provider
from test_cli_ops import _audit, _cli, _db, _error, _ok, _paths, runner


def _files(tmp_path: Path, provider: str = "openai_compatible") -> tuple[Path, Path]:
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({provider: PROVIDER_KEY}))
    keys.chmod(0o600)
    return rates, keys


def _doctor(
    tmp_path: Path,
    *extra: str,
    upstream: str = "openai_compatible=http://127.0.0.1:9",
    key_for: str = "openai_compatible",
) -> Result:
    rates, keys = _files(tmp_path, key_for)
    return _cli(
        tmp_path,
        "doctor",
        "model-proxy",
        "--as",
        STAFF,
        "--upstream",
        upstream,
        "--rate-table",
        str(rates),
        "--keys-file",
        str(keys),
        *extra,
    )


def test_doctor_says_ok_when_the_proxy_could_start(tmp_path: Path) -> None:
    result = _ok(_doctor(tmp_path))
    assert result.stdout.strip() == "model-proxy: ok"


def test_doctor_fails_for_an_upstream_the_policy_lists_no_model_for(tmp_path: Path) -> None:
    result = _doctor(tmp_path, upstream="anthropic=http://127.0.0.1:9", key_for="anthropic")
    assert result.exit_code == 1
    assert "model-proxy: FAIL" in result.stdout and PROVIDER_KEY not in result.output


def test_doctor_fails_without_a_key_and_never_prints_one(tmp_path: Path) -> None:
    result = _doctor(tmp_path, key_for="anthropic")
    assert result.exit_code == 1 and "no provider key" in result.stdout
    assert PROVIDER_KEY not in result.output


def test_a_malformed_upstream_is_an_error_before_anything_opens(tmp_path: Path) -> None:
    rates, keys = _files(tmp_path)
    result = runner.invoke(
        app,
        [
            "doctor", "model-proxy", "--as", STAFF, "--upstream", "deepseek",
            "--rate-table", str(rates), "--keys-file", str(keys), *_paths(tmp_path),
        ],
    )  # fmt: skip
    assert result.exit_code == 1 and "--upstream" in result.stderr
    assert not _db(tmp_path).exists()


@pytest.mark.parametrize("stop", [signal.SIGINT, signal.SIGTERM])
def test_serve_runs_a_run_for_a_client_and_ends_it_on_a_stop_signal(
    tmp_path: Path, stop: signal.Signals
) -> None:
    executable = shutil.which("agentgate", path=str(Path(sys.executable).parent))
    assert executable is not None
    rates, keys = _files(tmp_path)
    token_file = tmp_path / "token"
    provider = Provider()
    command = [
        executable,
        "model-proxy",
        "serve",
        "--as",
        STAFF,
        "--upstream",
        f"openai_compatible={provider.origin}",
        "--rate-table",
        str(rates),
        "--keys-file",
        str(keys),
        "--token-file",
        str(token_file),
        "--port",
        "0",
        "--db",
        str(_db(tmp_path)),
        "--audit-db",
        str(_audit(tmp_path)),
        "--policies",
        str(POLICIES),
        "--fixtures",
        str(FIXTURES),
    ]
    log = tmp_path / "stderr.txt"
    with log.open("w") as errlog:
        server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errlog, text=True)
    try:
        port = _wait_for_port(log, server)
        assert stat.S_IMODE(os.stat(token_file).st_mode) == 0o600
        token = token_file.read_text().strip()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(BODY),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 200, response.read()
        response.read()
        conn.close()
        (seen,) = provider.seen
        assert dict((k.lower(), v) for k, v in seen.headers)["authorization"] == (
            f"Bearer {PROVIDER_KEY}"
        )
    finally:
        server.send_signal(stop)
        server.wait(timeout=30)
        provider.close()
    assert server.returncode == 0, log.read_text()
    assert server.stdout is not None and server.stdout.read() == ""  # the token is in the file
    assert not token_file.exists()
    reasons = [e.reason for e in read_events(_audit(tmp_path))]
    assert "token_issued" in reasons and "model_settled" in reasons
    assert reasons.index("token_revoked") > reasons.index("model_settled")
    assert PROVIDER_KEY not in log.read_text() and token not in log.read_text()


def _wait_for_port(log: Path, server: subprocess.Popen[str]) -> int:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise AssertionError(log.read_text())
        for line in log.read_text().splitlines():
            if line.startswith("model proxy:"):
                return int(line.rsplit(":", 1)[1])
        time.sleep(0.05)
    raise AssertionError("the proxy did not start: " + log.read_text())


def test_the_gateway_needs_its_model_options_together(tmp_path: Path) -> None:
    rates, _ = _files(tmp_path)
    upstream = ["--upstream", "openai_compatible=http://127.0.0.1:9"]
    for args in (
        ["--model-proxy"],
        ["--model-proxy", *upstream, "--rate-table", str(rates)],  # no --token-file
        [*upstream],
        ["--max-calls", "3"],
    ):
        result = runner.invoke(app, ["gateway", "serve", "--as", STAFF, *args, *_paths(tmp_path)])
        assert result.exit_code == 1 and "--model-proxy" in result.stderr, (args, result.stderr)
        assert not _db(tmp_path).exists()


def test_spend_show_reports_a_runs_model_spend(tmp_path: Path) -> None:
    from decimal import Decimal

    from proxyharness import OK_COST, rig

    with rig(tmp_path) as (made, _):
        assert made.post().status == 200
        made.reservations()
        run = made.run.run_id
        lines = dict(
            line.split(" ", 1)
            for line in _ok(
                _cli(tmp_path, "spend", "show", "--as", STAFF, "--run", run)
            ).stdout.splitlines()
        )
    assert Decimal(lines["spent_usd"]) == OK_COST and lines["calls"] == "1"
    assert lines["settled"] == "1" and lines["open"] == "0"
    assert "not found" in _error(
        _cli(tmp_path, "spend", "show", "--as", "jr-owner-marisol", "--run", run)
    )


def test_serve_writes_a_ready_file_with_the_caps_in_effect(tmp_path: Path) -> None:
    """A models-only experiment from init-experiment, served with --ready-file: the file
    appears complete and private once listening, states the tightened caps, and the proxy
    admits the listed model. It is removed on stop."""
    executable = shutil.which("agentgate", path=str(Path(sys.executable).parent))
    assert executable is not None
    lab = tmp_path / "lab"
    init = runner.invoke(
        app,
        [
            "init-experiment", str(lab), "--model", "openai_compatible:deepseek-flash",
            "--max-usd", "0.50", "--max-calls", "40", "--day-usd", "5",
        ],
    )  # fmt: skip
    assert init.exit_code == 0, init.output
    rates, keys = _files(tmp_path)
    token_file, ready_file = tmp_path / "token", tmp_path / "ready.json"
    provider = Provider()
    command = [
        executable, "model-proxy", "serve", "--as", "lab-runner",
        "--upstream", f"openai_compatible={provider.origin}",
        "--rate-table", str(rates), "--keys-file", str(keys),
        "--token-file", str(token_file), "--ready-file", str(ready_file),
        "--max-usd", "0.25", "--max-calls", "100",
        "--db", str(_db(tmp_path)), "--audit-db", str(_audit(tmp_path)),
        "--policies", str(lab / "policies"), "--fixtures", str(lab / "fixtures"),
    ]  # fmt: skip
    log = tmp_path / "stderr.txt"
    with log.open("w") as errlog:
        server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errlog, text=True)
    try:
        deadline = time.monotonic() + 30
        while not ready_file.exists():
            assert server.poll() is None, log.read_text()
            assert time.monotonic() < deadline, log.read_text()
            time.sleep(0.05)
        assert stat.S_IMODE(os.stat(ready_file).st_mode) == 0o600
        ready = json.loads(ready_file.read_text())
        port = _wait_for_port(log, server)  # the stderr line is kept
        run_line = next(line for line in log.read_text().splitlines() if "model proxy:" in line)
        assert ready == {
            "url": f"http://127.0.0.1:{port}",
            "host": "127.0.0.1",
            "port": port,
            "pid": server.pid,
            "run_id": ready["run_id"],
            "user": "lab-runner",
            "customer": "lab",
            # --max-usd tightens the policy's 0.50; --max-calls cannot raise its 40.
            "max_usd": "0.25",
            "max_calls": 40,
            "per_customer_day_usd": "5",
            "token_file": str(token_file),
        }
        assert f"run {ready['run_id']} for lab-runner" in run_line
        token = token_file.read_text().strip()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(BODY),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 200, response.read()
        response.read()
        conn.close()
    finally:
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=30)
        provider.close()
    assert server.returncode == 0, log.read_text()
    assert not ready_file.exists() and not token_file.exists()
    assert not list(tmp_path.glob(".ready.json.*"))


def test_serve_refuses_an_existing_ready_file_before_starting(tmp_path: Path) -> None:
    rates, keys = _files(tmp_path)
    ready_file = tmp_path / "ready.json"
    ready_file.write_text("stale")
    result = runner.invoke(
        app,
        [
            "model-proxy", "serve", "--as", STAFF,
            "--upstream", "openai_compatible=http://127.0.0.1:9",
            "--rate-table", str(rates), "--keys-file", str(keys),
            "--token-file", str(tmp_path / "token"), "--ready-file", str(ready_file),
            *_paths(tmp_path),
        ],
    )  # fmt: skip
    assert result.exit_code == 1 and "already exists" in result.stderr
    assert ready_file.read_text() == "stale" and not (tmp_path / "token").exists()
    assert not _db(tmp_path).exists()


def test_ready_file_cleanup_removes_only_this_process_file(tmp_path: Path) -> None:
    from agentgate.cli import _remove_own_ready_file

    def identity(path: Path) -> tuple[int, int]:
        info = os.lstat(path)
        return info.st_dev, info.st_ino

    mine, theirs, link = tmp_path / "mine.json", tmp_path / "theirs.json", tmp_path / "link.json"
    mine.write_text(json.dumps({"pid": os.getpid(), "run_id": "r1"}))
    theirs.write_text(json.dumps({"pid": os.getpid() + 1, "run_id": "r1"}))
    link.symlink_to(mine)
    _remove_own_ready_file(link, "r1", identity(mine))
    _remove_own_ready_file(theirs, "r1", identity(theirs))
    _remove_own_ready_file(mine, "other-run", identity(mine))
    assert link.is_symlink() and theirs.exists() and mine.exists()
    _remove_own_ready_file(mine, "r1", identity(mine))
    assert not mine.exists()


def test_ready_file_cleanup_spares_a_replacement_with_the_same_contents(tmp_path: Path) -> None:
    """Another process replaced the ready file with one that even names this pid and run:
    a different inode, so it is left alone."""
    from agentgate.cli import _remove_own_ready_file, _write_ready_file

    ready = tmp_path / "ready.json"
    info = {"pid": os.getpid(), "run_id": "r1"}
    published = _write_ready_file(ready, info)
    ready.unlink()
    ready.write_text(json.dumps(info))
    _remove_own_ready_file(ready, "r1", published)
    assert ready.exists()
    ready.unlink()
    _remove_own_ready_file(ready, "r1", _write_ready_file(ready, info))
    assert not ready.exists()


def test_ready_file_caps_are_the_caps_admission_enforces(tmp_path: Path) -> None:
    """The ready file reports max_calls 1 (tightened from the policy's 40).

    The second call is refused."""
    executable = shutil.which("agentgate", path=str(Path(sys.executable).parent))
    assert executable is not None
    lab = tmp_path / "lab"
    init = runner.invoke(
        app,
        [
            "init-experiment",
            str(lab),
            "--model",
            "openai_compatible:deepseek-flash",
            "--max-usd",
            "0.50",
            "--max-calls",
            "40",
        ],
    )
    assert init.exit_code == 0, init.output
    rates, keys = _files(tmp_path)
    token_file, ready_file = tmp_path / "token", tmp_path / "ready.json"
    provider = Provider()
    command = [
        executable, "model-proxy", "serve", "--as", "lab-runner",
        "--upstream", f"openai_compatible={provider.origin}",
        "--rate-table", str(rates), "--keys-file", str(keys),
        "--token-file", str(token_file), "--ready-file", str(ready_file), "--max-calls", "1",
        "--db", str(_db(tmp_path)), "--audit-db", str(_audit(tmp_path)),
        "--policies", str(lab / "policies"), "--fixtures", str(lab / "fixtures"),
    ]  # fmt: skip
    log = tmp_path / "stderr.txt"
    with log.open("w") as errlog:
        server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errlog, text=True)
    try:
        deadline = time.monotonic() + 30
        while not ready_file.exists():
            assert server.poll() is None, log.read_text()
            assert time.monotonic() < deadline, log.read_text()
            time.sleep(0.05)
        ready = json.loads(ready_file.read_text())
        assert ready["max_calls"] == 1
        token = token_file.read_text().strip()
        statuses = []
        for _ in range(2):
            conn = http.client.HTTPConnection("127.0.0.1", ready["port"], timeout=30)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(BODY),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            response = conn.getresponse()
            statuses.append(response.status)
            response.read()
            conn.close()
        assert statuses[0] == 200 and statuses[1] in (402, 403, 429), statuses
    finally:
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=30)
        provider.close()
    assert server.returncode == 0, log.read_text()
    assert not ready_file.exists()
