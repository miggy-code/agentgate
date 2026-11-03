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

from click.testing import Result

from agentgate.audit.panoptes import read_events
from agentgate.cli import app
from conftest import FIXTURES, POLICIES
from proxyharness import BODY, PROVIDER_KEY, RATES, STAFF, Provider
from test_cli_ops import _audit, _cli, _db, _ok, _paths, runner


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


def test_serve_runs_a_run_for_a_client_and_ends_it_on_interrupt(tmp_path: Path) -> None:
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
        server.send_signal(signal.SIGINT)
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
