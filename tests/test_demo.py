"""The demo: its output matches the golden file byte for byte, every scenario behaves, a
misbehaving policy is reported, and nothing it prints leaks a document body or a canary.
"""

import difflib
import io
import shutil
from pathlib import Path

import pytest

from agentgate.audit.panoptes import PanoptesSink
from agentgate.demo import AUDIT_FILE, SCENARIOS, WIDTH, run_demo
from agentgate.fixtures import find_canaries, load_fixtures
from agentgate.store import Store
from conftest import FIXTURES, KC, POLICIES

GOLDEN = Path(__file__).resolve().parent / "golden" / "demo.txt"


def demo_output(policies: Path = POLICIES, audit_dir: Path | None = None) -> tuple[bool, str]:
    out = io.StringIO()
    passed = run_demo(out, audit_dir=audit_dir, policies=policies, fixtures=FIXTURES)
    return passed, out.getvalue()


def assert_matches_golden(text: str) -> None:
    """Equal to the golden file, or fail with a unified diff of what changed."""
    expected = GOLDEN.read_text(encoding="utf-8")
    if text == expected:
        return
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile="tests/golden/demo.txt",
            tofile="demo output",
        )
    )
    pytest.fail(f"demo output differs from the golden file:\n{diff}")


def test_output_matches_golden_file() -> None:
    passed, text = demo_output()
    assert passed
    assert_matches_golden(text)


def test_golden_helper_reports_a_diff() -> None:
    _, text = demo_output()
    changed = text.replace("ok\n", "UNEXPECTED\n", 1)
    with pytest.raises(pytest.fail.Exception, match=r"(?s)^demo output differs.*\n\+UNEXPECTED"):
        assert_matches_golden(changed)


def test_every_scenario_behaves() -> None:
    passed, text = demo_output()
    assert passed
    lines = text.splitlines()
    assert lines.count("ok") == len(SCENARIOS) == 10
    assert "UNEXPECTED" not in text
    assert lines[-1] == "demo: all 10 scenarios behaved as expected"


def test_two_runs_are_identical() -> None:
    assert demo_output()[1] == demo_output()[1]


def test_output_is_plain_ascii_within_width() -> None:
    _, text = demo_output()
    assert text.isascii()
    assert "\x1b" not in text
    for line in text.splitlines():
        assert len(line) <= WIDTH, line
        assert line == line.rstrip()


def test_output_has_no_canary_and_no_document_body() -> None:
    _, text = demo_output()
    assert find_canaries(text) == []
    store = Store()
    try:
        load_fixtures(store, FIXTURES)
        documents = [d for c in store.list_customers() for d in store.list_documents(c.id)]
    finally:
        store.close()
    assert documents
    for document in documents:
        for line in document.body.splitlines():
            # Short lines (a heading word, a table rule) can match by chance; real content
            # lines are longer than any title the demo prints.
            if len(line.strip()) >= 25:
                assert line.strip() not in text, (document.id, line)


def test_shows_the_key_decisions() -> None:
    _, text = demo_output()
    # Scenario 1: the filter and the not_found read with its real audit reason.
    assert "filter: 20 documents in the tenant, 13 permitted, 7 filtered (default_deny 7)" in text
    assert "gate: denied, model told not_found (audit: deny default_deny)" in text
    # Scenario 4: the rule that stopped the export.
    assert "rule\n        no-external-after-confidential)" in text
    # Scenario 7, 9 and the audit trailer.
    assert "(audit: deny document_customer_mismatch)" in text
    assert "start_run: refused, IdentityError suspended_user" in text
    assert "caller told invalid_run (audit: deny invalid_run)" in text
    assert "audit chain: intact (" in text


def _loosen_staff_reads(tmp_path: Path) -> Path:
    policies = tmp_path / "policies"
    shutil.copytree(POLICIES, policies)
    path = policies / KC / "policy.yaml"
    original = path.read_text(encoding="utf-8")
    staff_rule = (
        "  - id: staff-read\n"
        "    effect: allow\n"
        "    description: Staff may search and read documents up to internal.\n"
        "    roles: [staff]\n"
        "    tools: [docs.search, docs.read]\n"
        "    actions: [read]\n"
        "    labels: {max: internal}\n"
    )
    assert staff_rule in original
    path.write_text(
        original.replace(staff_rule, staff_rule.replace("{max: internal}", "{max: confidential}")),
        encoding="utf-8",
    )
    return policies


def test_a_misbehaving_policy_is_reported(tmp_path: Path) -> None:
    passed, text = demo_output(_loosen_staff_reads(tmp_path))
    assert not passed
    assert "UNEXPECTED: " in text
    assert find_canaries(text) == []
    last = text.splitlines()[-1]
    assert last.startswith("demo: ")
    assert last.endswith(" of 10 scenarios UNEXPECTED")
    # The staff search scenario is the first to notice.
    first = text.split("== 2.")[0]
    assert "UNEXPECTED: no sensitive document returned" in first


def test_audit_dir_keeps_a_verifiable_log(tmp_path: Path) -> None:
    passed, text = demo_output(audit_dir=tmp_path)
    assert passed
    sink = PanoptesSink(tmp_path / AUDIT_FILE, create=False)
    try:
        assert sink.verify()["valid"] is True
        count = sink.count()
    finally:
        sink.close()
    assert f"audit chain: intact ({count} events)" in text
    assert f"events: {count}\n" in text
    # The same output as with a temporary directory: the path is never printed.
    assert_matches_golden(text)


def test_audit_dir_with_an_existing_log_is_refused(tmp_path: Path) -> None:
    (tmp_path / AUDIT_FILE).write_bytes(b"")
    with pytest.raises(FileExistsError, match="already exists"):
        demo_output(audit_dir=tmp_path)
