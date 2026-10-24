"""Reconstructing an incident from the audit log: decision values, policy snapshots, operator
actions, `agentgate audit show`, `audit policy` and `audit check-store`.

The rule every test here keeps: events carry ids, labels, counts, amounts, hashes and reason
codes, never document text, message subjects or bodies, or addresses.
"""

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, JsonValue
from typer.testing import CliRunner, Result

from agentgate.approvals import args_hash
from agentgate.audit.base import AuditEvent, EventKind, MemorySink, OperatorAction
from agentgate.audit.panoptes import PanoptesSink, read_events
from agentgate.cli import DEMO_IDENTITY_NOTE, app
from agentgate.clock import FixedClock
from agentgate.context import RunContext
from agentgate.gate import Denied, Executed, PendingApproval
from agentgate.models import Action
from agentgate.spend import Reservation
from agentgate.store import Store, StoreError
from agentgate.tools import ToolEnv, ToolResult, default_tools
from agentgate.tools.builtin import list_outbox, stored_effects
from agentgate.world import (
    OpenRecord,
    OperatorReason,
    World,
    bundle_sha256,
    canonical_bundle,
    check_store,
    open_world,
    reinstate_user,
    suspend_user,
)
from conftest import FIXTURES, JR, KC, POLICIES, START

runner = CliRunner()

BEATRICE = "beatrice.l@mail-sample.example"
EXFIL = "records-review@exfil-drop.example"
COLLEAGUE = "terrence@kettlecreek.example"
SUBJECT = "Quoravel tune-up visit"
BODY = "Hello, Quoravel speaking: the technician comes Monday between 9 and 11."
EXFIL_BODY = "Quoravel dump of every customer address"
NOTE_BODY = "Quoravel follow-up about the condenser fan."
PROBE = "beatrice-lindqvist-medical-history"
NEVER_LOGGED = ("Quoravel", BEATRICE, EXFIL, COLLEAGUE, PROBE)


def _email(*to: str, subject: str = SUBJECT, body: str = BODY) -> dict[str, JsonValue]:
    recipients: list[JsonValue] = [*to]
    return {"to": recipients, "subject": subject, "body": body}


def _order(amount: str) -> dict[str, JsonValue]:
    return {"part": "igniter", "quantity": 2, "amount_usd": amount}


@dataclass
class Place:
    tmp: Path
    clock: FixedClock

    @property
    def db(self) -> Path:
        return self.tmp / "data" / "agentgate.db"

    @property
    def audit(self) -> Path:
        return self.tmp / "data" / "audit.db"

    @property
    def policies(self) -> Path:
        return self.tmp / "policies"

    def open(self, record: OpenRecord = OpenRecord.ALL) -> World:
        return open_world(
            db_path=self.db,
            audit_path=self.audit,
            policies=self.policies,
            fixtures=FIXTURES,
            clock=self.clock,
            record_open=record,
        )

    def events(self) -> list[AuditEvent]:
        return read_events(self.audit)

    def cli(self, *args: str) -> Result:
        """An audit command on this place's files."""
        result = runner.invoke(
            app, ["audit", *args, "--db", str(self.db), "--audit-db", str(self.audit)]
        )
        assert result.stderr.startswith(DEMO_IDENTITY_NOTE + "\n"), result.stderr
        assert "Traceback" not in result.output
        return result

    def world_cli(self, *args: str) -> Result:
        """A world-opening command on this place's files."""
        paths = ["--db", str(self.db), "--audit-db", str(self.audit)]
        paths += ["--policies", str(self.policies), "--fixtures", str(FIXTURES)]
        return runner.invoke(app, [*args, *paths])

    def edit_policy(self, old: str, new: str) -> None:
        path = self.policies / KC / "policy.yaml"
        text = path.read_text()
        assert text.count(old) == 1
        path.write_text(text.replace(old, new))


@pytest.fixture
def place(tmp_path: Path) -> Place:
    shutil.copytree(POLICIES, tmp_path / "policies")
    return Place(tmp_path, FixedClock(START))


def _approved_email(world: World, ctx: RunContext, args: dict[str, JsonValue]) -> str:
    pending = world.gate.call_tool(ctx, "email.send", Action.SEND, args)
    assert isinstance(pending, PendingApproval), pending
    world.gate.approve(pending.approval_id, "kc-mgr-terrence", "fine")
    executed = world.gate.execute_approved(
        ctx, pending.approval_id, "email.send", Action.SEND, args
    )
    assert isinstance(executed, Executed), executed
    return pending.approval_id


def _flow(place: Place) -> str:
    """An approved customer email, an internal email, a 480 USD order and a note."""
    with place.open() as world:
        staff = world.gate.start_run(KC, "kc-staff-nadine")
        approval_id = _approved_email(world, staff, _email(BEATRICE))
        internal = world.gate.call_tool(staff, "email.send", Action.SEND, _email(COLLEAGUE))
        assert isinstance(internal, Executed), internal
        manager = world.gate.start_run(KC, "kc-mgr-marisol")
        order = world.gate.call_tool(manager, "parts.order", Action.SPEND, _order("480.00"))
        assert isinstance(order, Executed), order
        note = {"title": "Call back", "body": NOTE_BODY, "label": "internal"}
        assert isinstance(
            world.gate.call_tool(manager, "notes.write", Action.WRITE, note), Executed
        )
    return approval_id


def _sql(place: Place, statement: str, params: tuple[str, ...]) -> None:
    with closing(sqlite3.connect(place.db)) as conn, conn:
        assert conn.execute(statement, params).rowcount == 1


def _log_bytes(place: Place) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(place.audit.parent.glob("audit.db*")))


# Decision values


def test_decision_events_carry_hashes_counts_and_amounts(place: Place) -> None:
    approval_id = _flow(place)
    with place.open(OpenRecord.NONE) as world:
        approval = world.gate.approvals.get(approval_id)
        outbox = list_outbox(world.store, KC)
        effects = {e.tool: e.id for e in stored_effects(world.store, KC)}
    assert approval is not None
    events = place.events()
    about = [e for e in events if e.subject == approval_id]
    assert [e.kind for e in about] == [
        EventKind.APPROVAL_REQUIRED,
        EventKind.APPROVED,
        EventKind.ALLOW,
        EventKind.EXECUTED,
    ]
    assert all(e.details["approval_args_hash"] == approval.args_hash for e in about)
    customer_hash = args_hash("email.send", Action.SEND, _email(BEATRICE))
    for event in about:
        if event.kind is not EventKind.APPROVED:
            assert event.details["args_hash"] == customer_hash
            assert (event.details["recipients"], event.details["recipients_external"]) == (1, 1)
    assert about[-1].details["effect_ref"] == outbox[0].id

    internal = [
        e
        for e in events
        if e.details.get("args_hash") == args_hash("email.send", Action.SEND, _email(COLLEAGUE))
    ]
    assert [e.kind for e in internal] == [EventKind.ALLOW, EventKind.EXECUTED]
    assert internal[0].details["recipients_internal"] == 1
    assert internal[0].details["recipients_external"] == 0

    order = [e for e in events if e.tool == "parts.order"]
    assert [(e.kind, e.details["amount_usd"]) for e in order] == [
        (EventKind.ALLOW, "480.00"),
        (EventKind.EXECUTED, "480.00"),
    ]
    assert order[-1].details["effect_ref"] == effects["parts.order"]
    note = [e for e in events if e.tool == "notes.write" and e.kind is EventKind.EXECUTED]
    assert note[0].details["effect_ref"] == effects["notes.write"]


def test_args_hash_is_over_the_validated_args(place: Place) -> None:
    # A recipient as typed and as the tool stores it: the event hash is of the stored form.
    typed = _email(" Terrence@KettleCreek.example ")
    with place.open() as world:
        ctx = world.gate.start_run(KC, "kc-staff-nadine")
        assert isinstance(world.gate.call_tool(ctx, "email.send", Action.SEND, typed), Executed)
    executed = [e for e in place.events() if e.kind is EventKind.EXECUTED][-1]
    assert executed.details["args_hash"] == args_hash("email.send", Action.SEND, _email(COLLEAGUE))


def test_denials_carry_values_but_probes_and_searches_do_not(place: Place) -> None:
    with place.open() as world:
        ctx = world.gate.start_run(KC, "kc-staff-nadine")
        exfil = _email(EXFIL, body=EXFIL_BODY)
        assert isinstance(world.gate.call_tool(ctx, "email.send", Action.SEND, exfil), Denied)
        probe = {"document_id": PROBE}
        assert world.gate.call_tool(ctx, "docs.read", Action.READ, probe) == Denied("not_found")
        query = {"query": "Quoravel furnace"}
        assert isinstance(world.gate.call_tool(ctx, "docs.search", Action.READ, query), Executed)
    events = place.events()
    denied = [e for e in events if e.kind is EventKind.DENY]
    assert denied[0].details["args_hash"] == args_hash("email.send", Action.SEND, exfil)
    assert denied[0].details["recipients_external"] == 1
    assert "args_hash" not in denied[1].details and denied[1].subject.startswith("ref:")
    search = [e for e in events if e.tool == "docs.search"]
    assert search and all("args_hash" not in e.details for e in search)
    assert not [text for text in NEVER_LOGGED if text.encode() in _log_bytes(place)]


class _TextRefTool:
    """parts.order, but its effect_ref is text rather than an id."""

    tool_id = "parts.order"

    def __init__(self) -> None:
        self._real = default_tools()["parts.order"]
        self.args_model = self._real.args_model

    def facts(self, args: BaseModel) -> Any:
        return self._real.facts(args)

    def run(self, env: ToolEnv, ctx: RunContext, args: BaseModel) -> ToolResult:
        return ToolResult(content={"status": "ordered"}, effect_ref="Quoravel order for Bea")


def test_an_effect_ref_that_is_not_an_id_is_recorded_as_a_reference(place: Place) -> None:
    with place.open() as world:
        tools = {**default_tools(), "parts.order": _TextRefTool()}
        gate = type(world.gate)(
            store=world.store,
            engines=world.engines,
            sink=world.sink,
            clock=world.clock,
            index=world.index,
            tools=tools,
        )
        ctx = gate.start_run(KC, "kc-mgr-marisol")
        assert isinstance(
            gate.call_tool(ctx, "parts.order", Action.SPEND, _order("20.00")), Executed
        )
        expected = gate.audit_reference("Quoravel order for Bea")
    executed = [e for e in place.events() if e.kind is EventKind.EXECUTED][-1]
    assert executed.details["effect_ref"] == expected
    assert b"Quoravel" not in _log_bytes(place)


def test_read_events_returns_what_was_recorded(tmp_path: Path) -> None:
    sink = MemorySink()
    with open_world(
        db_path=tmp_path / "db",
        audit_path=tmp_path / "unused",
        policies=POLICIES,
        fixtures=FIXTURES,
        clock=FixedClock(START),
        sink=sink,
        record_open=OpenRecord.ALL,
    ) as world:
        ctx = world.gate.start_run(KC, "kc-staff-nadine")
        world.gate.call_tool(ctx, "docs.read", Action.READ, {"document_id": PROBE})
        _approved_email(world, ctx, _email(BEATRICE))
    panoptes = PanoptesSink(tmp_path / "audit.db")
    for event in sink.events:
        panoptes.record(event)
    panoptes.close()
    assert read_events(tmp_path / "audit.db") == [e.recorded() for e in sink.events]


def test_read_events_refuses_a_missing_or_foreign_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_events(tmp_path / "missing.db")
    store = Store(tmp_path / "store.db")
    store.close()
    with pytest.raises(ValueError):
        read_events(tmp_path / "store.db")


# Policy snapshots


def test_each_open_records_the_policy_and_snapshots_it_once(place: Place) -> None:
    place.open().close()
    place.open().close()
    with place.open(OpenRecord.NONE) as world:
        versions = {c: e.version for c, e in world.engines.items()}
        snapshots = {c: world.store.get_policy_snapshot(v) for c, v in versions.items()}
        listed = world.store.list_policy_snapshots(KC)
        bundles = {c: world.engines[c].bundle for c in versions}
    loaded = [e for e in place.events() if e.kind is EventKind.POLICY_LOADED]
    assert sorted((e.customer_id, e.details["snapshot"]) for e in loaded) == [
        (JR, "kept"),
        (JR, "written"),
        (KC, "kept"),
        (KC, "written"),
    ]
    assert len(listed) == 1
    for customer_id, snapshot in snapshots.items():
        assert snapshot is not None
        assert snapshot.bundle == canonical_bundle(bundles[customer_id])
        assert snapshot.sha256 == bundle_sha256(snapshot.bundle)
        assert versions[customer_id] == "sha256:" + snapshot.sha256[:16]
        mine = [e for e in loaded if e.customer_id == customer_id]
        assert all(e.policy_version == versions[customer_id] for e in mine)
        assert all(e.details["bundle_sha256"] == snapshot.sha256 for e in mine)
        assert all(e.details["snapshot_matches"] is True for e in mine)


def test_record_modes(place: Place) -> None:
    place.open(OpenRecord.NONE).close()
    assert not place.audit.exists() or place.events() == []
    with Store(place.db, readonly=True) as store:
        assert len(store.list_policy_snapshots(KC)) == 1
    place.open(OpenRecord.CHANGES).close()
    assert place.events() == []
    place.edit_policy('amount_usd_lte: "500.00"', 'amount_usd_lte: "400.00"')
    place.open(OpenRecord.CHANGES).close()
    loaded = [(e.kind, e.customer_id) for e in place.events()]
    assert loaded == [(EventKind.POLICY_LOADED, KC)]


def test_an_edited_snapshot_shows_on_the_next_recorded_open(place: Place) -> None:
    with place.open() as world:
        version = world.engines[KC].version
    _sql(place, "UPDATE policy_snapshots SET bundle = 'edited' WHERE version = ?", (version,))
    place.open().close()
    last = [e for e in place.events() if e.kind is EventKind.POLICY_LOADED and e.customer_id == KC]
    assert last[-1].details["snapshot_matches"] is False


# Operator actions


def test_suspend_and_reinstate_are_recorded(place: Place) -> None:
    with place.open() as world:
        suspend_user(
            world, "kc-staff-priya", acting_user_id="kc-owner-dale", reason=OperatorReason.SECURITY
        )
        reinstate_user(
            world, "kc-staff-priya", acting_user_id="kc-owner-dale", reason=OperatorReason.RESOLVED
        )
    ops = [e for e in place.events() if e.kind is EventKind.OPERATOR_ACTION]
    found = [(e.reason, e.user_id, e.subject, e.details.get("reason")) for e in ops[-3:]]
    assert found == [
        ("user_suspended", "kc-owner-dale", "kc-staff-priya", "security"),
        ("user_reinstated", "kc-owner-dale", "kc-staff-priya", "resolved"),
        ("fixture_sync", "kc-owner-dale", None, None),
    ]
    assert ops[-1].details["users_updated"] == ["kc-staff-priya"]
    assert ops[-1].details["user_fields_updated"] == {"kc-staff-priya": ["status"]}


def test_cli_suspension_takes_a_reason_code(place: Place) -> None:
    result = place.world_cli(
        "users", "suspend", "kc-staff-priya", "--as", "kc-owner-dale", "--reason", "offboarding"
    )
    assert result.exit_code == 0, result.output
    bad = place.world_cli(
        "users", "suspend", "kc-staff-colby", "--as", "kc-owner-dale", "--reason", "free text"
    )
    assert bad.exit_code == 2
    event = [e for e in place.events() if e.kind is EventKind.OPERATOR_ACTION][-1]
    assert (event.reason, event.details["reason"], event.details["actor"]) == (
        "user_suspended",
        "offboarding",
        "user",
    )


def test_fixture_load_and_sync_are_recorded(place: Place, tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, fixtures)
    first = open_world(
        db_path=place.db,
        audit_path=place.audit,
        policies=place.policies,
        fixtures=fixtures,
        clock=place.clock,
        record_open=OpenRecord.ALL,
    )
    first.close()
    users = fixtures / KC / "users.yaml"
    users.write_text(
        users.read_text().replace(
            "email: marisol@kettlecreek.example\n  role: manager",
            "email: marisol@kettlecreek.example\n  role: staff",
        )
    )
    open_world(
        db_path=place.db,
        audit_path=place.audit,
        policies=place.policies,
        fixtures=fixtures,
        clock=place.clock,
        record_open=OpenRecord.ALL,
    ).close()
    syncs = [e for e in place.events() if e.reason == OperatorAction.FIXTURE_SYNC.value]
    assert [(e.customer_id, e.details["initial"]) for e in syncs] == [
        (JR, True),
        (KC, True),
        (KC, False),
    ]
    assert syncs[1].details["counts"] == {"documents_added": 20, "users_added": 8}
    assert syncs[2].details["users_updated"] == ["kc-mgr-marisol"]
    assert syncs[2].details["user_fields_updated"] == {"kc-mgr-marisol": ["role"]}
    assert b"@kettlecreek.example" not in _log_bytes(place)


def test_the_stale_sweep_is_recorded_per_customer(place: Place) -> None:
    with place.open() as world:
        runs = [
            world.gate.start_run(KC, "kc-staff-nadine"),
            world.gate.start_run(JR, "jr-staff-priya"),
        ]
        for ctx in runs:
            assert isinstance(world.gate.charge_model(ctx, Decimal("0.30")), Reservation)
    place.clock.advance(timedelta(minutes=20))
    with place.open() as world:
        assert world.stale_released == 2
    sweeps = {e.customer_id: e for e in place.events() if e.reason == "stale_reservations_released"}
    assert {c: (e.details["released"], e.details["released_usd"]) for c, e in sweeps.items()} == {
        KC: (1, "0.30"),
        JR: (1, "0.30"),
    }
    assert sweeps[KC].details["run_ids"] == [runs[0].run_id]
    assert sweeps[KC].user_id is None and sweeps[KC].details["actor"] == "system"


def test_cli_sweep_names_the_owner_only_in_their_own_log(place: Place) -> None:
    with place.open() as world:
        ctx = world.gate.start_run(JR, "jr-staff-priya")
        assert isinstance(world.gate.charge_model(ctx, Decimal("0.30")), Reservation)
    result = place.world_cli("spend", "release-stale", "--as", "kc-owner-dale", "--older-than", "1")
    # The FixedClock is the test's; the CLI's clock is the system's, a day or more later.
    assert result.exit_code == 0, result.output
    sweeps = [e for e in place.events() if e.reason == "stale_reservations_released"]
    by_customer = {e.customer_id: e for e in sweeps}
    assert by_customer[JR].user_id is None and by_customer[JR].details["actor"] == "operator"
    assert by_customer[KC].user_id == "kc-owner-dale" and by_customer[KC].details["released"] == 0


# audit show


def test_audit_show_prints_the_customers_events(place: Place) -> None:
    _flow(place)
    with place.open() as world:
        ctx = world.gate.start_run(KC, "kc-staff-nadine")
        world.gate.call_tool(ctx, "docs.read", Action.READ, {"document_id": PROBE})
        jr = world.gate.start_run(JR, "jr-owner-marisol")
    result = place.cli("show", "--as", "kc-owner-dale")
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    kc_events = [e for e in place.events() if e.customer_id == KC]
    assert lines[-1] == f"{KC}  {len(kc_events)} events"
    assert len(lines) == len(kc_events) + 1
    assert jr.run_id not in result.stdout
    assert any("approval_required" in line and "recipients_external=1" in line for line in lines)
    probe = [line for line in lines if "document_not_found" in line]
    assert len(probe) == 1 and "(unknown value)" in probe[0] and PROBE not in probe[0]

    manager = place.cli("show", "--as", "kc-mgr-marisol", "--kind", "executed", "--limit", "2")
    assert manager.exit_code == 0
    shown = manager.stdout.splitlines()
    assert len(shown) == 3 and all("  executed  " in line for line in shown[:2])

    run = place.cli("show", "--as", "kc-owner-dale", "--run", ctx.run_id, "--json")
    events = json.loads(run.stdout)
    assert [e["kind"] for e in events] == ["run_started", "deny"]
    assert events[1]["references"]["subject"]["why"] == "unknown"


def test_audit_show_refusals(place: Place) -> None:
    _flow(place)
    staff = place.cli("show", "--as", "kc-staff-nadine")
    assert staff.exit_code == 1 and "owners and managers only" in staff.stderr
    other = place.cli("show", "--as", "kc-owner-dale", "--customer", JR)
    assert other.exit_code == 1 and f"customer {JR!r} not found" in other.stderr
    unknown = place.cli("show", "--as", "nobody")
    assert unknown.exit_code == 1 and "unknown user" in unknown.stderr
    missing = runner.invoke(
        app, ["audit", "show", "--as", "kc-owner-dale", "--db", str(place.tmp / "nope.db")]
    )
    assert missing.exit_code == 1 and "store not found" in missing.stderr


def test_audit_commands_are_read_only(place: Place) -> None:
    _flow(place)
    before = (place.db.read_bytes(), place.audit.read_bytes())
    for args in (
        ("show", "--as", "kc-owner-dale"),
        ("show", "--as", "kc-owner-dale", "--json"),
        ("check-store", "--as", "kc-owner-dale"),
    ):
        assert place.cli(*args).exit_code == 0
    with Store(place.db, readonly=True) as store:
        version = store.list_policy_snapshots(KC)[0].version
    assert place.cli("policy", version, "--as", "kc-owner-dale").exit_code == 0
    assert (place.db.read_bytes(), place.audit.read_bytes()) == before


# audit policy


def test_audit_policy_prints_and_checks_the_bundle(place: Place) -> None:
    with place.open() as world:
        version = world.engines[KC].version
    result = place.cli("policy", version, "--as", "kc-mgr-marisol")
    assert result.exit_code == 0, result.output
    *body, summary = result.stdout.splitlines()
    bundle = json.loads("\n".join(body))
    assert sorted(bundle) == ["policy", "tools"]
    assert summary.endswith("1 policy_loaded events  ok")
    other = place.cli("policy", version, "--as", "jr-owner-marisol")
    assert other.exit_code == 1 and "not found" in other.stderr

    _sql(
        place,
        "UPDATE policy_snapshots SET bundle = replace(bundle, '\"500\"', '\"5000\"')"
        " WHERE version = ?",
        (version,),
    )
    tampered = place.cli("policy", version, "--as", "kc-owner-dale")
    assert tampered.exit_code == 1
    assert "MISMATCH the stored bundle does not match a policy_loaded event" in tampered.stdout
    assert "MISMATCH the stored bundle does not hash to its version" in tampered.stdout


def test_audit_policy_without_a_load_event_checks_the_version(place: Place) -> None:
    with place.open(OpenRecord.NONE) as world:
        version = world.engines[KC].version
        world.gate.start_run(KC, "kc-staff-nadine")
    result = place.cli("policy", version, "--as", "kc-owner-dale")
    assert result.exit_code == 0, result.output
    assert "0 policy_loaded events (checked against the version only)  ok" in result.stdout


# audit check-store


def test_check_store_passes_an_honest_store(place: Place) -> None:
    _flow(place)
    result = place.cli("check-store", "--as", "kc-owner-dale")
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        f"{KC}  1 approvals, 2 outbox emails, 1 parts orders, 0 refunds, 1 notes checked"
        "  0 mismatches\n"
    )
    for user in ("kc-mgr-marisol", "kc-staff-nadine"):
        refused = place.cli("check-store", "--as", user)
        assert refused.exit_code == 1 and "owners only" in refused.stderr


def test_check_store_catches_edited_approval_args(place: Place) -> None:
    approval_id = _flow(place)
    forged: dict[str, JsonValue] = _email("harvest@exfil-drop.example")
    _sql(place, "UPDATE approvals SET args = ? WHERE id = ?", (json.dumps(forged), approval_id))
    result = place.cli("check-store", "--as", "kc-owner-dale")
    assert result.exit_code == 1
    assert (
        f"MISMATCH approval {approval_id}: args do not match the row's args_hash" in result.stdout
    )
    # Rehashed so the row agrees with itself: the chain still disagrees.
    rehashed = args_hash("email.send", Action.SEND, forged)
    _sql(place, "UPDATE approvals SET args_hash = ? WHERE id = ?", (rehashed, approval_id))
    result = place.cli("check-store", "--as", "kc-owner-dale")
    assert result.exit_code == 1
    assert f"MISMATCH approval {approval_id}: args_hash differs from the chain" in result.stdout
    assert "harvest" not in result.stdout


def test_check_store_catches_an_edited_outbox_recipient(place: Place) -> None:
    _flow(place)
    with Store(place.db, readonly=True) as store:
        email = list_outbox(store, KC)[0]
    _sql(
        place,
        "UPDATE outbox SET recipients = ? WHERE id = ?",
        (json.dumps(["harvest@exfil-drop.example"]), email.id),
    )
    result = place.cli("check-store", "--as", "kc-owner-dale")
    assert result.exit_code == 1
    assert result.stdout.splitlines()[0] == (
        f"MISMATCH email.send {email.id}: content differs from the chain"
    )
    assert "harvest" not in result.output


def test_check_store_refuses_a_broken_chain(place: Place) -> None:
    _flow(place)
    with closing(sqlite3.connect(place.audit)) as conn, conn:
        conn.execute("DELETE FROM events WHERE rowid = 3")
    result = place.cli("check-store", "--as", "kc-owner-dale")
    assert result.exit_code == 1 and "chain BROKEN" in result.stdout


def test_check_store_reports_rows_recorded_before_hashes(place: Place) -> None:
    _flow(place)
    events = [
        e.model_copy(update={"details": {k: v for k, v in e.details.items() if "hash" not in k}})
        for e in place.events()
    ]
    with Store(place.db, readonly=True) as store:
        result = check_store(store, events, KC)
    assert result.ok and len(result.unchecked) == 5


# The incident, from the log alone


def test_an_incident_is_reconstructed_from_the_audit_commands(place: Place) -> None:
    with place.open() as world:
        old_version = world.engines[KC].version
        staff = world.gate.start_run(KC, "kc-staff-nadine")
        approval_id = _approved_email(world, staff, _email(BEATRICE))
        manager = world.gate.start_run(KC, "kc-mgr-marisol")
        assert isinstance(
            world.gate.call_tool(manager, "parts.order", Action.SPEND, _order("480.00")), Executed
        )
        exfil = world.gate.call_tool(
            staff, "email.send", Action.SEND, _email(EXFIL, body=EXFIL_BODY)
        )
        assert isinstance(exfil, Denied)
    suspended = place.world_cli(
        "users", "suspend", "kc-staff-nadine", "--as", "kc-owner-dale", "--reason", "security"
    )
    assert suspended.exit_code == 0, suspended.output
    place.edit_policy('amount_usd_lte: "500.00"', 'amount_usd_lte: "300.00"')
    place.edit_policy('amount_usd_gt: "500.00"', 'amount_usd_gt: "300.00"')
    with place.open() as world:
        new_version = world.engines[KC].version
        manager = world.gate.start_run(KC, "kc-mgr-marisol")
        later = world.gate.call_tool(manager, "parts.order", Action.SPEND, _order("480.00"))
        assert isinstance(later, PendingApproval)
    assert new_version != old_version

    shown = place.cli("show", "--as", "kc-owner-dale", "--json")
    assert shown.exit_code == 0
    events = json.loads(shown.stdout)

    def find(kind: str, **match: Any) -> list[dict[str, Any]]:
        return [
            e
            for e in events
            if e["kind"] == kind
            and all(e.get(k) == v or e["details"].get(k) == v for k, v in match.items())
        ]

    # Who asked, for which tool, to how many recipients, approved by whom, with which args.
    [asked] = find("approval_required", subject=approval_id)
    assert (asked["user_id"], asked["tool"], asked["policy_version"]) == (
        "kc-staff-nadine",
        "email.send",
        old_version,
    )
    assert (asked["details"]["recipients"], asked["details"]["recipients_external"]) == (1, 1)
    [approved] = find("approved", subject=approval_id)
    assert approved["user_id"] == "kc-mgr-terrence"
    [ran] = find("executed", subject=approval_id)
    assert ran["details"]["args_hash"] == asked["details"]["args_hash"]
    assert ran["details"]["approval_args_hash"] == asked["details"]["approval_args_hash"]
    assert ran["details"]["effect_ref"].startswith("email-")
    # The 480 USD order: who, how much, under which policy; and the same order later.
    [order] = find("executed", tool="parts.order")
    assert (order["user_id"], order["details"]["amount_usd"], order["policy_version"]) == (
        "kc-mgr-marisol",
        "480.00",
        old_version,
    )
    [queued] = find("approval_required", tool="parts.order")
    assert (queued["details"]["amount_usd"], queued["policy_version"]) == ("480.00", new_version)
    # The denied exfiltration: who, which rule outcome, how many outside recipients.
    [denied] = find("deny", tool="email.send")
    assert denied["user_id"] == "kc-staff-nadine" and denied["reason"] == "default_deny"
    assert denied["details"]["recipients_external"] == 1
    # The suspension: who, whom, why.
    [suspension] = find("operator_action", action="user_suspended")
    assert (suspension["user_id"], suspension["subject"], suspension["details"]["reason"]) == (
        "kc-owner-dale",
        "kc-staff-nadine",
        "security",
    )
    # Both policy versions were loaded, and each says what it allowed.
    assert find("policy_loaded", policy_version=old_version)
    assert find("policy_loaded", policy_version=new_version)
    for version, limit in ((old_version, '"500"'), (new_version, '"300"')):
        policy = place.cli("policy", version, "--as", "kc-owner-dale")
        assert policy.exit_code == 0, policy.output
        assert f'"amount_usd_lte": {limit}' in policy.stdout
    text = place.cli("show", "--as", "kc-owner-dale").stdout
    assert f"subject={approval_id}" in text and "recipients_external=1" in text
    assert place.cli("check-store", "--as", "kc-owner-dale").exit_code == 0

    # Nothing anyone wrote reached the log: no subject, body, address or probe.
    raw = _log_bytes(place)
    assert [s for s in NEVER_LOGGED if s.encode() in raw] == []


# Store and tools


def test_a_read_only_store_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Store(tmp_path / "missing.db", readonly=True)
    with closing(sqlite3.connect(tmp_path / "other.db")) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    with pytest.raises(StoreError):
        Store(tmp_path / "other.db", readonly=True)
    Store(tmp_path / "store.db").close()
    with Store(tmp_path / "store.db", readonly=True) as store:
        assert list_outbox(store) == []
        assert store.get_policy_snapshot("sha256:0") is None
        with pytest.raises(sqlite3.OperationalError):
            store.suspend_user("anyone")


def test_reading_side_effects_creates_no_table() -> None:
    with Store() as store:
        assert stored_effects(store, KC) == []
        tables = {r[0] for r in store.connection.execute("SELECT name FROM sqlite_master")}
        assert "outbox" not in tables
