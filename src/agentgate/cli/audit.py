"""`agentgate audit`: verify the log, show events and policies, check the store."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from agentgate.audit.base import (
    AuditEvent,
    EventKind,
    Reference,
    ReferencedField,
)
from agentgate.audit.panoptes import read_events, verify_log
from agentgate.cli._shared import (
    DEMO_IDENTITY_NOTE,
    AsOption,
    AuditDbOption,
    database_error,
    fail,
)
from agentgate.domain import Role, User, UserStatus
from agentgate.gate import GateReason, store_audit_reference
from agentgate.policy.engine import Reason
from agentgate.store import Store
from agentgate.world import (
    DEFAULT_AUDIT_DB,
    DEFAULT_DB,
    StoreCheck,
    bundle_sha256,
    check_store,
)

audit_app = typer.Typer(help="Inspect the audit log.", no_args_is_help=True)


@audit_app.command("verify")
def verify(
    db: Annotated[Path, typer.Option("--db", help="Panoptes audit database to check.")] = Path(
        "data/audit.db"
    ),
) -> None:
    """Check the audit log's hash chain. Exits 1 if any stored event was changed, or any
    event is outside the chain. Read-only: a snapshot of the file is verified."""
    try:
        # A mistyped path fails here; nothing is created or changed.
        result, count = verify_log(db)
    except sqlite3.Error as exc:
        fail(database_error(exc))
    except (OSError, ValueError, RuntimeError) as exc:
        fail(str(exc))
    if result["valid"]:
        typer.echo(f"{db}  {count} events  chain intact")
        return
    line = f"{db}  {count} events  chain BROKEN: {result['error']}"
    if result.get("broken_event_id"):
        line += f" at event {result['broken_event_id']}"
    typer.echo(line)
    raise typer.Exit(code=1)


def _reading_user(store: Store, user_id: str, *, owners_only: bool) -> User:
    """The acting user of a read-only audit command: active, an owner (or a manager when
    owners_only is false)."""
    user = store.get_user(user_id)
    if user is None:
        fail(f"unknown user {user_id!r}")
    if user.status is not UserStatus.ACTIVE:
        fail(f"user {user_id!r} is not active")
    allowed = {Role.OWNER} if owners_only else {Role.OWNER, Role.MANAGER}
    if user.role not in allowed:
        who = "owners only" if owners_only else "owners and managers only"
        fail(f"role {user.role.value!r} may not read the audit log; {who}")
    return user


@contextmanager
def _audit_reader(
    db: Path, audit_db: Path, as_user: str, *, owners_only: bool
) -> Iterator[tuple[Store, User, list[AuditEvent]]]:
    """The store opened read-only, the acting user and every event in the log."""
    typer.echo(DEMO_IDENTITY_NOTE, err=True)
    try:
        store = Store(db, readonly=True)
    except sqlite3.Error as exc:
        fail(database_error(exc))
    except (OSError, ValueError) as exc:
        fail(str(exc))
    try:
        user = _reading_user(store, as_user, owners_only=owners_only)
        try:
            events = read_events(audit_db)
        except (OSError, ValueError) as exc:
            fail(str(exc))
        yield store, user, events
    except sqlite3.Error as exc:
        fail(database_error(exc))
    finally:
        store.close()


def _shown(value: str) -> str:
    """A stored value as one line of plain ASCII: anything else is shown escaped."""
    if value.isascii() and value.isprintable() and " " not in value:
        return value
    return json.dumps(value, ensure_ascii=True)


def _shown_detail(value: JsonValue) -> str:
    if isinstance(value, str):
        return _shown(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ",".join(_shown(str(item)) for item in value) or "[]"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _field(event: AuditEvent, name: ReferencedField) -> str:
    value = getattr(event, name)
    if value is None:
        return "-"
    reference = event.references.get(name)
    if reference is not None:
        return f"{reference.ref} ({reference.why} value)"
    return _shown(str(value))


def _event_line(event: AuditEvent) -> str:
    stamp = event.timestamp.astimezone(UTC).isoformat(timespec="milliseconds")
    parts = [
        stamp.replace("+00:00", "Z"),
        event.kind.value,
        _shown(event.reason),
        f"user={_field(event, 'user_id')}",
        f"role={'-' if event.role is None else event.role.value}",
        f"tool={_field(event, 'tool')}",
        f"subject={_field(event, 'subject')}",
        f"rules={','.join(_shown(r) for r in event.rule_ids) or '-'}",
        f"policy={_shown(event.policy_version) if event.policy_version else '-'}",
    ]
    if event.run_id is not None:
        parts.append(f"run={_field(event, 'run_id')}")
    parts += [f"{key}={_shown_detail(value)}" for key, value in sorted(event.details.items())]
    return "  ".join(parts)


def _tenant_view(store: Store, event: AuditEvent) -> AuditEvent:
    """The event as its own customer's readers see it.

    A denial naming another customer's document is shown as a missing document is recorded:
    its reason, id and args hash would otherwise tell a tenant which ids exist elsewhere.
    """
    if event.kind is not EventKind.DENY or event.reason != Reason.DOCUMENT_CUSTOMER_MISMATCH:
        return event
    if event.subject is None:
        return event
    # Every reference is the same keyed hash of its value, so an id the log already holds
    # as a reference keeps it.
    held = event.references.get("subject")
    ref = held.ref if held is not None else store_audit_reference(store, event.subject)
    return event.model_copy(
        update={
            "reason": GateReason.DOCUMENT_NOT_FOUND.value,
            "subject": ref,
            "rule_ids": (),
            "details": {},
            "references": {**event.references, "subject": Reference(ref=ref, why="unknown")},
        }
    )


@audit_app.command("show")
def audit_show(
    as_user: AsOption,
    db: Annotated[Path, typer.Option("--db", help="agentgate database, read-only.")] = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
    run: Annotated[
        str | None, typer.Option("--run", metavar="RUN_ID", help="Only this run's events.")
    ] = None,
    customer: Annotated[
        str | None,
        typer.Option(
            "--customer", metavar="ID", help="Must be the acting user's own customer, if given."
        ),
    ] = None,
    kind: Annotated[
        EventKind | None, typer.Option("--kind", help="Only events of this kind.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Only the last N matching events.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the stored events as JSON instead.")
    ] = False,
) -> None:
    """Print the acting user's customer's audit events, oldest first, one per line.

    Owners and managers only. Read-only: neither file is changed. A value the log holds as a
    reference prints as its ref: and why ("unknown value": it named nothing the store knew).
    A denial naming another customer's document prints as a missing document, as the model
    was told; the log itself keeps the real reason for the operator.
    Run `agentgate audit verify` to check the chain itself."""
    with _audit_reader(db, audit_db, as_user, owners_only=False) as (store, user, events):
        # Another customer reads exactly like a missing one.
        if customer is not None and customer != user.customer_id:
            fail(f"customer {customer!r} not found")
        shown = [
            _tenant_view(store, event)
            for event in events
            if event.customer_id == user.customer_id
            and (run is None or event.run_id == run)
            and (kind is None or event.kind is kind)
        ]
    if limit is not None:
        shown = shown[-limit:]
    if as_json:
        typer.echo(
            json.dumps(
                [event.model_dump(mode="json") for event in shown],
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
        )
        return
    for event in shown:
        typer.echo(_event_line(event))
    typer.echo(f"{user.customer_id}  {len(shown)} events")


@audit_app.command("policy")
def audit_policy(
    version: Annotated[str, typer.Argument(metavar="VERSION", help="A policy version hash.")],
    as_user: AsOption,
    db: Annotated[Path, typer.Option("--db", help="agentgate database, read-only.")] = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
) -> None:
    """Print the policy bundle stored for a version, and check it against the chain.

    The bundle is the canonical JSON the version is computed over. Its sha256 must match
    the snapshot row, the version, and every policy_loaded event for it on the chain.
    Exits 1 on any mismatch. Owners and managers only; read-only."""
    with _audit_reader(db, audit_db, as_user, owners_only=False) as (store, user, events):
        snapshot = store.get_policy_snapshot(version) if version.isprintable() else None
        loaded = [
            event
            for event in events
            if event.kind is EventKind.POLICY_LOADED
            and event.customer_id == user.customer_id
            and event.policy_version == version
        ]
        # A version is the hash of the bundle alone, so two customers with the same rules
        # share one snapshot; the customer's own chain shows it loaded that version. Any
        # other customer's version reads exactly like a missing one.
        if snapshot is None or (snapshot.customer_id != user.customer_id and not loaded):
            fail(f"policy version {version!r} not found")
    digest = bundle_sha256(snapshot.bundle)
    try:
        bundle = json.loads(snapshot.bundle)
    except ValueError:
        bundle = None
    problems: list[str] = []
    if bundle is None:
        problems.append("the stored bundle is not JSON")
    else:
        typer.echo(json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=True))
    if digest != snapshot.sha256:
        problems.append("the stored bundle does not match its stored sha256")
    if "sha256:" + digest[:16] != version:
        problems.append("the stored bundle does not hash to its version")
    recorded = [event.details.get("bundle_sha256") for event in loaded]
    if any(value != digest for value in recorded):
        problems.append("the stored bundle does not match a policy_loaded event")
    for problem in problems:
        typer.echo(f"MISMATCH {problem}")
    on_chain = f"{len(loaded)} policy_loaded events"
    if not loaded:
        on_chain += " (checked against the version only)"
    status = "ok" if not problems else "MISMATCH"
    typer.echo(f"policy {version}  {user.customer_id}  sha256 {digest}  {on_chain}  {status}")
    if problems:
        raise typer.Exit(code=1)


_CHECKED_KINDS = (
    ("approval", "approvals"),
    ("email.send", "outbox emails"),
    ("parts.order", "parts orders"),
    ("refund.issue", "refunds"),
    ("notes.write", "notes"),
)


@audit_app.command("check-store")
def audit_check_store(
    as_user: AsOption,
    db: Annotated[Path, typer.Option("--db", help="agentgate database, read-only.")] = DEFAULT_DB,
    audit_db: AuditDbOption = DEFAULT_AUDIT_DB,
) -> None:
    """Check the acting owner's customer's approval, outbox, parts order, refund and note
    rows against the audit chain. Exits 1 if any row does not match, or the chain is broken.

    The chain is verified first. Each approval row's args and args_hash must match the
    approval_args_hash its events carry, and each side-effect row must hash to the args_hash
    of the executed event that names it. Owners only; read-only."""
    with _audit_reader(db, audit_db, as_user, owners_only=True) as (store, user, events):
        try:
            verified, _ = verify_log(audit_db)
        except (OSError, ValueError, RuntimeError) as exc:
            fail(str(exc))
        if not verified["valid"]:
            typer.echo(f"{audit_db}  chain BROKEN: {verified['error']}")
            raise typer.Exit(code=1)
        result = check_store(store, events, user.customer_id)
    for line in _check_lines(result):
        typer.echo(line)
    if not result.ok:
        raise typer.Exit(code=1)


def _check_lines(result: StoreCheck) -> list[str]:
    lines = [f"MISMATCH {m.kind} {m.id}: {m.problem}" for m in result.mismatches]
    if result.unchecked:
        lines.append(
            f"note: {len(result.unchecked)} rows were recorded before the chain carried"
            " hashes; only their ids were checked"
        )
    counts = ", ".join(f"{result.checked.get(kind, 0)} {label}" for kind, label in _CHECKED_KINDS)
    lines.append(f"{result.customer_id}  {counts} checked  {len(result.mismatches)} mismatches")
    return lines
