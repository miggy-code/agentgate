"""The gate's audit side: every event it records, and what it records in place of values.

Every outcome becomes at least one audit event carrying ids, labels, counts and reason
codes, never document bodies, message text or argument values.

Values a caller chose (a document or approval id, a tool name, a claimed run, user or
customer, an approver's name) can be probes: a guessed slug that names a person, or text
aimed at whoever reads the log. Only ids the store knows go into the durable audit log as
written. Every other caller value, and any real id the log would refuse (Panoptes refuses
secret-shaped and control-character identifiers), is attached to the event as a Reference,
`ref:` plus the first 16 hex characters of HMAC-SHA256 under a key kept in this store, and
the durable sink stores the reference instead. The same value always gives the same
reference, so repeated probes can be counted and an operator holding the store can check a
value with `reference`; a reader of the audit log alone cannot test guesses against it, and
deleting the key leaves past references pointing at nothing. If a sink still refuses an
event for its content, a reference-only version of it is recorded instead, so a refusal
never turns into an unrecorded exception; only a sink that is down raises.
"""

import hashlib
import hmac
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import NoReturn

from pydantic import JsonValue

from agentgate.audit.base import (
    REFERENCED_FIELDS,
    AuditEvent,
    AuditRejected,
    AuditSink,
    EventKind,
    Reference,
    ReferencedField,
)
from agentgate.audit.panoptes import storable_identifier
from agentgate.clock import Clock
from agentgate.domain import Action, Role
from agentgate.gate._values import (
    UNKNOWN_CUSTOMER,
    get_customer,
    get_user,
    is_lookupable,
    plain_id,
)
from agentgate.gate.outcomes import Denied, GateReason, IdentityError
from agentgate.store import Store

# The key audit references are computed with. One row, made once per store database.
_KEY_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_key (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    key BLOB NOT NULL
) STRICT
"""
REFERENCE_PREFIX = "ref:"
# Hex characters of the HMAC kept in a reference: 64 bits, far from colliding at audit volumes.
REFERENCE_HEX = 16


@dataclass(frozen=True)
class Call:
    """Who and what one gate call is about, for its audit events."""

    started_ns: int
    customer_id: str
    user_id: str | None = None
    role: Role | None = None
    run_id: str | None = None
    tool: str | None = None
    action: Action | None = None
    policy_version: str | None = None
    # Fields still holding the caller's claim, not yet confirmed against the store.
    claimed: frozenset[str] = frozenset()
    # The tool name the caller asked for, as given, when it differs from `tool` or may be
    # unknown; None when the tool came from the gate's own records.
    asked_tool: str | None = None
    # For a model request through the proxy: the ids that join its events to the exchange.
    exchange_id: str | None = None
    correlation_id: str | None = None

    def latency_us(self) -> int:
        return (time.perf_counter_ns() - self.started_ns) // 1000


class AuditRecorder:
    """Records the gate's events to a sink, with caller values as references."""

    def __init__(
        self, *, store: Store, sink: AuditSink, clock: Clock, known_tools: frozenset[str]
    ) -> None:
        self._store = store
        self._sink = sink
        self._clock = clock
        self._known_tools = known_tools
        self._key = _load_audit_key(store)

    def reference(self, value: str) -> str:
        """The reference the durable audit log stores for this value in place of it."""
        return _reference(self._key, value)

    def identity_denied(
        self,
        call: Call,
        reason: GateReason,
        message: str,
        *,
        details: dict[str, JsonValue] | None = None,
    ) -> NoReturn:
        self.emit(call, EventKind.DENY, reason, decision=True, details=details)
        raise IdentityError(reason.value, message)

    def deny(
        self,
        call: Call,
        reason: str,
        *,
        model_reason: str | None = None,
        subject: str | None = None,
        unknown_subject: str | None = None,
        rule_ids: tuple[str, ...] = (),
        policy_version: str | None = None,
        details: dict[str, JsonValue] | None = None,
    ) -> Denied:
        self.emit(
            call,
            EventKind.DENY,
            reason,
            subject=subject,
            unknown_subject=unknown_subject,
            rule_ids=rule_ids,
            policy_version=policy_version,
            decision=True,
            details=details,
        )
        return Denied(str(model_reason or reason))

    def emit(
        self,
        call: Call,
        kind: EventKind,
        reason: str,
        *,
        subject: str | None = None,
        unknown_subject: str | None = None,
        rule_ids: tuple[str, ...] = (),
        policy_version: str | None = None,
        decision: bool = False,
        latency_us: int | None = None,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        """Record one event. `unknown_subject` is a caller's value that names nothing the
        store knows; it is recorded only as a reference, whatever `subject` shows."""
        if latency_us is None and decision:
            latency_us = call.latency_us()
        event = AuditEvent(
            timestamp=self._clock.now(),
            kind=kind,
            customer_id=call.customer_id,
            user_id=call.user_id,
            role=call.role,
            run_id=call.run_id,
            tool=call.tool,
            action=call.action,
            subject=subject,
            reason=str(reason),
            rule_ids=rule_ids,
            policy_version=policy_version or call.policy_version,
            latency_us=latency_us,
            details=details or {},
            exchange_id=call.exchange_id,
            correlation_id=call.correlation_id,
            references=self._references(call, subject, unknown_subject),
        )
        try:
            self._sink.record(event)
        except AuditRejected:
            # The sink refused the event's content and stored nothing. Its reference-only
            # version is recorded instead; if that fails too, the sink is down and its
            # error propagates, as it does for any other event.
            self._sink.record(self._reference_only(event))

    def effect_ref(self, value: object) -> JsonValue:
        """A tool's effect_ref as an event records it: the id, or a reference to anything
        that is not one plain id."""
        if isinstance(value, str) and plain_id(value) is not None:
            return value
        text = value if isinstance(value, str) else repr(type(value))
        return self.reference(text)

    def _references(
        self, call: Call, subject: str | None, unknown_subject: str | None
    ) -> dict[ReferencedField, Reference]:
        """The fields of an event about `call` that the durable log stores as references."""
        refs: dict[ReferencedField, Reference] = {}
        claimed: tuple[tuple[ReferencedField, str | None], ...] = (
            ("customer_id", call.customer_id),
            ("user_id", call.user_id),
            ("run_id", call.run_id),
        )
        for name, value in claimed:
            if value is not None:
                known = name not in call.claimed or self._knows(name, value)
                self._reference_unless(refs, name, value, known)
        asked = call.asked_tool
        if asked is not None:
            self._reference_unless(refs, "tool", asked, asked in self._known_tools)
        elif call.tool is not None:
            self._reference_unless(refs, "tool", call.tool, True)
        if unknown_subject is not None:
            self._reference_unless(refs, "subject", unknown_subject, False)
        elif subject is not None:
            self._reference_unless(refs, "subject", subject, True)
        return refs

    def _reference_unless(
        self,
        refs: dict[ReferencedField, Reference],
        name: ReferencedField,
        value: str,
        known: bool,
    ) -> None:
        if not known:
            refs[name] = Reference(ref=self.reference(value), why="unknown")
        elif not storable_identifier(value):
            refs[name] = Reference(ref=self.reference(value), why="unstorable")

    def _knows(self, name: str, value: str) -> bool:
        """Whether a claimed customer, user or run id names a row the store holds."""
        if name == "customer_id":
            return value == UNKNOWN_CUSTOMER or get_customer(self._store, value) is not None
        if name == "user_id":
            return get_user(self._store, value) is not None
        if not is_lookupable(value):
            return False
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (value,)
            ).fetchone()
        return row is not None

    def _reference_only(self, event: AuditEvent) -> AuditEvent:
        """The event with every id a caller or the configuration supplied as a reference,
        and no details: what is recorded when a sink refuses the event as it was."""
        recorded = event.recorded()
        refs: dict[ReferencedField, Reference] = dict(event.references)
        for name in REFERENCED_FIELDS:
            value = getattr(recorded, name)
            if value is not None and name not in refs:
                refs[name] = Reference(ref=self.reference(value), why="unstorable")
        version = event.policy_version
        return AuditEvent(
            timestamp=event.timestamp,
            kind=event.kind,
            customer_id=refs["customer_id"].ref,
            user_id=None if "user_id" not in refs else refs["user_id"].ref,
            role=event.role,
            run_id=None if "run_id" not in refs else refs["run_id"].ref,
            tool=None if "tool" not in refs else refs["tool"].ref,
            action=event.action,
            subject=None if "subject" not in refs else refs["subject"].ref,
            reason=event.reason if storable_identifier(event.reason) else "audit_refused",
            rule_ids=tuple(self.reference(rule_id) for rule_id in event.rule_ids),
            policy_version=(
                version
                if version is None or storable_identifier(version)
                else self.reference(version)
            ),
            latency_us=event.latency_us,
            details={"reference_only": True},
            references=refs,
        )


def store_audit_reference(store: Store, value: str) -> str:
    """The reference a gate on this store records for value, for readers without a gate."""
    return _reference(_load_audit_key(store), value)


def _reference(key: bytes, value: str) -> str:
    digest = hmac.new(key, value.encode("utf-8", "surrogatepass"), hashlib.sha256).hexdigest()
    return REFERENCE_PREFIX + digest[:REFERENCE_HEX]


def _load_audit_key(store: Store) -> bytes:
    """The store's audit reference key, made on first use.

    Read first, so a read-only store that already has its key opens without a write. A
    read-only store from before keys existed gets a key for this process only: references
    then hold within the process, and the store gets its own key once it is opened writable.
    """
    select = "SELECT key FROM audit_key WHERE id = 1"
    with store.locked():
        try:
            row = store.connection.execute(select).fetchone()
        except sqlite3.OperationalError:
            row = None
    if row is not None:
        return bytes(row["key"])
    try:
        with store.locked():
            store.connection.execute(_KEY_SCHEMA)
        with store.transaction():
            # INSERT OR IGNORE, so gates opening the same database at once agree on one key.
            store.connection.execute(
                "INSERT OR IGNORE INTO audit_key (id, key) VALUES (1, ?)",
                (secrets.token_bytes(32),),
            )
            row = store.connection.execute(select).fetchone()
    except sqlite3.OperationalError:
        return secrets.token_bytes(32)
    return bytes(row["key"])
