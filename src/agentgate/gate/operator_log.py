"""Operator records: what is not a tool call but changes what the log means.

The policy a world loaded (`record_policy_loaded`) and operator actions
(`record_operator_action`, `record_token_event`, `release_stale`): fixture syncs,
suspensions, reinstatements, run tokens and stale spend sweeps.
"""

import sqlite3
import time
from collections.abc import Mapping
from datetime import timedelta
from decimal import Decimal

from pydantic import JsonValue

from agentgate.audit.base import EventKind, OperatorAction
from agentgate.domain import User
from agentgate.gate.outcomes import GateReason
from agentgate.gate.recorder import AuditRecorder, Call
from agentgate.model.tokens import TokenRecord
from agentgate.policy.engine import PolicyEngine
from agentgate.spend import SpendLedger
from agentgate.store import Store

# IN lists of the stale sweep's follow-up query are sent in chunks of this many ids.
_ID_CHUNK = 500


class OperatorLog:
    """Records policy loads and operator actions, and the stale spend sweep."""

    def __init__(
        self,
        *,
        store: Store,
        engines: Mapping[str, PolicyEngine],
        ledger: SpendLedger,
        recorder: AuditRecorder,
    ) -> None:
        self._store = store
        self._engines = engines
        self._ledger = ledger
        self._recorder = recorder

    def record_policy_loaded(
        self, customer_id: str, *, bundle_sha256: str, details: Mapping[str, JsonValue]
    ) -> None:
        engine = self._engines.get(customer_id)
        if engine is None:
            raise ValueError(f"no policy for customer {customer_id!r}")
        call = Call(time.perf_counter_ns(), customer_id, policy_version=engine.version)
        self._recorder.emit(
            call,
            EventKind.POLICY_LOADED,
            GateReason.POLICY_LOADED,
            subject=engine.version,
            details={**details, "bundle_sha256": bundle_sha256},
        )

    def record_operator_action(
        self,
        action: OperatorAction,
        *,
        customer_id: str,
        actor: User | None,
        target: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        engine = self._engines.get(customer_id)
        own = actor is not None and actor.customer_id == customer_id
        call = Call(
            time.perf_counter_ns(),
            customer_id,
            user_id=actor.id if own and actor is not None else None,
            role=actor.role if own and actor is not None else None,
            policy_version=None if engine is None else engine.version,
        )
        recorded: dict[str, JsonValue] = {
            **(details or {}),
            "action": action.value,
            "actor": "user" if own else ("system" if actor is None else "operator"),
        }
        self._recorder.emit(
            call,
            EventKind.OPERATOR_ACTION,
            action.value,
            subject=target,
            details=recorded,
        )

    def record_token_event(
        self,
        action: OperatorAction,
        token: TokenRecord,
        *,
        customer_id: str,
        actor: User | None,
    ) -> None:
        details: dict[str, JsonValue] = {
            "run_id": token.run_id,
            "expires_at": token.expires_at.isoformat(),
        }
        if action is OperatorAction.TOKEN_REVOKED and token.revoked_reason is not None:
            details["reason"] = token.revoked_reason
        self.record_operator_action(
            action, customer_id=customer_id, actor=actor, target=token.token_id, details=details
        )

    def release_stale(
        self, older_than: timedelta, *, actor: User | None = None, record: bool = True
    ) -> int:
        if not record:
            return self._ledger.release_stale(older_than)
        conn = self._store.connection
        with self._store.locked():
            open_ids = [
                str(row["id"])
                for row in conn.execute(
                    "SELECT id FROM spend_reservations WHERE status = 'open'"
                ).fetchall()
            ]
        released = self._ledger.release_stale(older_than)
        # The rows open before the sweep that it reclaimed (a reclaimed row keeps its
        # reclaimed_at even if a late settle charges it afterwards).
        rows: list[sqlite3.Row] = []
        for start in range(0, len(open_ids) if released else 0, _ID_CHUNK):
            chunk = open_ids[start : start + _ID_CHUNK]
            marks = ", ".join("?" for _ in chunk)
            with self._store.locked():
                rows += conn.execute(
                    "SELECT id, run_id, customer_id, reserved_usd FROM spend_reservations"
                    f" WHERE reclaimed_at IS NOT NULL AND id IN ({marks})",
                    chunk,
                ).fetchall()
        by_customer: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_customer.setdefault(str(row["customer_id"]), []).append(row)
        if actor is not None:
            by_customer.setdefault(actor.customer_id, [])
        minutes = int(older_than.total_seconds() // 60)
        for customer_id in sorted(by_customer):
            group = by_customer[customer_id]
            total = sum((Decimal(row["reserved_usd"]) for row in group), Decimal(0))
            run_ids: list[JsonValue] = [*sorted({str(row["run_id"]) for row in group})]
            self.record_operator_action(
                OperatorAction.STALE_RESERVATIONS_RELEASED,
                customer_id=customer_id,
                actor=actor,
                details={
                    "released": len(group),
                    "released_usd": str(total),
                    "run_ids": run_ids,
                    "older_than_minutes": minutes,
                },
            )
        return released
