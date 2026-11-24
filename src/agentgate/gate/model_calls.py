"""Model calls: a model request through the proxy, decided and priced before it is sent.

`admit` checks the run, reads the request strictly, lets the policy's models section decide,
reserves the request's worst-case cost and records the allow; `finish` ends that
reservation by what happened to the exchange. `charge` and `settle` are the in-process
reference agent's path, which trusts its host's estimate.
"""

import time
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal

from pydantic import JsonValue

from agentgate.audit.base import EventKind
from agentgate.context import RunContext
from agentgate.domain import Role
from agentgate.gate._values import plain_id
from agentgate.gate.outcomes import CapReached, Denied, GateReason, IdentityError
from agentgate.gate.recorder import AuditRecorder, Call
from agentgate.gate.runs import RunRegistry, bind
from agentgate.model.admission import Caps, ModelCall, ModelExchange
from agentgate.model.pricing import RateTable
from agentgate.policy.engine import ModelRequest, PolicyEngine, Reason
from agentgate.policy.schema import Effect
from agentgate.spend import Reservation, SpendLedger
from agentgate.store import Store


class ModelAdmission:
    """Admits model requests against policy and spend, and settles what they cost."""

    def __init__(
        self,
        *,
        store: Store,
        engines: Mapping[str, PolicyEngine],
        ledger: SpendLedger,
        runs: RunRegistry,
        recorder: AuditRecorder,
    ) -> None:
        self._store = store
        self._engines = engines
        self._ledger = ledger
        self._runs = runs
        self._recorder = recorder
        # Reservations admit made and finish has not ended. Only finish may end these, and
        # it may end nothing else, so a model call cannot be settled at an amount its
        # exchange did not give.
        self._gateway_reservations: set[str] = set()

    def admit(
        self,
        ctx: RunContext,
        call: ModelCall,
        *,
        rates: RateTable | None = None,
        caps: Caps | None = None,
    ) -> Reservation | Denied | CapReached:
        base = self._runs.begin(ctx, None, None)
        run = self._runs.check(ctx, base)
        if isinstance(run, Denied):
            return run
        asked = replace(
            bind(base, run),
            exchange_id=plain_id(call.exchange_id),
            correlation_id=plain_id(call.correlation_id),
        )
        values = _model_values(self._recorder, call)
        if not call.valid:
            return self._recorder.deny(asked, GateReason.MODEL_REQUEST_INVALID, details=values)

        if call.unknown_fields:
            return self._recorder.deny(asked, GateReason.MODEL_UNKNOWN_FIELD, details=values)

        decision = run.engine.decide_model(
            ModelRequest(
                customer_id=run.ctx.customer_id,
                role=run.ctx.role,
                provider=call.provider,
                model=call.model,
                max_tokens=call.max_tokens,
                n=call.n,
                request_bytes=call.request_bytes,
                server_features=bool(call.server_features),
            )
        )
        listed = run.engine.model_entry(call.provider, call.model) is not None
        # A model name is the caller's value: it is shown as written only if the policy lists it.
        subject = call.model or None
        unknown = None if listed else subject
        if decision.effect is not Effect.ALLOW:
            return self._recorder.deny(
                asked,
                decision.reason.value,
                subject=subject,
                unknown_subject=unknown,
                rule_ids=decision.rule_ids,
                policy_version=decision.policy_version,
                details=values,
            )

        models = run.policy.models
        if models is None or call.max_tokens is None:
            # The engine allows only a request that declares its limit, under a policy with
            # models; this keeps a bug there from becoming an unpriced reservation.
            reason = (
                Reason.MODEL_NOT_ALLOWED if models is None else Reason.MODEL_MAX_TOKENS_REQUIRED
            )
            return self._recorder.deny(
                asked,
                reason,
                subject=subject,
                rule_ids=decision.rule_ids,
                policy_version=decision.policy_version,
                details=values,
            )
        # The table's worst case is a floor the policy's fixed reserve cannot lower: a tiny
        # fixed reserve would otherwise let parallel full-size calls slip under the run cap.
        # The fixed amount only stands in for a model the table does not price.
        computed = (
            None
            if rates is None
            else rates.worst_case_usd(call.model, call.max_tokens, call.request_bytes)
        )
        if rates is not None:
            values["rate_table_version"] = rates.version
        fixed = models.reserve if isinstance(models.reserve, Decimal) else None
        amount = fixed if computed is None else computed if fixed is None else max(fixed, computed)
        basis = "computed" if amount == computed else "fixed"
        if amount is None:
            return self._recorder.deny(
                asked,
                GateReason.MODEL_UNPRICED,
                subject=subject,
                rule_ids=decision.rule_ids,
                policy_version=decision.policy_version,
                details=values,
            )

        limits = run.policy.spend if caps is None else caps.tighten(run.policy.spend)
        reserved = self._runs.reserve(run, asked, amount, details=values, limits=limits)
        if isinstance(reserved, CapReached):
            return reserved
        try:
            # The allow is on record before the request is forwarded, or it is not forwarded.
            self._recorder.emit(
                asked,
                EventKind.ALLOW,
                decision.reason.value,
                subject=subject,
                rule_ids=decision.rule_ids,
                policy_version=decision.policy_version,
                decision=True,
                details={
                    **values,
                    "reserved_usd": str(amount),
                    "reserve": basis,
                    "reservation_id": reserved.id,
                },
            )
        except Exception:
            self._ledger.release(reserved)
            return Denied(GateReason.AUDIT_UNAVAILABLE.value)
        self._gateway_reservations.add(reserved.id)
        return reserved

    def finish(self, ctx: RunContext, reservation: Reservation, exchange: ModelExchange) -> Decimal:
        if reservation.run_id != ctx.run_id:
            raise ValueError("reservation belongs to another run")
        if reservation.id not in self._gateway_reservations:
            raise ValueError("not a reservation admit_model made, or already finished")
        released = not exchange.request_sent or (
            exchange.provider_error and not exchange.produced_output
        )
        if released:
            self._ledger.release(reservation)
            charge = Decimal(0)
        else:
            if exchange.complete and exchange.cost_usd is not None:
                charge = exchange.cost_usd
            else:
                charge = reservation.amount_usd
            self._ledger.settle(reservation, charge)
        self._gateway_reservations.discard(reservation.id)
        # The money is settled first. If the event cannot be recorded the error propagates,
        # and the charge stands.
        self._record_model_end(reservation, exchange, charge, settled=not released)
        return charge

    def charge(self, ctx: RunContext, estimated_usd: Decimal) -> Reservation | CapReached:
        call = self._runs.begin(ctx, None, None)
        run = self._runs.check(ctx, call)
        if isinstance(run, Denied):
            raise IdentityError(run.reason, "run or user is not valid for model calls")
        call = bind(call, run)
        reserved = self._runs.reserve(run, call, estimated_usd, details={"source": "model"})
        return reserved

    def settle(self, ctx: RunContext, reservation: Reservation, actual_usd: Decimal) -> None:
        if reservation.run_id != ctx.run_id:
            raise ValueError("reservation belongs to another run")
        if reservation.id in self._gateway_reservations:
            raise ValueError("a gateway reservation is ended by finish_model")
        self._ledger.settle(reservation, actual_usd)

    def refuse_token(
        self, run: RunContext, token_reason: str, *, exchange_id: str | None = None
    ) -> Denied:
        base = self._runs.begin(run, None, None)
        checked = self._runs.check(run, base)
        if isinstance(checked, Denied):
            return checked
        asked = replace(bind(base, checked), exchange_id=plain_id(exchange_id))
        return self._recorder.deny(
            asked,
            GateReason.TOKEN_REFUSED,
            details={"source": "model", "refused_because": token_reason},
        )

    def _record_model_end(
        self, reservation: Reservation, exchange: ModelExchange, charge: Decimal, *, settled: bool
    ) -> None:
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (reservation.run_id,)
            ).fetchone()
        if row is None:
            return
        engine = self._engines.get(row["customer_id"])
        call = Call(
            time.perf_counter_ns(),
            row["customer_id"],
            user_id=row["user_id"],
            role=Role(row["role"]),
            run_id=row["run_id"],
            policy_version=None if engine is None else engine.version,
            exchange_id=plain_id(exchange.exchange_id),
        )
        self._recorder.emit(
            call,
            EventKind.EXECUTED,
            GateReason.MODEL_SETTLED if settled else GateReason.MODEL_RELEASED,
            details={
                "source": "model",
                "reservation_id": reservation.id,
                "charged_usd": str(charge),
                "reserved_usd": str(reservation.amount_usd),
                "request_sent": exchange.request_sent,
                "provider_error": exchange.provider_error,
                "complete": exchange.complete,
                "produced_output": exchange.produced_output,
                "priced": exchange.cost_usd is not None,
            },
        )


def _model_values(recorder: AuditRecorder, call: ModelCall) -> dict[str, JsonValue]:
    """What the events of a model call record about it: counts and ids, never the request's
    text. The correlation id is the client's claim, so it is recorded as asserted and only
    in a plain shape; any other is recorded as a reference."""
    values: dict[str, JsonValue] = {
        "source": "model",
        "provider": call.provider.value,
        "request_bytes": call.request_bytes,
    }
    if call.valid:
        values["max_tokens"] = call.max_tokens
        values["n"] = call.n
        if call.unknown_fields:
            values["unknown_fields"] = call.unknown_fields
        if call.server_features:
            values["server_features"] = list(call.server_features)
    asserted = call.correlation_id
    if asserted is not None and call.correlation_source == "client":
        if plain_id(asserted) is not None:
            values["asserted_correlation_id"] = asserted
        else:
            values["asserted_correlation_ref"] = recorder.reference(asserted)
    return values
