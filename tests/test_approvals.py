import hashlib
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import JsonValue

from agentgate.approvals import (
    Approval,
    ApprovalError,
    ApprovalErrorReason,
    ApprovalQueue,
    ApprovalStatus,
    args_hash,
)
from agentgate.clock import FixedClock
from agentgate.models import Action, Role, User, UserStatus
from agentgate.policy.schema import ApprovalSettings
from agentgate.store import Store

START = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
TTL = timedelta(minutes=30)
TOOL = "email.send"
ARGS: dict[str, JsonValue] = {
    "to": ["pat@client.example"],
    "subject": "Quote",
    "body": "Hi Pat, the quote is attached.",
}
SETTINGS = ApprovalSettings(
    approver_roles=frozenset({Role.OWNER, Role.MANAGER}), ttl_minutes=30, risk_threshold=50
)


def _user(
    user_id: str,
    role: Role = Role.MANAGER,
    customer_id: str = "acme",
    status: UserStatus = UserStatus.ACTIVE,
) -> User:
    return User(
        id=user_id,
        customer_id=customer_id,
        name=user_id.title(),
        email=f"{user_id}@{customer_id}.example",
        role=role,
        status=status,
    )


MANAGER = _user("maya")
REQUESTER = _user("sam", role=Role.STAFF)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(START)


@pytest.fixture
def queue(store: Store, clock: FixedClock) -> ApprovalQueue:
    return ApprovalQueue(store, clock)


def _create(
    queue: ApprovalQueue,
    *,
    customer_id: str = "acme",
    requested_by: str = "sam",
    args: dict[str, JsonValue] | None = None,
    ttl: timedelta = TTL,
) -> Approval:
    return queue.create(
        customer_id=customer_id,
        run_id="run-1",
        requested_by=requested_by,
        tool=TOOL,
        action=Action.SEND,
        args=ARGS if args is None else args,
        risk_score=70,
        rule_ids=["external-email-needs-approval", "high-risk"],
        policy_version="sha256:0123456789abcdef",
        ttl=ttl,
    )


def _approved(queue: ApprovalQueue, **kwargs: object) -> Approval:
    approval = _create(queue, **kwargs)  # type: ignore[arg-type]
    return queue.approve(approval.id, MANAGER, SETTINGS)


def _claim(
    queue: ApprovalQueue,
    approval_id: str,
    *,
    customer_id: str = "acme",
    tool: str = TOOL,
    action: Action = Action.SEND,
    args: dict[str, JsonValue] | None = None,
) -> Approval:
    return queue.claim_for_execution(
        approval_id,
        customer_id=customer_id,
        tool=tool,
        action=action,
        args=ARGS if args is None else args,
    )


def _reason(exc: pytest.ExceptionInfo[ApprovalError]) -> ApprovalErrorReason:
    return exc.value.reason


# args_hash


def test_args_hash_format_and_canonical_form() -> None:
    digest = args_hash(TOOL, Action.SEND, ARGS)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    int(digest.removeprefix("sha256:"), 16)
    canonical = json.dumps(
        {"tool": TOOL, "action": "send", "args": ARGS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert digest == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def test_key_order_does_not_change_the_hash() -> None:
    reordered = {"body": ARGS["body"], "subject": ARGS["subject"], "to": ARGS["to"]}
    nested_a: dict[str, JsonValue] = {"meta": {"a": 1, "b": [1, {"y": 2, "x": 1}]}}
    nested_b: dict[str, JsonValue] = {"meta": {"b": [1, {"x": 1, "y": 2}], "a": 1}}
    assert args_hash(TOOL, Action.SEND, reordered) == args_hash(TOOL, Action.SEND, ARGS)
    assert args_hash(TOOL, Action.SEND, nested_a) == args_hash(TOOL, Action.SEND, nested_b)


def test_hash_covers_tool_action_and_every_value() -> None:
    base = args_hash(TOOL, Action.SEND, ARGS)
    assert args_hash("email.draft", Action.SEND, ARGS) != base
    assert args_hash(TOOL, Action.WRITE, ARGS) != base
    assert args_hash(TOOL, Action.SEND, ARGS | {"cc": "x@y.example"}) != base
    assert args_hash(TOOL, Action.SEND, {"n": 1}) != args_hash(TOOL, Action.SEND, {"n": 1.0})
    # List order is meaningful (recipients, line items), so it is not normalised.
    assert args_hash(TOOL, Action.SEND, {"to": ["a", "b"]}) != args_hash(
        TOOL, Action.SEND, {"to": ["b", "a"]}
    )


def test_unicode_args_hash_as_utf8() -> None:
    args: dict[str, JsonValue] = {"to": ["jose@cliente.example"], "subject": "Cotizacion n.º 7 ☕"}
    digest = args_hash(TOOL, Action.SEND, args)
    assert digest == args_hash(TOOL, Action.SEND, dict(reversed(list(args.items()))))
    assert digest != args_hash(TOOL, Action.SEND, {**args, "subject": "Cotizacion n.o 7 ☕"})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_hash_rejects_non_finite_floats(bad: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        args_hash(TOOL, Action.SPEND, {"amount": bad})
    with pytest.raises(ValueError, match="NaN or infinite"):
        args_hash(TOOL, Action.SPEND, {"items": [{"amount": bad}]})


def test_hash_rejects_non_json_values() -> None:
    with pytest.raises(ValueError, match="keys must be strings"):
        args_hash(TOOL, Action.SEND, {"meta": {1: "a"}})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="JSON values"):
        args_hash(TOOL, Action.SEND, {"to": ("a", "b")})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="JSON values"):
        args_hash(TOOL, Action.SEND, {"when": START})  # type: ignore[dict-item]


# create, get, list


def test_create_records_a_pending_approval(queue: ApprovalQueue, store: Store) -> None:
    approval = _create(queue)

    assert approval.status is ApprovalStatus.PENDING
    assert approval.customer_id == "acme"
    assert approval.run_id == "run-1"
    assert approval.requested_by == "sam"
    assert approval.tool == TOOL
    assert approval.action is Action.SEND
    assert approval.args == ARGS
    assert approval.args_hash == args_hash(TOOL, Action.SEND, ARGS)
    assert approval.risk_score == 70
    assert approval.rule_ids == ("external-email-needs-approval", "high-risk")
    assert approval.policy_version == "sha256:0123456789abcdef"
    assert approval.created_at == START
    assert approval.expires_at == START + TTL
    assert approval.decided_by is None
    assert approval.decided_at is None
    assert approval.note is None
    assert approval.executed_at is None
    assert queue.get(approval.id) == approval

    raw = store.connection.execute(
        "SELECT args FROM approvals WHERE id = ?", (approval.id,)
    ).fetchone()
    assert raw["args"] == json.dumps(ARGS, sort_keys=True, separators=(",", ":"))


def test_unicode_args_round_trip(queue: ApprovalQueue) -> None:
    args: dict[str, JsonValue] = {"to": ["zoë@client.example"], "body": "Grüße, 東京 ☕"}
    approval = _create(queue, args=args)
    assert queue.get(approval.id) == approval
    assert approval.args == args
    queue.approve(approval.id, MANAGER, SETTINGS)
    assert _claim(queue, approval.id, args=args).status is ApprovalStatus.EXECUTED


def test_ids_are_random_and_cli_safe(queue: ApprovalQueue) -> None:
    ids = {_create(queue).id for _ in range(50)}
    assert len(ids) == 50
    for approval_id in ids:
        assert approval_id.startswith("apr_")
        assert len(approval_id) >= 30


@pytest.mark.parametrize("ttl", [timedelta(0), timedelta(seconds=-1)])
def test_ttl_must_be_positive(queue: ApprovalQueue, ttl: timedelta) -> None:
    with pytest.raises(ValueError, match="ttl must be positive"):
        _create(queue, ttl=ttl)
    assert queue.list("acme") == []


def test_create_rejects_bad_args_and_risk(queue: ApprovalQueue) -> None:
    with pytest.raises(ValueError, match="NaN"):
        _create(queue, args={"amount": float("nan")})
    with pytest.raises(ValueError, match="risk_score"):
        queue.create(
            customer_id="acme",
            run_id="run-1",
            requested_by="sam",
            tool=TOOL,
            action=Action.SEND,
            args=ARGS,
            risk_score=-1,
            rule_ids=[],
            policy_version="v",
            ttl=TTL,
        )
    assert queue.list("acme") == []


def test_create_needs_an_aware_clock(store: Store) -> None:
    class NaiveClock:
        def now(self) -> datetime:
            return datetime(2026, 10, 22, 9, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        _create(ApprovalQueue(store, NaiveClock()))


def test_get_missing_returns_none(queue: ApprovalQueue) -> None:
    assert queue.get("apr_nope") is None


def test_list_is_oldest_first_per_customer_with_status_filter(
    queue: ApprovalQueue, clock: FixedClock
) -> None:
    first = _create(queue)
    clock.advance(timedelta(seconds=1))
    other = _create(queue, customer_id="globex", requested_by="gus")
    second = _create(queue)
    # Same timestamp as second: insertion order breaks the tie.
    third = _create(queue)
    queue.approve(second.id, MANAGER, SETTINGS)
    queue.reject(third.id, MANAGER, SETTINGS)

    assert [a.id for a in queue.list("acme")] == [first.id, second.id, third.id]
    assert [a.id for a in queue.list("acme", ApprovalStatus.PENDING)] == [first.id]
    assert [a.id for a in queue.list("acme", ApprovalStatus.APPROVED)] == [second.id]
    assert [a.id for a in queue.list("acme", ApprovalStatus.REJECTED)] == [third.id]
    assert queue.list("acme", ApprovalStatus.EXECUTED) == []
    assert [a.id for a in queue.list("globex")] == [other.id]
    assert queue.list("initech") == []


def test_approvals_persist_in_a_file_database(tmp_path: Path, clock: FixedClock) -> None:
    path = tmp_path / "agentgate.db"
    with Store(path) as store:
        approval = _create(ApprovalQueue(store, clock))
    with Store(path) as store:
        assert ApprovalQueue(store, clock).get(approval.id) == approval


# Happy path


def test_happy_path_end_to_end(queue: ApprovalQueue, clock: FixedClock) -> None:
    approval = _create(queue)
    clock.advance(timedelta(minutes=5))

    approved = queue.approve(approval.id, MANAGER, SETTINGS, note="looks right")
    assert approved.status is ApprovalStatus.APPROVED
    assert approved.decided_by == "maya"
    assert approved.decided_at == START + timedelta(minutes=5)
    assert approved.note == "looks right"
    assert approved.executed_at is None

    clock.advance(timedelta(minutes=5))
    reordered = {"body": ARGS["body"], "to": ARGS["to"], "subject": ARGS["subject"]}
    executed = _claim(queue, approval.id, args=reordered)
    assert executed.status is ApprovalStatus.EXECUTED
    assert executed.executed_at == START + timedelta(minutes=10)
    assert executed.decided_by == "maya"
    assert executed.args == ARGS
    assert queue.get(approval.id) == executed


def test_reject_records_the_decision(queue: ApprovalQueue) -> None:
    approval = _create(queue)
    rejected = queue.reject(approval.id, MANAGER, SETTINGS, note="wrong client")
    assert rejected.status is ApprovalStatus.REJECTED
    assert rejected.decided_by == "maya"
    assert rejected.decided_at == START
    assert rejected.note == "wrong client"


# approve and reject


@pytest.mark.parametrize("decide", ["approve", "reject"])
def test_decide_missing_is_not_found(queue: ApprovalQueue, decide: str) -> None:
    with pytest.raises(ApprovalError) as exc:
        getattr(queue, decide)("apr_missing", MANAGER, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.NOT_FOUND


@pytest.mark.parametrize("decide", ["approve", "reject"])
def test_cross_customer_approver_cannot_tell_it_exists(queue: ApprovalQueue, decide: str) -> None:
    approval = _create(queue)
    outsider = _user("gina", role=Role.OWNER, customer_id="globex")

    with pytest.raises(ApprovalError) as existing:
        getattr(queue, decide)(approval.id, outsider, SETTINGS)
    with pytest.raises(ApprovalError) as missing:
        getattr(queue, decide)("apr_missing", outsider, SETTINGS)

    assert _reason(existing) is ApprovalErrorReason.NOT_FOUND
    assert _reason(missing) is ApprovalErrorReason.NOT_FOUND
    assert str(existing.value) == str(missing.value).replace("apr_missing", approval.id)
    assert queue.get(approval.id) == approval


def test_cross_customer_approver_learns_nothing_about_expiry(
    queue: ApprovalQueue, clock: FixedClock
) -> None:
    approval = _create(queue)
    clock.advance(TTL)
    outsider = _user("gina", role=Role.OWNER, customer_id="globex")
    with pytest.raises(ApprovalError) as exc:
        queue.approve(approval.id, outsider, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.NOT_FOUND
    # No side effect either: the outsider's call did not mark it expired.
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.PENDING


@pytest.mark.parametrize("decide", ["approve", "reject"])
def test_suspended_approver_is_not_allowed(queue: ApprovalQueue, decide: str) -> None:
    approval = _create(queue)
    suspended = _user("olga", role=Role.OWNER, status=UserStatus.SUSPENDED)
    with pytest.raises(ApprovalError) as exc:
        getattr(queue, decide)(approval.id, suspended, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.APPROVER_NOT_ALLOWED
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.PENDING


@pytest.mark.parametrize(
    ("role", "allowed"),
    [
        (Role.OWNER, True),
        (Role.MANAGER, True),
        (Role.STAFF, False),
        (Role.CONTRACTOR, False),
    ],
)
def test_approver_role_follows_settings(queue: ApprovalQueue, role: Role, allowed: bool) -> None:
    approval = _create(queue)
    approver = _user("alex", role=role)
    if allowed:
        assert queue.approve(approval.id, approver, SETTINGS).status is ApprovalStatus.APPROVED
        return
    with pytest.raises(ApprovalError) as exc:
        queue.approve(approval.id, approver, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.APPROVER_NOT_ALLOWED
    with pytest.raises(ApprovalError) as exc:
        queue.reject(approval.id, approver, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.APPROVER_NOT_ALLOWED


def test_settings_can_allow_staff_and_exclude_managers(queue: ApprovalQueue) -> None:
    staff_only = ApprovalSettings(
        approver_roles=frozenset({Role.STAFF}), ttl_minutes=30, risk_threshold=50
    )
    approval = _create(queue)
    with pytest.raises(ApprovalError) as exc:
        queue.approve(approval.id, MANAGER, staff_only)
    assert _reason(exc) is ApprovalErrorReason.APPROVER_NOT_ALLOWED

    staff = _user("stef", role=Role.STAFF)
    assert queue.approve(approval.id, staff, staff_only).decided_by == "stef"


def test_contractor_never_approves_unless_listed(queue: ApprovalQueue) -> None:
    with_contractors = ApprovalSettings(
        approver_roles=frozenset({Role.CONTRACTOR}), ttl_minutes=30, risk_threshold=50
    )
    contractor = _user("cory", role=Role.CONTRACTOR)
    approval = _create(queue)
    with pytest.raises(ApprovalError) as exc:
        queue.approve(approval.id, contractor, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.APPROVER_NOT_ALLOWED
    assert queue.approve(approval.id, contractor, with_contractors).decided_by == "cory"


@pytest.mark.parametrize("decide", ["approve", "reject"])
def test_requester_cannot_decide_their_own_approval(queue: ApprovalQueue, decide: str) -> None:
    approval = _create(queue, requested_by="maya")
    with pytest.raises(ApprovalError) as exc:
        getattr(queue, decide)(approval.id, MANAGER, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.SELF_APPROVAL
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.PENDING


def test_approve_twice_is_not_pending(queue: ApprovalQueue) -> None:
    approval = _approved(queue)
    other = _user("omar", role=Role.OWNER)
    with pytest.raises(ApprovalError) as exc:
        queue.approve(approval.id, other, SETTINGS, note="again")
    assert _reason(exc) is ApprovalErrorReason.NOT_PENDING
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.decided_by == "maya"
    assert stored.note is None


@pytest.mark.parametrize(
    ("first", "second"),
    [("approve", "reject"), ("reject", "approve"), ("reject", "reject")],
)
def test_decided_approvals_cannot_be_redecided(
    queue: ApprovalQueue, first: str, second: str
) -> None:
    approval = _create(queue)
    getattr(queue, first)(approval.id, MANAGER, SETTINGS)
    with pytest.raises(ApprovalError) as exc:
        getattr(queue, second)(approval.id, MANAGER, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.NOT_PENDING


def test_executed_approval_cannot_be_redecided(queue: ApprovalQueue) -> None:
    approval = _approved(queue)
    _claim(queue, approval.id)
    with pytest.raises(ApprovalError) as exc:
        queue.reject(approval.id, MANAGER, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.NOT_PENDING


def test_approve_one_second_before_expiry_succeeds(queue: ApprovalQueue, clock: FixedClock) -> None:
    approval = _create(queue)
    clock.advance(TTL - timedelta(seconds=1))
    assert queue.approve(approval.id, MANAGER, SETTINGS).status is ApprovalStatus.APPROVED


@pytest.mark.parametrize("decide", ["approve", "reject"])
@pytest.mark.parametrize("past", [timedelta(0), timedelta(hours=2)])
def test_decide_at_or_after_expiry_marks_expired(
    queue: ApprovalQueue, clock: FixedClock, decide: str, past: timedelta
) -> None:
    approval = _create(queue)
    clock.advance(TTL + past)
    with pytest.raises(ApprovalError) as exc:
        getattr(queue, decide)(approval.id, MANAGER, SETTINGS)
    assert _reason(exc) is ApprovalErrorReason.EXPIRED

    # The EXPIRED mark committed even though the call raised.
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.EXPIRED
    assert stored.decided_by is None

    with pytest.raises(ApprovalError) as again:
        queue.approve(approval.id, MANAGER, SETTINGS)
    assert _reason(again) is ApprovalErrorReason.EXPIRED


# claim_for_execution


def test_claim_missing_is_not_found(queue: ApprovalQueue) -> None:
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, "apr_missing")
    assert _reason(exc) is ApprovalErrorReason.NOT_FOUND


def test_claim_from_another_customer_is_wrong_customer(queue: ApprovalQueue) -> None:
    approval = _approved(queue)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id, customer_id="globex")
    assert _reason(exc) is ApprovalErrorReason.WRONG_CUSTOMER
    assert _claim(queue, approval.id).status is ApprovalStatus.EXECUTED


def test_wrong_customer_comes_before_every_other_fact(
    queue: ApprovalQueue, clock: FixedClock
) -> None:
    executed = _approved(queue)
    _claim(queue, executed.id)
    pending = _create(queue)
    rejected = _create(queue)
    queue.reject(rejected.id, MANAGER, SETTINGS)
    due = _approved(queue)
    clock.advance(TTL)

    for approval in (executed, pending, rejected, due):
        with pytest.raises(ApprovalError) as exc:
            _claim(queue, approval.id, customer_id="globex", args={"to": ["evil@x.example"]})
        assert _reason(exc) is ApprovalErrorReason.WRONG_CUSTOMER
    # A wrong-customer claim writes nothing, not even an expiry mark.
    stored = queue.get(due.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.APPROVED


def test_changed_recipient_after_approval_is_args_mismatch(queue: ApprovalQueue) -> None:
    approval = _approved(queue)
    changed = ARGS | {"to": ["attacker@evil.example"]}
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id, args=changed)
    assert _reason(exc) is ApprovalErrorReason.ARGS_MISMATCH
    # A mismatch does not consume the approval: the exact approved action still runs.
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.APPROVED
    assert _claim(queue, approval.id).status is ApprovalStatus.EXECUTED


@pytest.mark.parametrize(
    "changed",
    [
        ARGS | {"bcc": ["attacker@evil.example"]},
        ARGS | {"to": [*ARGS["to"], "attacker@evil.example"]},  # type: ignore[misc]
        {k: v for k, v in ARGS.items() if k != "body"},
        ARGS | {"subject": "Quote "},
        {},
    ],
    ids=["added-key", "added-recipient", "removed-key", "whitespace", "empty"],
)
def test_any_args_change_is_args_mismatch(
    queue: ApprovalQueue, changed: dict[str, JsonValue]
) -> None:
    approval = _approved(queue)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id, args=changed)
    assert _reason(exc) is ApprovalErrorReason.ARGS_MISMATCH


def test_changed_tool_or_action_is_args_mismatch(queue: ApprovalQueue) -> None:
    approval = _approved(queue)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id, tool="email.send_bulk")
    assert _reason(exc) is ApprovalErrorReason.ARGS_MISMATCH
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id, action=Action.WRITE)
    assert _reason(exc) is ApprovalErrorReason.ARGS_MISMATCH


def test_unhashable_claim_args_are_args_mismatch(queue: ApprovalQueue) -> None:
    approval = _approved(queue)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id, args=ARGS | {"amount": float("nan")})
    assert _reason(exc) is ApprovalErrorReason.ARGS_MISMATCH


def test_double_execute_is_already_executed(queue: ApprovalQueue, clock: FixedClock) -> None:
    approval = _approved(queue)
    first = _claim(queue, approval.id)
    clock.advance(timedelta(minutes=1))
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id)
    assert _reason(exc) is ApprovalErrorReason.ALREADY_EXECUTED
    assert queue.get(approval.id) == first


def test_already_executed_even_after_expiry(queue: ApprovalQueue, clock: FixedClock) -> None:
    approval = _approved(queue)
    _claim(queue, approval.id)
    clock.advance(TTL * 2)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id)
    assert _reason(exc) is ApprovalErrorReason.ALREADY_EXECUTED
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.EXECUTED


def test_execute_pending_is_not_approved(queue: ApprovalQueue) -> None:
    approval = _create(queue)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id)
    assert _reason(exc) is ApprovalErrorReason.NOT_APPROVED


def test_reject_then_execute_is_not_approved(queue: ApprovalQueue, clock: FixedClock) -> None:
    approval = _create(queue)
    queue.reject(approval.id, MANAGER, SETTINGS)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id)
    assert _reason(exc) is ApprovalErrorReason.NOT_APPROVED
    # Still NOT_APPROVED after expiry, and a rejection is never rewritten as expired.
    clock.advance(TTL)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id)
    assert _reason(exc) is ApprovalErrorReason.NOT_APPROVED
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.REJECTED


def test_execute_one_second_before_expiry_succeeds(queue: ApprovalQueue, clock: FixedClock) -> None:
    approval = _approved(queue)
    clock.advance(TTL - timedelta(seconds=1))
    assert _claim(queue, approval.id).status is ApprovalStatus.EXECUTED


@pytest.mark.parametrize("past", [timedelta(0), timedelta(microseconds=1), timedelta(days=1)])
def test_execute_approved_at_or_after_expiry_is_expired(
    queue: ApprovalQueue, clock: FixedClock, past: timedelta
) -> None:
    approval = _approved(queue)
    clock.advance(TTL + past)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id)
    assert _reason(exc) is ApprovalErrorReason.EXPIRED
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.EXPIRED
    assert stored.executed_at is None
    assert stored.decided_by == "maya"

    with pytest.raises(ApprovalError) as again:
        _claim(queue, approval.id)
    assert _reason(again) is ApprovalErrorReason.EXPIRED


def test_execute_pending_after_expiry_is_expired(queue: ApprovalQueue, clock: FixedClock) -> None:
    approval = _create(queue)
    clock.advance(TTL)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id)
    assert _reason(exc) is ApprovalErrorReason.EXPIRED
    stored = queue.get(approval.id)
    assert stored is not None
    assert stored.status is ApprovalStatus.EXPIRED


def test_args_mismatch_is_checked_before_status(queue: ApprovalQueue) -> None:
    # A caller with the wrong args learns nothing about whether the approval was decided.
    approval = _create(queue)
    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approval.id, args={"to": ["x@y.example"]})
    assert _reason(exc) is ApprovalErrorReason.ARGS_MISMATCH


def test_concurrent_claims_succeed_exactly_once(tmp_path: Path, clock: FixedClock) -> None:
    with Store(tmp_path / "agentgate.db") as store:
        queue = ApprovalQueue(store, clock)
        approval = _approved(queue)
        barrier = threading.Barrier(10)
        results: list[Approval | ApprovalErrorReason] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            try:
                outcome: Approval | ApprovalErrorReason = _claim(queue, approval.id)
            except ApprovalError as exc:
                outcome = exc.reason
            with results_lock:
                results.append(outcome)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        successes = [r for r in results if isinstance(r, Approval)]
        assert len(results) == 10
        assert len(successes) == 1
        assert successes[0].status is ApprovalStatus.EXECUTED
        assert results.count(ApprovalErrorReason.ALREADY_EXECUTED) == 9


# expire_due


def test_expire_due_marks_live_approvals_past_expiry(
    queue: ApprovalQueue, clock: FixedClock
) -> None:
    pending_due = _create(queue)
    approved_due = _approved(queue)
    rejected_due = _create(queue)
    queue.reject(rejected_due.id, MANAGER, SETTINGS)
    executed_due = _approved(queue)
    _claim(queue, executed_due.id)
    other_customer_due = _create(queue, customer_id="globex", requested_by="gus")
    not_due = _create(queue, ttl=TTL + timedelta(seconds=1))

    clock.advance(TTL)
    expired = queue.expire_due()

    assert [a.id for a in expired] == [pending_due.id, approved_due.id, other_customer_due.id]
    assert all(a.status is ApprovalStatus.EXPIRED for a in expired)
    statuses = {a.id: a.status for a in queue.list("acme")}
    assert statuses == {
        pending_due.id: ApprovalStatus.EXPIRED,
        approved_due.id: ApprovalStatus.EXPIRED,
        rejected_due.id: ApprovalStatus.REJECTED,
        executed_due.id: ApprovalStatus.EXECUTED,
        not_due.id: ApprovalStatus.PENDING,
    }
    assert queue.expire_due() == []

    with pytest.raises(ApprovalError) as exc:
        _claim(queue, approved_due.id)
    assert _reason(exc) is ApprovalErrorReason.EXPIRED

    clock.advance(timedelta(seconds=1))
    assert [a.id for a in queue.expire_due()] == [not_due.id]


def test_expire_due_with_nothing_due(queue: ApprovalQueue, clock: FixedClock) -> None:
    _create(queue)
    clock.advance(TTL - timedelta(microseconds=1))
    assert queue.expire_due() == []


def test_error_carries_reason_and_message() -> None:
    error = ApprovalError(ApprovalErrorReason.EXPIRED, "approval 'x' has expired")
    assert error.reason is ApprovalErrorReason.EXPIRED
    assert str(error) == "approval 'x' has expired"
    assert ApprovalStatus.EXECUTED.value == "executed"
    assert ApprovalErrorReason.APPROVER_NOT_ALLOWED.value == "approver_not_allowed"


# Record hooks: a state change commits only together with its record


class Boom(Exception):
    pass


class Recorder:
    """Records what it is called with, and whether the call was inside a transaction."""

    def __init__(self, store: Store, *, fail: bool = False) -> None:
        self.store = store
        self.fail = fail
        self.calls: list[tuple[str, ApprovalStatus, bool]] = []

    def __call__(self, approval: Approval) -> None:
        self.calls.append((approval.id, approval.status, self.store.connection.in_transaction))
        if self.fail:
            raise Boom


def test_create_records_inside_its_transaction(queue: ApprovalQueue, store: Store) -> None:
    record = Recorder(store)
    approval = queue.create(
        customer_id="acme",
        run_id="run-1",
        requested_by="sam",
        tool=TOOL,
        action=Action.SEND,
        args=ARGS,
        risk_score=70,
        rule_ids=[],
        policy_version="sha256:0123456789abcdef",
        ttl=TTL,
        record=record,
    )
    assert record.calls == [(approval.id, ApprovalStatus.PENDING, True)]


def test_create_whose_record_fails_leaves_no_row(queue: ApprovalQueue, store: Store) -> None:
    with pytest.raises(Boom):
        queue.create(
            customer_id="acme",
            run_id="run-1",
            requested_by="sam",
            tool=TOOL,
            action=Action.SEND,
            args=ARGS,
            risk_score=70,
            rule_ids=[],
            policy_version="sha256:0123456789abcdef",
            ttl=TTL,
            record=Recorder(store, fail=True),
        )
    assert queue.list("acme") == []
    assert not store.connection.in_transaction


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_decision_whose_record_fails_stays_pending(
    queue: ApprovalQueue, store: Store, decision: str
) -> None:
    approval = _create(queue)
    decide = queue.approve if decision == "approve" else queue.reject
    with pytest.raises(Boom):
        decide(approval.id, MANAGER, SETTINGS, "note", record=Recorder(store, fail=True))
    stored = queue.get(approval.id)
    assert stored is not None
    assert (stored.status, stored.decided_by, stored.note) == (ApprovalStatus.PENDING, None, None)
    # A retry with a working record goes through, and records the decided state.
    record = Recorder(store)
    decided = decide(approval.id, MANAGER, SETTINGS, "note", record=record)
    assert record.calls == [(approval.id, decided.status, True)]
    assert decided.status is not ApprovalStatus.PENDING


def test_claim_whose_record_fails_stays_approved(queue: ApprovalQueue, store: Store) -> None:
    approval = _approved(queue)
    with pytest.raises(Boom):
        queue.claim_for_execution(
            approval.id,
            customer_id="acme",
            tool=TOOL,
            action=Action.SEND,
            args=ARGS,
            record=Recorder(store, fail=True),
        )
    stored = queue.get(approval.id)
    assert stored is not None
    assert (stored.status, stored.executed_at) == (ApprovalStatus.APPROVED, None)
    record = Recorder(store)
    claimed = queue.claim_for_execution(
        approval.id, customer_id="acme", tool=TOOL, action=Action.SEND, args=ARGS, record=record
    )
    assert claimed.status is ApprovalStatus.EXECUTED
    assert record.calls == [(approval.id, ApprovalStatus.EXECUTED, True)]


def test_record_is_not_called_for_a_refused_claim(queue: ApprovalQueue, store: Store) -> None:
    approval = _approved(queue)
    record = Recorder(store)
    with pytest.raises(ApprovalError):
        queue.claim_for_execution(
            approval.id,
            customer_id="acme",
            tool=TOOL,
            action=Action.SEND,
            args={**ARGS, "subject": "Changed"},
            record=record,
        )
    assert record.calls == []


def test_expire_due_whose_record_fails_expires_nothing(
    queue: ApprovalQueue, store: Store, clock: FixedClock
) -> None:
    first, second = _create(queue), _approved(queue)
    clock.advance(TTL)
    with pytest.raises(Boom):
        queue.expire_due(record=Recorder(store, fail=True))
    statuses = [queue.get(a.id).status for a in (first, second)]  # type: ignore[union-attr]
    assert statuses == [ApprovalStatus.PENDING, ApprovalStatus.APPROVED]
    record = Recorder(store)
    assert [a.id for a in queue.expire_due(record=record)] == [first.id, second.id]
    assert [(i, s) for i, s, _ in record.calls] == [
        (first.id, ApprovalStatus.EXPIRED),
        (second.id, ApprovalStatus.EXPIRED),
    ]


def test_expiry_found_by_approve_is_recorded_with_its_mark(
    queue: ApprovalQueue, store: Store, clock: FixedClock
) -> None:
    approval = _create(queue)
    clock.advance(TTL)
    # The mark cannot be recorded: nothing is marked and the record's error propagates.
    with pytest.raises(Boom):
        queue.approve(approval.id, MANAGER, SETTINGS, expired=Recorder(store, fail=True))
    assert queue.get(approval.id).status is ApprovalStatus.PENDING  # type: ignore[union-attr]
    expired = Recorder(store)
    decision = Recorder(store)
    with pytest.raises(ApprovalError) as caught:
        queue.approve(approval.id, MANAGER, SETTINGS, record=decision, expired=expired)
    assert _reason(caught) is ApprovalErrorReason.EXPIRED
    assert expired.calls == [(approval.id, ApprovalStatus.EXPIRED, True)]
    assert decision.calls == []
    assert queue.get(approval.id).status is ApprovalStatus.EXPIRED  # type: ignore[union-attr]


def test_expiry_found_by_claim_is_recorded_with_its_mark(
    queue: ApprovalQueue, store: Store, clock: FixedClock
) -> None:
    approval = _approved(queue)
    clock.advance(TTL)
    with pytest.raises(Boom):
        queue.claim_for_execution(
            approval.id,
            customer_id="acme",
            tool=TOOL,
            action=Action.SEND,
            args=ARGS,
            expired=Recorder(store, fail=True),
        )
    assert queue.get(approval.id).status is ApprovalStatus.APPROVED  # type: ignore[union-attr]
    expired = Recorder(store)
    with pytest.raises(ApprovalError) as caught:
        queue.claim_for_execution(
            approval.id,
            customer_id="acme",
            tool=TOOL,
            action=Action.SEND,
            args=ARGS,
            expired=expired,
        )
    assert _reason(caught) is ApprovalErrorReason.EXPIRED
    assert expired.calls == [(approval.id, ApprovalStatus.EXPIRED, True)]
    assert queue.get(approval.id).status is ApprovalStatus.EXPIRED  # type: ignore[union-attr]
