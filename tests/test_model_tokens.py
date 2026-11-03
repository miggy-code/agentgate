"""Run tokens: only a hash is stored, a token is bound to its run, and it is refused once the
token or the run's user is no longer good."""

import os
import stat
from datetime import timedelta
from pathlib import Path

import pytest

from agentgate.audit.base import EventKind, MemorySink, OperatorAction
from agentgate.context import RunContext
from agentgate.fixtures import load_fixtures
from agentgate.gate import Gate
from agentgate.model.tokens import (
    DEFAULT_TTL,
    MAX_LIVE_TOKENS_PER_RUN,
    MAX_TTL,
    TokenError,
    TokenReason,
    TokenRecord,
    TokenRejected,
    TokenStore,
    presented_token,
    token_hash,
    write_token_file,
)
from agentgate.retrieval import DocumentIndex
from agentgate.store import Store
from conftest import FIXTURES, KC, World, build_world

STAFF = "kc-staff-nadine"
OTHER = "kc-owner-dale"


@pytest.fixture
def tokens(world: World) -> TokenStore:
    return TokenStore(world.store, world.clock)


def _reason(tokens: TokenStore, token: str | None, **kwargs: str) -> TokenReason:
    with pytest.raises(TokenRejected) as rejected:
        tokens.resolve(token, **kwargs)
    return rejected.value.reason


# Issue and resolve


def test_a_token_resolves_to_its_run(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    assert issued.token.startswith("agt_")
    assert tokens.resolve(issued.token) == ctx
    assert tokens.resolve(issued.token, run_id=ctx.run_id) == ctx


def test_tokens_are_random_and_distinct(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    first, second = tokens.issue(ctx.run_id), tokens.issue(ctx.run_id)
    assert first.token != second.token and first.record.token_id != second.record.token_id


def test_only_a_hash_reaches_the_database_file(tmp_path: Path) -> None:
    # On a file, so the bytes SQLite wrote can be searched, not only the rows.
    path = tmp_path / "store.db"
    world = build_world()
    with Store(path) as store:
        load_fixtures(store, FIXTURES)
        gate = Gate(
            store=store,
            engines=world.engines,
            sink=MemorySink(),
            clock=world.clock,
            index=DocumentIndex(store),
        )
        user = store.get_user(STAFF)
        assert user is not None
        run = gate.start_run(user.customer_id, user.id)
        issued = TokenStore(store, world.clock).issue(run.run_id)
        store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    data = b"".join(p.read_bytes() for p in tmp_path.glob("store.db*"))
    assert issued.token.encode() not in data
    assert token_hash(issued.token).encode() in data
    world.store.close()


def test_the_stored_row_holds_the_hash_and_not_the_token(world: World, tokens: TokenStore) -> None:
    issued = tokens.issue(world.run(STAFF).run_id)
    rows = world.store.connection.execute("SELECT * FROM run_tokens").fetchall()
    values = [str(value) for row in rows for value in tuple(row)]
    assert token_hash(issued.token) in values
    assert not any(issued.token in value for value in values)
    assert issued.token not in repr(issued)


def test_listing_never_returns_a_hash_or_a_token(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    [listed] = tokens.list_tokens(ctx.run_id)
    assert listed == issued.record
    assert issued.token not in repr(listed) and token_hash(issued.token) not in repr(listed)


@pytest.mark.parametrize("token", [None, ""])
def test_a_missing_token_is_refused(tokens: TokenStore, token: str | None) -> None:
    assert _reason(tokens, token) is TokenReason.MISSING


@pytest.mark.parametrize(
    "token", ["agt_" + "0" * 43, "agt_", "sk-ant-not-ours", "AGT_x", " agt_x", "agt_x\x00"]
)
def test_an_unknown_token_is_refused(tokens: TokenStore, token: str) -> None:
    assert _reason(tokens, token) is TokenReason.UNKNOWN


def test_a_token_with_one_character_changed_is_refused(world: World, tokens: TokenStore) -> None:
    issued = tokens.issue(world.run(STAFF).run_id)
    last = "A" if issued.token[-1] != "A" else "B"
    assert _reason(tokens, issued.token[:-1] + last) is TokenReason.UNKNOWN
    assert _reason(tokens, issued.token + "x") is TokenReason.UNKNOWN


def test_a_token_expires(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id, timedelta(minutes=10))
    world.clock.advance(timedelta(minutes=10) - timedelta(seconds=1))
    assert tokens.resolve(issued.token) == ctx
    world.clock.advance(timedelta(seconds=1))
    assert _reason(tokens, issued.token) is TokenReason.EXPIRED


def test_a_revoked_token_is_refused(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    assert tokens.revoke(issued.record.token_id) is True
    assert _reason(tokens, issued.token) is TokenReason.REVOKED
    assert tokens.revoke(issued.record.token_id) is False
    stored = tokens.get(issued.record.token_id)
    assert stored is not None and stored.revoked_reason == "operator"


def test_revoking_one_token_leaves_the_others(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    first, second = tokens.issue(ctx.run_id), tokens.issue(ctx.run_id)
    tokens.revoke(first.record.token_id)
    assert tokens.resolve(second.token) == ctx


def test_an_unknown_token_id_revokes_nothing(tokens: TokenStore) -> None:
    assert tokens.revoke("0123456789abcdef") is False


# Bound to a run


def test_a_token_from_another_run_is_refused(world: World, tokens: TokenStore) -> None:
    mine, theirs = world.run(STAFF), world.run(STAFF)
    other = tokens.issue(theirs.run_id)
    with pytest.raises(TokenRejected) as rejected:
        tokens.resolve(other.token, run_id=mine.run_id)
    assert rejected.value.reason is TokenReason.WRONG_RUN
    # The refusal names the token's own run, so it can be audited there.
    assert rejected.value.run_id == theirs.run_id


def test_a_token_from_another_users_run_is_refused(world: World, tokens: TokenStore) -> None:
    staff, owner = world.run(STAFF), world.run(OTHER)
    assert _reason(tokens, tokens.issue(owner.run_id).token, run_id=staff.run_id) is (
        TokenReason.WRONG_RUN
    )


def test_a_token_cannot_be_issued_for_an_unknown_run(tokens: TokenStore) -> None:
    for run_id in ("no-such-run", "", "' OR 1=1 --"):
        with pytest.raises(TokenError, match="unknown run"):
            tokens.issue(run_id)
    assert tokens.list_tokens() == []


def test_end_run_revokes_that_runs_tokens_only(world: World, tokens: TokenStore) -> None:
    ended, kept = world.run(STAFF), world.run(STAFF)
    first, second = tokens.issue(ended.run_id), tokens.issue(ended.run_id)
    survivor = tokens.issue(kept.run_id)
    assert tokens.end_run(ended.run_id, record=lambda _: None) == 2
    assert _reason(tokens, first.token) is TokenReason.REVOKED
    assert _reason(tokens, second.token) is TokenReason.REVOKED
    assert tokens.resolve(survivor.token) == kept
    stored = tokens.get(first.record.token_id)
    assert stored is not None and stored.revoked_reason == "run_ended"
    assert tokens.end_run(ended.run_id, record=lambda _: None) == 0


def test_end_run_records_each_revocation_in_its_transaction(
    world: World, tokens: TokenStore
) -> None:
    ctx = world.run(STAFF)
    issued = [tokens.issue(ctx.run_id) for _ in range(3)]
    tokens.revoke(issued[0].record.token_id)
    seen: list[TokenRecord] = []
    assert tokens.end_run(ctx.run_id, record=seen.append) == 2
    assert {t.token_id for t in seen} == {i.record.token_id for i in issued[1:]}
    assert all(t.revoked_reason == "run_ended" and t.revoked_at is not None for t in seen)


def test_an_end_run_that_cannot_be_recorded_revokes_nothing(
    world: World, tokens: TokenStore
) -> None:
    ctx = world.run(STAFF)
    issued = [tokens.issue(ctx.run_id) for _ in range(3)]
    calls = 0

    def refuse_the_second(record: TokenRecord) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("audit down")

    with pytest.raises(RuntimeError):
        tokens.end_run(ctx.run_id, record=refuse_the_second)
    assert all(tokens.resolve(i.token) == ctx for i in issued)


def test_the_gate_records_a_run_ended_revocation_with_no_actor(
    world: World, tokens: TokenStore
) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    tokens.end_run(
        ctx.run_id,
        record=lambda t: world.gate.record_token_event(
            OperatorAction.TOKEN_REVOKED, t, customer_id=ctx.customer_id, actor=None
        ),
    )
    [event] = [e for e in world.events(EventKind.OPERATOR_ACTION) if e.reason == "token_revoked"]
    assert event.subject == issued.record.token_id and event.user_id is None
    assert event.details["reason"] == "run_ended" and event.details["run_id"] == ctx.run_id
    assert event.details["actor"] == "system"


# The run's user


def test_a_token_is_refused_once_its_runs_user_is_suspended(
    world: World, tokens: TokenStore
) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    world.set_user(STAFF, status="suspended")
    with pytest.raises(TokenRejected) as rejected:
        tokens.resolve(issued.token, run_id=ctx.run_id)
    assert rejected.value.reason is TokenReason.USER_SUSPENDED
    assert rejected.value.run_id == ctx.run_id
    # Nothing about the token changed, so lifting the suspension lifts the refusal.
    world.set_user(STAFF, status="active")
    assert tokens.resolve(issued.token) == ctx


def test_no_token_is_issued_for_a_suspended_users_run(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    world.set_user(STAFF, status="suspended")
    with pytest.raises(TokenError, match="suspended"):
        tokens.issue(ctx.run_id)


def test_a_token_is_refused_when_its_users_customer_changed(
    world: World, tokens: TokenStore
) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    world.set_user(STAFF, customer_id="juniper-row-apothecary")
    assert _reason(tokens, issued.token) is TokenReason.RUN_INVALID


def test_a_token_is_refused_when_its_user_is_gone(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    world.store.connection.execute("PRAGMA foreign_keys = OFF")
    world.store.connection.execute("DELETE FROM users WHERE id = ?", (STAFF,))
    assert _reason(tokens, issued.token) is TokenReason.RUN_INVALID


def test_the_resolved_run_is_the_stored_one(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    resolved = tokens.resolve(issued.token)
    assert isinstance(resolved, RunContext) and resolved.customer_id == KC
    assert resolved.user_id == STAFF and resolved.role == ctx.role


# Lifetime and limits


@pytest.mark.parametrize(
    "ttl", [timedelta(0), timedelta(seconds=-1), MAX_TTL + timedelta(seconds=1), timedelta.max]
)
def test_a_bad_lifetime_is_refused(world: World, tokens: TokenStore, ttl: timedelta) -> None:
    with pytest.raises(TokenError, match="lifetime"):
        tokens.issue(world.run(STAFF).run_id, ttl)


def test_the_longest_lifetime_is_accepted(world: World, tokens: TokenStore) -> None:
    issued = tokens.issue(world.run(STAFF).run_id, MAX_TTL)
    assert issued.record.expires_at - issued.record.issued_at == MAX_TTL
    default = tokens.issue(world.run(STAFF).run_id)
    assert default.record.expires_at - default.record.issued_at == DEFAULT_TTL


def test_a_run_holds_a_bounded_number_of_live_tokens(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    issued = [tokens.issue(ctx.run_id) for _ in range(MAX_LIVE_TOKENS_PER_RUN)]
    with pytest.raises(TokenError, match="live tokens"):
        tokens.issue(ctx.run_id)
    # Another run is not affected, and revoking one frees a place.
    tokens.issue(world.run(STAFF).run_id)
    tokens.revoke(issued[0].record.token_id)
    tokens.issue(ctx.run_id)


def test_an_expired_token_does_not_count_toward_the_limit(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)
    for _ in range(MAX_LIVE_TOKENS_PER_RUN):
        tokens.issue(ctx.run_id, timedelta(minutes=1))
    world.clock.advance(timedelta(minutes=2))
    tokens.issue(ctx.run_id)


# Recording


def test_an_issue_that_cannot_be_recorded_leaves_no_token(world: World, tokens: TokenStore) -> None:
    ctx = world.run(STAFF)

    def refuse(record: TokenRecord) -> None:
        raise RuntimeError("audit down")

    with pytest.raises(RuntimeError):
        tokens.issue(ctx.run_id, record=refuse)
    assert tokens.list_tokens() == []


def test_a_revocation_that_cannot_be_recorded_is_not_made(world: World, tokens: TokenStore) -> None:
    issued = tokens.issue(world.run(STAFF).run_id)

    def refuse(record: TokenRecord) -> None:
        raise RuntimeError("audit down")

    with pytest.raises(RuntimeError):
        tokens.revoke(issued.record.token_id, record=refuse)
    assert tokens.resolve(issued.token) is not None


def test_the_record_callback_sees_the_token_as_stored(world: World, tokens: TokenStore) -> None:
    seen: list[TokenRecord] = []
    issued = tokens.issue(world.run(STAFF).run_id, record=seen.append)
    assert seen == [issued.record]
    tokens.revoke(issued.record.token_id, record=seen.append)
    assert seen[1].revoked_at is not None and seen[1].token_id == issued.record.token_id


# What a client presents


@pytest.mark.parametrize(
    ("headers", "query", "expected"),
    [
        ([("Authorization", "Bearer agt_a")], [], "agt_a"),
        ([("authorization", "bearer  agt_a ")], [], "agt_a"),
        ([("X-Api-Key", "agt_a")], [], "agt_a"),
        ([("x-goog-api-key", "agt_a")], [], "agt_a"),
        ([("api-key", "agt_a")], [], "agt_a"),
        ([], ["agt_a"], "agt_a"),
        ([("Authorization", "Bearer agt_a"), ("x-api-key", "agt_a")], ["agt_a"], "agt_a"),
        ([("content-type", "application/json")], [], None),
        ([("Authorization", "Bearer ")], [], None),
        ([], [], None),
    ],
)
def test_one_presented_credential_is_read(
    headers: list[tuple[str, str]], query: list[str], expected: str | None
) -> None:
    assert presented_token(headers, query) == expected


@pytest.mark.parametrize(
    ("headers", "query"),
    [
        ([("Authorization", "Bearer agt_a"), ("x-api-key", "agt_b")], []),
        ([("Authorization", "Bearer agt_a"), ("authorization", "Bearer agt_b")], []),
        ([("x-api-key", "agt_a")], ["agt_b"]),
        ([("Authorization", "Bearer agt_a")], ["sk-real-provider-key"]),
    ],
)
def test_different_presented_credentials_are_ambiguous(
    headers: list[tuple[str, str]], query: list[str]
) -> None:
    with pytest.raises(TokenRejected) as rejected:
        presented_token(headers, query)
    assert rejected.value.reason is TokenReason.AMBIGUOUS


# The token file


def test_a_token_file_is_owner_only_from_creation(tmp_path: Path) -> None:
    previous = os.umask(0o000)
    try:
        path = tmp_path / "run.token"
        write_token_file(path, "agt_secret")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text() == "agt_secret\n"


def test_a_token_file_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "run.token"
    path.write_text("precious")
    with pytest.raises(FileExistsError):
        write_token_file(path, "agt_secret")
    assert path.read_text() == "precious"


def test_a_token_file_does_not_follow_a_link(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    link = tmp_path / "run.token"
    link.symlink_to(target)
    with pytest.raises(OSError):
        write_token_file(link, "agt_secret")
    assert not target.exists()


def test_end_run_has_no_unrecorded_form(tokens: TokenStore) -> None:
    with pytest.raises(TypeError):
        tokens.end_run("any-run")  # type: ignore[call-arg]


def test_a_token_that_vanishes_during_revocation_is_an_error_not_an_assert(
    world: World, tokens: TokenStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = world.run(STAFF)
    issued = tokens.issue(ctx.run_id)
    monkeypatch.setattr(TokenStore, "get", lambda self, token_id: None)
    with pytest.raises(TokenError, match="disappeared"):
        tokens.revoke(issued.record.token_id, record=lambda _: None)
    with pytest.raises(TokenError, match="disappeared"):
        tokens.end_run(ctx.run_id, record=lambda _: None)
    monkeypatch.undo()
    assert tokens.resolve(issued.token) == ctx
