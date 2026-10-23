"""The access red-team scenarios (cross_tenant, privilege_escalation, identity_spoofing):
they all pass against the real gate, their ids and titles are well formed, and none is
vacuous. For every scenario, a mutation removes the one protection it targets and the
scenario must then fail on its own checks or the leak detector, without raising.
"""

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import pytest
from pydantic import ConfigDict

from agentgate.approvals import ApprovalQueue
from agentgate.context import RunContext
from agentgate.gate import Gate, _Run
from agentgate.models import Action, Document, Role, User, UserStatus
from agentgate.policy.engine import Decision, DocumentFacts, PolicyEngine, Reason, Request
from agentgate.policy.schema import Conditions, Effect, LabelBound, PolicyBundle, Rule
from agentgate.redteam import core
from agentgate.redteam.core import (
    REDTEAM_START,
    Category,
    RegisteredScenario,
    ScenarioResult,
    all_scenarios,
    run_scenario,
    run_suite,
)
from agentgate.redteam.scenarios import cross_tenant, identity_spoofing, privilege_escalation
from agentgate.retrieval import _Filter, _read_request
from agentgate.world import World
from conftest import FIXTURES, KC, POLICIES

MODULES = {
    cross_tenant.__name__: Category.CROSS_TENANT,
    privilege_escalation.__name__: Category.PRIVILEGE_ESCALATION,
    identity_spoofing.__name__: Category.IDENTITY_SPOOFING,
}
CATEGORIES = list(MODULES.values())

Mutation = Callable[[World, pytest.MonkeyPatch], None]


def ours() -> list[RegisteredScenario]:
    return [s for s in all_scenarios() if s.module in MODULES]


def run(s: RegisteredScenario) -> ScenarioResult:
    return run_scenario(s, policies=POLICIES, fixtures=FIXTURES)


# The suite


def test_access_scenarios_all_pass() -> None:
    suite = run_suite(categories=CATEGORIES, policies=POLICIES, fixtures=FIXTURES)
    failed = [
        (r.id, r.error, r.leaks, [c.name for c in r.checks if not c.passed]) for r in suite.failed
    ]
    assert failed == []
    ids = {r.id for r in suite.results}
    assert {s.id for s in ours()} <= ids
    for category in CATEGORIES:
        passed, total = suite.totals[category]
        assert passed == total
    for result in suite.results:
        assert result.checks and result.leaks == () and result.error is None


def test_ids_titles_and_categories() -> None:
    scenarios = ours()
    by_module: dict[str, list[RegisteredScenario]] = {}
    for s in scenarios:
        by_module.setdefault(s.module, []).append(s)
    assert set(by_module) == set(MODULES)
    for module, found in by_module.items():
        category = MODULES[module]
        assert 10 <= len(found) <= 15, module
        for s in found:
            assert s.category is category
            assert s.id.startswith(category.prefix + "-")
            assert s.id != "pe-000"
            assert s.title.strip() == s.title and s.title
            assert "\u2014" not in s.title
            assert "\u2013" not in s.title
    numbers = sorted(int(s.id[3:]) for s in by_module[cross_tenant.__name__])
    assert numbers == list(range(1, len(numbers) + 1))
    numbers = sorted(int(s.id[3:]) for s in by_module[privilege_escalation.__name__])
    assert numbers == list(range(1, len(numbers) + 1))
    numbers = sorted(int(s.id[3:]) for s in by_module[identity_spoofing.__name__])
    assert numbers == list(range(1, len(numbers) + 1))


# Mutations: each removes one protection from the arena's gate (never from the leak
# detector's reference engines, which the arena builds separately).


def _track_runs(world: World, mp: pytest.MonkeyPatch) -> dict[str, str]:
    """Remember the customer and run id of the last run the gate accepted."""
    current: dict[str, str] = {}
    real = world.gate._check_run

    def check_run(ctx: RunContext, call: Any) -> Any:
        result = real(ctx, call)
        if isinstance(result, _Run):
            current["customer"] = result.ctx.customer_id
            current["run"] = result.ctx.run_id
        return result

    mp.setattr(world.gate, "_check_run", check_run)
    return current


def tenant_blind_documents(world: World, mp: pytest.MonkeyPatch) -> None:
    """Every document looks as if it belonged to the customer of the run asking for it."""
    current = _track_runs(world, mp)
    real = world.store.get_document

    def get_document(doc_id: str) -> Document | None:
        document = real(doc_id)
        customer = current.get("customer")
        if document is None or customer is None:
            return document
        return document.model_copy(update={"customer_id": customer})

    mp.setattr(world.store, "get_document", get_document)


def tenant_blind_search(world: World, mp: pytest.MonkeyPatch) -> None:
    """Search considers every customer's documents, as if they were the run's own."""
    store = world.store

    def permitted(
        engine: PolicyEngine, ctx: RunContext, tool: str, run_max_label: str | None
    ) -> _Filter:
        rows = store.connection.execute(
            "SELECT id, label, tags, assigned_user_ids FROM documents ORDER BY id"
        ).fetchall()
        allowed: list[str] = []
        for row in rows:
            facts = DocumentFacts(
                customer_id=ctx.customer_id,
                label=row["label"],
                tags=frozenset(__import__("json").loads(row["tags"])),
                assigned_user_ids=frozenset(__import__("json").loads(row["assigned_user_ids"])),
            )
            decision = engine.decide(_read_request(ctx, tool, facts, run_max_label))
            if decision.effect is Effect.ALLOW:
                allowed.append(row["id"])
        return _Filter(len(rows), tuple(allowed), {}, ())

    def recheck(
        engine: PolicyEngine, ctx: RunContext, tool: str, label: str | None, doc_id: str
    ) -> Document:
        document = store.get_document(doc_id)
        assert document is not None
        return document

    mp.setattr(world.index, "_permitted", permitted)
    mp.setattr(world.index, "_recheck", recheck)


class _TenantBlindQueue:
    """An approval queue that ignores which customer and run an approval belongs to."""

    def __init__(self, real: ApprovalQueue, current: dict[str, str]) -> None:
        self._real = real
        self._current = current

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def get(self, approval_id: str) -> Any:
        approval = self._real.get(approval_id)
        if approval is None or "run" not in self._current:
            return approval
        return replace(approval, customer_id=self._current["customer"], run_id=self._current["run"])

    def claim_for_execution(self, approval_id: str, **kwargs: Any) -> Any:
        approval = self._real.get(approval_id)
        if approval is not None:
            kwargs["customer_id"] = approval.customer_id
        return self._real.claim_for_execution(approval_id, **kwargs)

    def approve(self, approval_id: str, approver: User, *args: Any) -> Any:
        return self._real.approve(approval_id, self._adopt(approval_id, approver), *args)

    def reject(self, approval_id: str, approver: User, *args: Any) -> Any:
        return self._real.reject(approval_id, self._adopt(approval_id, approver), *args)

    def _adopt(self, approval_id: str, approver: User) -> User:
        approval = self._real.get(approval_id)
        if approval is None:
            return approver
        return approver.model_copy(update={"customer_id": approval.customer_id})


def approval_tenant_blind(world: World, mp: pytest.MonkeyPatch) -> None:
    current = _track_runs(world, mp)
    mp.setattr(world.gate, "_approvals", _TenantBlindQueue(world.gate.approvals, current))


def lax_args(world: World, mp: pytest.MonkeyPatch) -> None:
    """Tool arguments ignore unknown keys instead of refusing them."""
    for adapter in world.gate._tools.values():
        strict = adapter.args_model
        config = ConfigDict(strict=True, extra="ignore", frozen=True)
        lax = type("Lax" + strict.__name__, (strict,), {"model_config": config})
        mp.setattr(adapter, "args_model", lax)


def allow_all(world: World, mp: pytest.MonkeyPatch) -> None:
    """Every engine of the gate allows every request, built-in checks included."""
    for engine in world.gate._engines.values():

        def decide(request: Request, engine: PolicyEngine = engine) -> Decision:
            return Decision(Effect.ALLOW, Reason.MATCHED_ALLOW, ("mutant",), engine.version)

        mp.setattr(engine, "decide", decide)


def _rebuild(
    world: World, mp: pytest.MonkeyPatch, edit: Callable[[PolicyBundle], PolicyBundle]
) -> None:
    engines: dict[str, PolicyEngine] = {}
    for customer_id, engine in world.gate._engines.items():
        customer = world.store.get_customer(customer_id)
        assert customer is not None
        engines[customer_id] = PolicyEngine(edit(engine.bundle), customer)
    mp.setattr(world.gate, "_engines", engines)


def _edit_rules(world: World, mp: pytest.MonkeyPatch, edit: Callable[[Rule], Rule]) -> None:
    def apply(bundle: PolicyBundle) -> PolicyBundle:
        rules = tuple(edit(rule) for rule in bundle.policy.rules)
        return bundle.model_copy(
            update={"policy": bundle.policy.model_copy(update={"rules": rules})}
        )

    _rebuild(world, mp, apply)


def _ceiling(ids: set[str], label: str = "restricted") -> Callable[[Rule], Rule]:
    def edit(rule: Rule) -> Rule:
        if rule.id not in ids:
            return rule
        return rule.model_copy(update={"labels": LabelBound(max=label)})

    return edit


def staff_read_ceiling_restricted(world: World, mp: pytest.MonkeyPatch) -> None:
    _edit_rules(world, mp, _ceiling({"staff-read"}))


def contractor_read_ceiling_restricted(world: World, mp: pytest.MonkeyPatch) -> None:
    _edit_rules(world, mp, _ceiling({"contractor-read-public"}))


def _unconditional(rule_id: str) -> Callable[[Rule], Rule]:
    def edit(rule: Rule) -> Rule:
        return rule.model_copy(update={"where": Conditions()}) if rule.id == rule_id else rule

    return edit


def assignment_unconditional(world: World, mp: pytest.MonkeyPatch) -> None:
    """Contractors read every internal document, assigned to them or not."""
    _edit_rules(world, mp, _unconditional("contractor-read-assigned"))


def orders_carveout_unconditional(world: World, mp: pytest.MonkeyPatch) -> None:
    """The manager's restricted carve-out covers every restricted document, payroll too."""
    _edit_rules(world, mp, _unconditional("manager-read-orders"))


def note_ceilings_restricted(world: World, mp: pytest.MonkeyPatch) -> None:
    _edit_rules(world, mp, _ceiling({"write-notes-internal", "write-notes-confidential"}))


def contractors_write(world: World, mp: pytest.MonkeyPatch) -> None:
    def edit(rule: Rule) -> Rule:
        if rule.id not in {"write-notes-internal", "email-internal"} or rule.roles is None:
            return rule
        return rule.model_copy(update={"roles": rule.roles | {Role.CONTRACTOR}})

    _edit_rules(world, mp, edit)


def bands_lifted(world: World, mp: pytest.MonkeyPatch) -> None:
    """Every amount band reaches a million and every role may spend."""

    def edit(rule: Rule) -> Rule:
        if rule.where.amount_usd_lte is None:
            return rule
        where = rule.where.model_copy(
            update={"amount_usd_lte": __import__("decimal").Decimal("1000000")}
        )
        roles = None if rule.roles is None else frozenset(Role)
        return rule.model_copy(update={"where": where, "roles": roles})

    _edit_rules(world, mp, edit)


def actions_ignored(world: World, mp: pytest.MonkeyPatch) -> None:
    """Every tool accepts every action and every rule matches any action."""

    def apply(bundle: PolicyBundle) -> PolicyBundle:
        everything = frozenset(Action)
        tools = {
            tool_id: tool.model_copy(update={"actions": everything})
            for tool_id, tool in bundle.tools.items()
        }
        rules = tuple(
            rule.model_copy(update={"actions": everything}) for rule in bundle.policy.rules
        )
        policy = bundle.policy.model_copy(update={"rules": rules})
        return bundle.model_copy(update={"tools": tools, "policy": policy})

    _rebuild(world, mp, apply)


def foreign_tools(world: World, mp: pytest.MonkeyPatch) -> None:
    """Each customer's tool list also holds every other customer's tools."""
    union: dict[str, Any] = {}
    for engine in world.gate._engines.values():
        union.update(engine.bundle.tools)

    _rebuild(
        world, mp, lambda bundle: bundle.model_copy(update={"tools": {**union, **bundle.tools}})
    )


def shared_internal_list(world: World, mp: pytest.MonkeyPatch) -> None:
    """Each customer's internal_addresses list also holds the other customer's domain."""
    shared: set[str] = set()
    for engine in world.gate._engines.values():
        shared |= set(engine.bundle.policy.lists.get("internal_addresses", frozenset()))

    def apply(bundle: PolicyBundle) -> PolicyBundle:
        lists = {**bundle.policy.lists, "internal_addresses": frozenset(shared)}
        return bundle.model_copy(
            update={"policy": bundle.policy.model_copy(update={"lists": lists})}
        )

    _rebuild(world, mp, apply)


def any_role_approves(world: World, mp: pytest.MonkeyPatch) -> None:
    def apply(bundle: PolicyBundle) -> PolicyBundle:
        approvals = bundle.policy.approvals.model_copy(update={"approver_roles": frozenset(Role)})
        policy = bundle.policy.model_copy(update={"approvals": approvals})
        return bundle.model_copy(update={"policy": policy})

    _rebuild(world, mp, apply)


def no_label_tracking(world: World, mp: pytest.MonkeyPatch) -> None:
    """The run's max label never rises, so write-down has nothing to compare against."""
    mp.setattr(world.gate, "_raise_label", lambda run, labels: None)


def trust_context(world: World, mp: pytest.MonkeyPatch) -> None:
    """The gate believes whatever RunContext-shaped object it is handed."""
    gate: Gate = world.gate
    real = gate._check_run

    def check_run(ctx: Any, call: Any) -> Any:
        customer_id = getattr(ctx, "customer_id", None)
        engine = gate._engines.get(customer_id) if isinstance(customer_id, str) else None
        customer = None if engine is None else world.store.get_customer(str(customer_id))
        if engine is None or customer is None:
            return real(ctx, call)
        role = Role(str(ctx.role))
        user = User.model_construct(
            id=str(ctx.user_id),
            customer_id=customer.id,
            name="Claimed caller",
            email="claimed@claimed.example",
            role=role,
            status=UserStatus.ACTIVE,
        )
        trusted = RunContext(
            run_id=str(ctx.run_id),
            customer_id=customer.id,
            user_id=str(ctx.user_id),
            role=role,
            started_at=REDTEAM_START,
        )
        # A run id the gate never issued is taken on trust too, so label tracking has a row.
        with world.store.transaction():
            world.store.connection.execute(
                "INSERT OR IGNORE INTO runs (run_id, customer_id, user_id, role, started_at,"
                " max_label) VALUES (?, ?, ?, ?, ?, NULL)",
                (trusted.run_id, customer.id, trusted.user_id, role.value, "2026-10-22T15:00:00"),
            )
        return _Run(ctx=trusted, user=user, customer=customer, engine=engine)

    mp.setattr(gate, "_check_run", check_run)


def ignore_suspension(world: World, mp: pytest.MonkeyPatch) -> None:
    real = world.store.get_user

    def get_user(user_id: str) -> User | None:
        user = real(user_id)
        return None if user is None else user.model_copy(update={"status": UserStatus.ACTIVE})

    mp.setattr(world.store, "get_user", get_user)


def unknown_approver_is_manager(world: World, mp: pytest.MonkeyPatch) -> None:
    real = world.store.get_user

    def get_user(user_id: str) -> User | None:
        user = real(user_id)
        if user is not None:
            return user
        return User.model_construct(
            id=user_id,
            customer_id=KC,
            name="Nobody",
            email="nobody@kettlecreek.example",
            role=Role.MANAGER,
            status=UserStatus.ACTIVE,
        )

    mp.setattr(world.store, "get_user", get_user)


def run_role_is_owner(world: World, mp: pytest.MonkeyPatch) -> None:
    """The gate honours an owner claim: every accepted run acts as the owner."""
    real = world.gate._check_run

    def check_run(ctx: RunContext, call: Any) -> Any:
        result = real(ctx, call)
        if isinstance(result, _Run):
            return replace(result, ctx=replace(result.ctx, role=Role.OWNER))
        return result

    mp.setattr(world.gate, "_check_run", check_run)


def start_run_customer_blind(world: World, mp: pytest.MonkeyPatch) -> None:
    """start_run takes any user into whichever customer the caller names."""
    gate = world.gate
    store = world.store
    real_start = gate.start_run
    real_get = store.get_user

    def start_run(customer_id: str, user_id: str) -> RunContext:
        def get_user(uid: str) -> User | None:
            user = real_get(uid)
            return None if user is None else user.model_copy(update={"customer_id": customer_id})

        store.get_user = get_user  # type: ignore[method-assign]
        try:
            return real_start(customer_id, user_id)
        finally:
            del store.get_user

    mp.setattr(gate, "start_run", start_run)


MUTATIONS: dict[str, tuple[Mutation, ...]] = {
    "xt-001": (tenant_blind_documents,),
    "xt-002": (tenant_blind_documents,),
    "xt-003": (tenant_blind_documents,),
    "xt-004": (tenant_blind_search,),
    "xt-005": (tenant_blind_search,),
    "xt-006": (approval_tenant_blind,),
    "xt-007": (approval_tenant_blind,),
    "xt-008": (lax_args, tenant_blind_documents),
    "xt-009": (lax_args,),
    "xt-010": (foreign_tools,),
    "xt-011": (shared_internal_list,),
    "xt-012": (trust_context,),
    "xt-013": (start_run_customer_blind,),
    "pe-001": (staff_read_ceiling_restricted, allow_all),
    "pe-002": (staff_read_ceiling_restricted,),
    "pe-003": (contractor_read_ceiling_restricted,),
    "pe-004": (assignment_unconditional,),
    "pe-005": (assignment_unconditional,),
    "pe-006": (orders_carveout_unconditional,),
    "pe-007": (note_ceilings_restricted,),
    "pe-008": (contractors_write,),
    "pe-009": (bands_lifted,),
    "pe-010": (bands_lifted,),
    "pe-011": (no_label_tracking,),
    "pe-012": (no_label_tracking,),
    "pe-013": (staff_read_ceiling_restricted,),
    "pe-014": (actions_ignored,),
    "id-001": (ignore_suspension,),
    "id-002": (ignore_suspension,),
    "id-003": (trust_context,),
    "id-004": (trust_context,),
    "id-005": (trust_context,),
    "id-006": (trust_context,),
    "id-007": (trust_context,),
    "id-008": (run_role_is_owner,),
    "id-009": (lax_args,),
    "id-010": (ignore_suspension,),
    "id-011": (unknown_approver_is_manager,),
    "id-012": (trust_context,),
    "id-013": (any_role_approves,),
    "id-014": (trust_context,),
}
CASES = [(sid, m) for sid, mutations in MUTATIONS.items() for m in mutations]


@pytest.fixture
def mutate(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Mutation], None]]:
    real_open = core.open_memory_world

    def install(mutation: Mutation) -> None:
        def mutated_world(**kwargs: Any) -> World:
            world = real_open(**kwargs)
            mutation(world, monkeypatch)
            return world

        monkeypatch.setattr(core, "open_memory_world", mutated_world)

    yield install


def test_every_access_scenario_has_a_mutation() -> None:
    assert sorted(MUTATIONS) == sorted(s.id for s in ours())


@pytest.mark.parametrize(
    ("scenario_id", "mutation"), CASES, ids=[f"{sid}-{m.__name__}" for sid, m in CASES]
)
def test_scenario_fails_without_its_protection(
    scenario_id: str, mutation: Mutation, mutate: Callable[[Mutation], None]
) -> None:
    target = next(s for s in all_scenarios() if s.id == scenario_id)
    assert run(target).passed
    mutate(mutation)
    result = run(target)
    assert not result.passed
    # A real failure: a check or the leak detector caught the attack, nothing crashed.
    assert result.error is None
    assert result.leaks or any(not check.passed for check in result.checks)
