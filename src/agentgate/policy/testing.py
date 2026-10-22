"""Policy tests: run the expected decisions in policies/<customer-id>/tests.yaml.

Each test case names a fixture user, a tool and action, and the facts of one request, plus
the decision the policy must reach. Role and customer always come from the fixtures, never
from the case, so a test cannot pass by claiming a role the user does not have. A setup
problem in one case (an unknown user or document) fails that case and the run continues,
while a problem with a whole file stops the run, since its results could not be trusted.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentgate._yaml import parse_yaml
from agentgate.fixtures import FixtureError, load_fixtures
from agentgate.models import Action
from agentgate.policy.engine import Decision, DocumentFacts, PolicyEngine, Request
from agentgate.policy.schema import Effect, PolicyBundle, PolicyError, load_bundles
from agentgate.store import Store

TESTS_FILE = "tests.yaml"


class PolicyTestCase(BaseModel):
    """One expected decision, exactly as written in tests.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    user: str
    tool: str
    action: Action
    document: str | None = None
    recipients: tuple[str, ...] = ()
    amount_usd: Decimal | None = None
    run_max_label: str | None = None
    expect: Effect
    # Deciding rule ids in policy file order. None leaves them unchecked; an empty list
    # asserts that no rule decided (a default or built-in deny).
    rules: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CaseResult:
    customer_id: str
    case: PolicyTestCase
    # None when the case could not be run, for example because its user does not exist.
    decision: Decision | None
    passed: bool
    problem: str | None = None


@dataclass(frozen=True)
class CustomerReport:
    customer_id: str
    policy_version: str
    results: tuple[CaseResult, ...]

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed


@dataclass(frozen=True)
class PolicyTestReport:
    # Sorted by customer id.
    customers: tuple[CustomerReport, ...]

    @property
    def passed(self) -> int:
        return sum(customer.passed for customer in self.customers)

    @property
    def failed(self) -> int:
        return sum(customer.failed for customer in self.customers)

    @property
    def total(self) -> int:
        return self.passed + self.failed


def load_cases(path: Path) -> list[PolicyTestCase]:
    """Load and validate every test case in one tests.yaml file."""
    raw = parse_yaml(path.read_text(), path, PolicyError)
    if not isinstance(raw, list) or not raw:
        raise PolicyError(f"{path}: expected a non-empty list of test cases")
    cases: list[PolicyTestCase] = []
    seen: dict[str, int] = {}
    for index, entry in enumerate(raw, start=1):
        try:
            case = PolicyTestCase.model_validate(entry)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or 'case'}: {err['msg']}"
                for err in exc.errors(include_url=False, include_input=False)
            )
            raise PolicyError(f"{path}: case {index}: {problems}") from exc
        if case.name in seen:
            raise PolicyError(
                f"{path}: case {index}: duplicate name {case.name!r} (also case {seen[case.name]})"
            )
        seen[case.name] = index
        cases.append(case)
    return cases


def run_cases(
    store: Store, engine: PolicyEngine, cases: Sequence[PolicyTestCase]
) -> list[CaseResult]:
    """Decide every case with engine and compare each decision with its expectation."""
    return [_run_case(store, engine, case) for case in cases]


def _run_case(store: Store, engine: PolicyEngine, case: PolicyTestCase) -> CaseResult:
    customer_id = engine.bundle.customer_id

    def setup_error(problem: str) -> CaseResult:
        return CaseResult(customer_id, case, decision=None, passed=False, problem=problem)

    user = store.get_user(case.user)
    if user is None:
        return setup_error(f"unknown user {case.user!r}")
    if user.customer_id != customer_id:
        # Cross-tenant requests are tested through documents; the acting user must be the
        # policy's own, or the case would only ever exercise the customer mismatch check.
        return setup_error(f"user {user.id!r} belongs to customer {user.customer_id!r}")

    facts: DocumentFacts | None = None
    if case.document is not None:
        # Not tenant-scoped on purpose: a case may name another customer's document to
        # check that the engine refuses it.
        document = store.get_document(case.document)
        if document is None:
            return setup_error(f"unknown document {case.document!r}")
        facts = DocumentFacts.from_document(document)

    decision = engine.decide(
        Request(
            customer_id=user.customer_id,
            user_id=user.id,
            role=user.role,
            tool=case.tool,
            action=case.action,
            document=facts,
            recipients=case.recipients,
            amount_usd=case.amount_usd,
            run_max_label=case.run_max_label,
        )
    )
    problem = _mismatch(case, decision)
    return CaseResult(customer_id, case, decision, passed=problem is None, problem=problem)


def _mismatch(case: PolicyTestCase, decision: Decision) -> str | None:
    if decision.effect is not case.expect:
        detail = decision.reason.value
        if decision.rule_ids:
            detail += f" {_format_rules(decision.rule_ids)}"
        return f"expected {case.expect}, got {decision.effect} ({detail})"
    if case.rules is not None and case.rules != decision.rule_ids:
        return f"expected rules {_format_rules(case.rules)}, got {_format_rules(decision.rule_ids)}"
    return None


def _format_rules(rule_ids: Sequence[str]) -> str:
    return "[" + ", ".join(rule_ids) + "]"


def load_policy_bundles(policies_root: Path) -> dict[str, PolicyBundle]:
    """Load every bundle under policies_root, refusing a missing or empty directory."""
    if not policies_root.is_dir():
        raise PolicyError(f"{policies_root}: policies directory not found")
    bundles = load_bundles(policies_root)
    if not bundles:
        # An empty tree would otherwise lint and test clean while checking nothing.
        raise PolicyError(f"{policies_root}: no policy directories found")
    return bundles


def run_policy_tests(policies_root: Path, fixtures_root: Path) -> PolicyTestReport:
    """Run every customer's tests.yaml against its policy and the fixture data."""
    if not fixtures_root.is_dir():
        raise FixtureError(f"{fixtures_root}: fixtures directory not found")
    bundles = load_policy_bundles(policies_root)
    reports: list[CustomerReport] = []
    with Store() as store:
        load_fixtures(store, fixtures_root)
        for customer_id in sorted(bundles):
            directory = policies_root / customer_id
            customer = store.get_customer(customer_id)
            if customer is None:
                raise PolicyError(f"{directory}: customer {customer_id!r} has no fixtures")
            engine = PolicyEngine(bundles[customer_id], customer)
            tests_path = directory / TESTS_FILE
            if not tests_path.is_file():
                raise PolicyError(f"{tests_path}: missing {TESTS_FILE}; every policy needs tests")
            results = run_cases(store, engine, load_cases(tests_path))
            reports.append(CustomerReport(customer_id, engine.version, tuple(results)))
    return PolicyTestReport(tuple(reports))
