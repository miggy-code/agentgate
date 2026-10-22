"""Policy evaluation: turn one request into allow, deny or require approval.

Evaluation is deny by default and fails closed. Built-in checks reject requests the policy
cannot reason about (another customer, an unknown tool, an unknown label) before any rule
is consulted. A rule condition whose fact is missing never holds, whatever the rule's
effect, so an allow rule cannot match on absent data. Deny beats require approval, which
beats allow.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from agentgate.models import Action, Customer, Document, Role
from agentgate.policy.schema import DOCUMENT_ACTIONS, Conditions, Effect, PolicyBundle, Rule
from agentgate.policy.version import bundle_version


class Reason(StrEnum):
    MATCHED_DENY = "matched_deny"
    MATCHED_APPROVAL = "matched_approval"
    MATCHED_ALLOW = "matched_allow"
    DEFAULT_DENY = "default_deny"
    CUSTOMER_MISMATCH = "customer_mismatch"
    DOCUMENT_CUSTOMER_MISMATCH = "document_customer_mismatch"
    UNKNOWN_TOOL = "unknown_tool"
    ACTION_NOT_SUPPORTED = "action_not_supported"
    UNKNOWN_LABEL = "unknown_label"
    DOCUMENT_REQUIRED = "document_required"


@dataclass(frozen=True)
class DocumentFacts:
    """What the engine needs to know about a document, without its body."""

    customer_id: str
    label: str
    tags: frozenset[str] = frozenset()
    assigned_user_ids: frozenset[str] = frozenset()

    @classmethod
    def from_document(cls, document: Document) -> "DocumentFacts":
        return cls(
            customer_id=document.customer_id,
            label=document.label,
            tags=frozenset(document.tags),
            assigned_user_ids=frozenset(document.assigned_user_ids),
        )


@dataclass(frozen=True)
class Request:
    customer_id: str
    user_id: str
    role: Role
    tool: str
    action: Action
    document: DocumentFacts | None = None
    recipients: tuple[str, ...] = ()
    amount_usd: Decimal | None = None
    # The most sensitive label this run has read so far.
    run_max_label: str | None = None


@dataclass(frozen=True)
class Decision:
    effect: Effect
    reason: Reason
    # Deciding rules in policy file order. Empty for built-in checks and the default deny.
    rule_ids: tuple[str, ...]
    policy_version: str


# One plain address: no separators, quotes, brackets or whitespace that could smuggle a
# second recipient past a check, and a dotted lowercase domain.
_ADDRESS = re.compile(r"^([^@\s,;:<>()\[\]\"\\]+)@([a-z0-9-]+(?:\.[a-z0-9-]+)+)$")


def _parse_address(raw: str) -> tuple[str, str] | None:
    """Split a recipient into (address, domain), or None if it is not one plain address."""
    match = _ADDRESS.fullmatch(raw.strip().lower())
    if match is None:
        return None
    return match.group(0), match.group(2)


class _AddressList:
    """A named recipient list: exact addresses plus "@domain" entries."""

    def __init__(self, entries: Iterable[str]) -> None:
        self._addresses: set[str] = set()
        self._domains: set[str] = set()
        for entry in entries:
            normalized = entry.strip().lower()
            if normalized.startswith("@"):
                self._domains.add(normalized[1:])
            else:
                self._addresses.add(normalized)

    def contains(self, raw: str) -> bool:
        parsed = _parse_address(raw)
        if parsed is None:
            return False
        address, domain = parsed
        # Exact domain only: "@partner.example" must not admit "evil.partner.example".
        return address in self._addresses or domain in self._domains


class PolicyEngine:
    """Evaluates requests for one customer against that customer's policy bundle."""

    def __init__(self, bundle: PolicyBundle, customer: Customer) -> None:
        if customer.id != bundle.customer_id:
            raise ValueError(
                f"customer {customer.id!r} does not match policy customer {bundle.customer_id!r}"
            )
        self._bundle = bundle
        self._version = bundle_version(bundle)
        self._ranks = {label: rank for rank, label in enumerate(bundle.policy.labels)}
        self._internal_domains = frozenset(domain.lower() for domain in customer.domains)
        self._lists = {name: _AddressList(entries) for name, entries in bundle.policy.lists.items()}

    @property
    def version(self) -> str:
        return self._version

    @property
    def bundle(self) -> PolicyBundle:
        return self._bundle

    def decide(self, request: Request) -> Decision:
        builtin = self._builtin_check(request)
        if builtin is not None:
            return self._decision(Effect.DENY, builtin)

        matched: dict[Effect, list[str]] = {effect: [] for effect in Effect}
        for rule in self._bundle.policy.rules:
            if self._matches(rule, request):
                matched[rule.effect].append(rule.id)

        if matched[Effect.DENY]:
            return self._decision(Effect.DENY, Reason.MATCHED_DENY, matched[Effect.DENY])
        if matched[Effect.REQUIRE_APPROVAL]:
            return self._decision(
                Effect.REQUIRE_APPROVAL, Reason.MATCHED_APPROVAL, matched[Effect.REQUIRE_APPROVAL]
            )
        if matched[Effect.ALLOW]:
            return self._decision(Effect.ALLOW, Reason.MATCHED_ALLOW, matched[Effect.ALLOW])
        return self._decision(Effect.DENY, Reason.DEFAULT_DENY)

    def _decision(self, effect: Effect, reason: Reason, rule_ids: Iterable[str] = ()) -> Decision:
        return Decision(
            effect=effect, reason=reason, rule_ids=tuple(rule_ids), policy_version=self._version
        )

    def _builtin_check(self, request: Request) -> Reason | None:
        customer_id = self._bundle.customer_id
        if request.customer_id != customer_id:
            return Reason.CUSTOMER_MISMATCH
        tool = self._bundle.tools.get(request.tool)
        if tool is None:
            return Reason.UNKNOWN_TOOL
        if request.action not in tool.actions:
            return Reason.ACTION_NOT_SUPPORTED
        document = request.document
        if document is None:
            if request.action in DOCUMENT_ACTIONS:
                return Reason.DOCUMENT_REQUIRED
        else:
            if document.customer_id != customer_id:
                return Reason.DOCUMENT_CUSTOMER_MISMATCH
            if document.label not in self._ranks:
                return Reason.UNKNOWN_LABEL
        if request.run_max_label is not None and request.run_max_label not in self._ranks:
            return Reason.UNKNOWN_LABEL
        return None

    def _matches(self, rule: Rule, request: Request) -> bool:
        if rule.roles is not None and request.role not in rule.roles:
            return False
        if request.tool not in rule.tools or request.action not in rule.actions:
            return False
        if rule.labels is not None:
            document = request.document
            if document is None:
                return False
            if self._ranks[document.label] > self._ranks[rule.labels.max]:
                return False
        return self._conditions_hold(rule.where, request)

    def _conditions_hold(self, where: Conditions, request: Request) -> bool:
        if where.recipient_in is not None and not self._all_in_list(
            where.recipient_in, request.recipients
        ):
            return False
        if where.recipient_external is not None:
            if not request.recipients:
                return False
            any_external = any(self._is_external(raw) for raw in request.recipients)
            if any_external is not where.recipient_external:
                return False
        if where.amount_usd_gt is not None or where.amount_usd_lte is not None:
            amount = _spend_amount(request.amount_usd)
            if amount is None:
                return False
            if where.amount_usd_gt is not None and not amount > where.amount_usd_gt:
                return False
            if where.amount_usd_lte is not None and not amount <= where.amount_usd_lte:
                return False
        if where.run_max_label_gte is not None:
            run_label = request.run_max_label
            if run_label is None or self._ranks[run_label] < self._ranks[where.run_max_label_gte]:
                return False
        document = request.document
        if where.doc_tags_any is not None and (
            document is None or not (document.tags & where.doc_tags_any)
        ):
            return False
        if where.doc_assigned_to_user is not None:
            if document is None:
                return False
            assigned = request.user_id in document.assigned_user_ids
            if assigned is not where.doc_assigned_to_user:
                return False
        return True

    def _all_in_list(self, name: str, recipients: tuple[str, ...]) -> bool:
        listed = self._lists.get(name)
        if listed is None or not recipients:
            return False
        return all(listed.contains(raw) for raw in recipients)

    def _is_external(self, raw: str) -> bool:
        parsed = _parse_address(raw)
        # Anything that is not one plain address is treated as leaving the company.
        return parsed is None or parsed[1] not in self._internal_domains


def _spend_amount(amount: Decimal | None) -> Decimal | None:
    """The amount as a fact amount conditions can use, or None if it is missing or invalid."""
    # NaN and infinities would raise or compare oddly, and a negative amount is never a
    # legitimate spend, so all of them count as a missing amount and satisfy no bound.
    if amount is None or not amount.is_finite() or amount < 0:
        return None
    return amount
