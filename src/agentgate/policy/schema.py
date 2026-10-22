"""Policy file format: the types behind policies/<customer-id>/policy.yaml and tools.yaml.

A policy is plain data. Conditions come from a closed set of named operators with no
expression language, so every policy can be validated up front and evaluated
deterministically. Anything the schema does not recognise is rejected, never ignored.
"""

from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agentgate._yaml import parse_yaml
from agentgate.models import Action, Label, Role, Slug, Tag, Tool, ToolId

API_VERSION = "agentgate/v1"

RuleId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64)]
ListName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
# An exact email address, or "@domain" to match every address at that domain.
ListEntry = Annotated[
    str, Field(pattern=r"^([^@\s]+@|@)[a-z0-9-]+(\.[a-z0-9-]+)+$", max_length=254)
]
Usd = Annotated[Decimal, Field(ge=0, max_digits=12, decimal_places=6)]


class PolicyError(ValueError):
    """A policy or tools file is missing, malformed or internally inconsistent."""


class Effect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LabelBound(_Model):
    max: Label


class Conditions(_Model):
    """Extra facts a request must satisfy for a rule to match. Unset means no constraint."""

    recipient_in: ListName | None = None
    recipient_external: bool | None = None
    amount_usd_gt: Usd | None = None
    amount_usd_lte: Usd | None = None
    run_max_label_gte: Label | None = None
    doc_tags_any: frozenset[Tag] | None = Field(default=None, min_length=1)
    doc_assigned_to_user: bool | None = None

    def set_names(self) -> frozenset[str]:
        return frozenset(name for name, value in self if value is not None)


# Which actions each condition can be evaluated against. A condition on an action that never
# carries the fact (a recipient on a read, say) would silently never match, so it is an error.
CONDITION_ACTIONS: dict[str, frozenset[Action]] = {
    "recipient_in": frozenset({Action.SEND}),
    "recipient_external": frozenset({Action.SEND}),
    "amount_usd_gt": frozenset({Action.SPEND}),
    "amount_usd_lte": frozenset({Action.SPEND}),
    "run_max_label_gte": frozenset(Action),
    "doc_tags_any": frozenset({Action.READ, Action.WRITE}),
    "doc_assigned_to_user": frozenset({Action.READ, Action.WRITE}),
}
DOCUMENT_ACTIONS = frozenset({Action.READ, Action.WRITE})


class Rule(_Model):
    id: RuleId
    effect: Effect
    description: str | None = Field(default=None, max_length=500)
    # Omitted roles means every role. Only deny rules may omit them, so a missing line can
    # never widen access.
    roles: frozenset[Role] | None = Field(default=None, min_length=1)
    tools: frozenset[ToolId] = Field(min_length=1)
    actions: frozenset[Action] = Field(min_length=1)
    labels: LabelBound | None = None
    where: Conditions = Conditions()

    @model_validator(mode="after")
    def _consistent(self) -> "Rule":
        if self.roles is None and self.effect is not Effect.DENY:
            raise ValueError(f"rule {self.id!r}: {self.effect} rules must list roles")
        if self.labels is not None and not self.actions <= DOCUMENT_ACTIONS:
            raise ValueError(f"rule {self.id!r}: labels apply only to read and write actions")
        for name in self.where.set_names():
            if not self.actions <= CONDITION_ACTIONS[name]:
                allowed = ", ".join(sorted(CONDITION_ACTIONS[name]))
                raise ValueError(f"rule {self.id!r}: {name} applies only to {allowed} actions")
        return self


class SpendLimits(_Model):
    per_run_usd: Usd
    per_customer_day_usd: Usd
    max_calls_per_run: int = Field(ge=1, le=10_000)


class ApprovalSettings(_Model):
    approver_roles: frozenset[Role] = Field(min_length=1)
    ttl_minutes: int = Field(ge=1, le=7 * 24 * 60)
    risk_threshold: int = Field(ge=0, le=100)


class Policy(_Model):
    api_version: Literal["agentgate/v1"] = Field(alias="apiVersion")
    customer: Slug
    # Ordered from least to most sensitive.
    labels: tuple[Label, ...] = Field(min_length=1)
    default: Literal["deny"]
    spend: SpendLimits
    approvals: ApprovalSettings
    lists: dict[ListName, frozenset[ListEntry]] = {}
    rules: tuple[Rule, ...] = Field(min_length=1)

    @field_validator("labels")
    @classmethod
    def _unique_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        return labels

    @model_validator(mode="after")
    def _references_resolve(self) -> "Policy":
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id {rule.id!r}")
            seen.add(rule.id)
            if rule.labels is not None and rule.labels.max not in self.labels:
                raise ValueError(f"rule {rule.id!r}: unknown label {rule.labels.max!r}")
            gte = rule.where.run_max_label_gte
            if gte is not None and gte not in self.labels:
                raise ValueError(f"rule {rule.id!r}: unknown label {gte!r}")
            listed = rule.where.recipient_in
            if listed is not None and listed not in self.lists:
                raise ValueError(f"rule {rule.id!r}: unknown list {listed!r}")
        return self

    def label_rank(self, label: str) -> int | None:
        """Position of label in the sensitivity order, or None if the policy does not know it."""
        try:
            return self.labels.index(label)
        except ValueError:
            return None


class PolicyBundle(_Model):
    """One customer's policy plus the tools it governs, validated against each other."""

    policy: Policy
    tools: dict[ToolId, Tool]

    @model_validator(mode="after")
    def _rules_match_tools(self) -> "PolicyBundle":
        for rule in self.policy.rules:
            for tool_id in sorted(rule.tools):
                tool = self.tools.get(tool_id)
                if tool is None:
                    raise ValueError(f"rule {rule.id!r}: unknown tool {tool_id!r}")
                unsupported = rule.actions - tool.actions
                if unsupported:
                    names = ", ".join(sorted(unsupported))
                    raise ValueError(f"rule {rule.id!r}: tool {tool_id!r} does not support {names}")
        return self

    @property
    def customer_id(self) -> str:
        return self.policy.customer


def _validate[M: BaseModel](model: type[M], data: object, path: Path) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            _describe(err["loc"], err["msg"], model.__name__)
            for err in exc.errors(include_url=False, include_input=False)
        )
        raise PolicyError(f"{path}: {problems}") from exc


def _describe(loc: tuple[int | str, ...], msg: str, fallback: str) -> str:
    where = ".".join(str(part) for part in loc) or fallback
    return f"{where}: {_message(msg)}"


def _message(msg: str) -> str:
    # Keep error lines short and free of input values, which could include document text.
    return msg.removeprefix("Value error, ")


def load_bundle(directory: Path) -> PolicyBundle:
    """Load and cross-check policy.yaml and tools.yaml from one customer directory."""
    policy_path = directory / "policy.yaml"
    tools_path = directory / "tools.yaml"
    for path in (policy_path, tools_path):
        if not path.is_file():
            raise PolicyError(f"{path}: missing {path.name}")

    policy = _validate(
        Policy, parse_yaml(policy_path.read_text(), policy_path, PolicyError), policy_path
    )
    if policy.customer != directory.name:
        raise PolicyError(
            f"{policy_path}: customer {policy.customer!r} does not match directory"
            f" {directory.name!r}"
        )

    raw_tools = parse_yaml(tools_path.read_text(), tools_path, PolicyError)
    if not isinstance(raw_tools, list) or not raw_tools:
        raise PolicyError(f"{tools_path}: expected a non-empty list of tools")
    tools: dict[str, Tool] = {}
    for entry in raw_tools:
        tool = _validate(Tool, entry, tools_path)
        if tool.id in tools:
            raise PolicyError(f"{tools_path}: duplicate tool id {tool.id!r}")
        tools[tool.id] = tool

    try:
        return PolicyBundle(policy=policy, tools=tools)
    except ValidationError as exc:
        problems = "; ".join(_message(err["msg"]) for err in exc.errors())
        raise PolicyError(f"{directory}: {problems}") from exc


def load_bundles(root: Path) -> dict[str, PolicyBundle]:
    """Load every customer directory under root, keyed by customer id."""
    bundles: dict[str, PolicyBundle] = {}
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and not directory.name.startswith("."):
            bundle = load_bundle(directory)
            bundles[bundle.customer_id] = bundle
    return bundles
