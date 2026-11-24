"""Checks on values a caller chose, before they are looked up or recorded.

A value SQLite cannot bind (a lone surrogate from a JSON escape) names nothing, so a lookup
of it is unknown rather than an exception.
"""

import re
from typing import TypeGuard

from agentgate.approvals import Approval, ApprovalQueue
from agentgate.audit.panoptes import storable_identifier
from agentgate.domain import Action, Customer, Role, User
from agentgate.policy.addresses import parse_address
from agentgate.store import Store

# Values the model chose (a tool name) go into events only in this shape, which cannot
# carry free text or a canary.
TOOL_ID = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
APPROVAL_ID = re.compile(r"^apr_[A-Za-z0-9_-]{1,64}$")
# A tool's effect_ref, and a correlation id a client sent, go into events as written only in
# this id shape; anything else is recorded as a reference.
PLAIN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# Customer id for audit events about a caller that matches no customer at all.
UNKNOWN_CUSTOMER = "unknown"


def plain_id(value: str | None) -> str | None:
    """value if it may be recorded as written: a plain id shape the audit log accepts."""
    if value is not None and PLAIN_ID.fullmatch(value) and storable_identifier(value):
        return value
    return None


def is_external(raw: str, domains: frozenset[str]) -> bool:
    # The engine's and the email tool's address rule: anything that is not one plain
    # address (a look-alike such as the Kelvin sign, a control character, a second "@")
    # counts as leaving the company.
    parsed = parse_address(raw)
    return parsed is None or parsed[1] not in domains


def as_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def is_lookupable(value: object) -> TypeGuard[str]:
    """Whether value is a string SQLite can bind, so a lookup cannot raise."""
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def safe_approval_id(value: object) -> str | None:
    """The approval id for an event subject when it has the approval-id shape, else None."""
    return value if isinstance(value, str) and APPROVAL_ID.fullmatch(value) else None


def safe_role(value: object) -> Role | None:
    try:
        return Role(value) if isinstance(value, str) else None
    except ValueError:
        return None


def safe_action(value: object) -> Action | None:
    try:
        return Action(value) if isinstance(value, str) else None
    except ValueError:
        return None


def get_user(store: Store, user_id: object) -> User | None:
    return store.get_user(user_id) if is_lookupable(user_id) else None


def get_customer(store: Store, customer_id: object) -> Customer | None:
    return store.get_customer(customer_id) if is_lookupable(customer_id) else None


def get_approval(queue: ApprovalQueue, approval_id: object) -> Approval | None:
    return queue.get(approval_id) if is_lookupable(approval_id) else None
