from decimal import Decimal

import pytest
from pydantic import ValidationError

from agentgate.models import Action, Customer, Document, Role, Tool, User, UserStatus


def _customer(**overrides: object) -> Customer:
    data: dict[str, object] = {
        "id": "acme",
        "name": "Acme",
        "domains": ["acme.example"],
        "timezone": "America/Chicago",
    }
    return Customer.model_validate(data | overrides)


def test_customer_rejects_unknown_timezone() -> None:
    with pytest.raises(ValidationError, match="unknown timezone"):
        _customer(timezone="Mars/Olympus")


def test_customer_requires_a_domain() -> None:
    with pytest.raises(ValidationError):
        _customer(domains=[])


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _customer(is_admin=True)


def test_models_are_immutable() -> None:
    customer = _customer()
    with pytest.raises(ValidationError):
        customer.name = "Other"  # type: ignore[misc]


def test_user_defaults_to_active_and_exposes_email_domain() -> None:
    user = User(
        id="dana",
        customer_id="acme",
        name="Dana",
        email="dana@acme.example",
        role=Role.STAFF,
    )
    assert user.status is UserStatus.ACTIVE
    assert user.email_domain == "acme.example"


def test_user_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        User.model_validate(
            {
                "id": "dana",
                "customer_id": "acme",
                "name": "Dana",
                "email": "dana@acme.example",
                "role": "admin",
            }
        )


@pytest.mark.parametrize("risk", [-1, 101])
def test_tool_risk_is_bounded(risk: int) -> None:
    with pytest.raises(ValidationError):
        Tool(
            id="email.send",
            description="Send an email",
            actions=frozenset({Action.SEND}),
            base_risk=risk,
            cost_usd=Decimal("0"),
        )


def test_tool_rejects_negative_cost_and_empty_actions() -> None:
    with pytest.raises(ValidationError):
        Tool(
            id="email.send",
            description="Send an email",
            actions=frozenset({Action.SEND}),
            base_risk=10,
            cost_usd=Decimal("-0.01"),
        )
    with pytest.raises(ValidationError):
        Tool(
            id="email.send",
            description="Send an email",
            actions=frozenset(),
            base_risk=10,
            cost_usd=Decimal("0"),
        )


def test_tool_id_must_be_namespaced() -> None:
    with pytest.raises(ValidationError):
        Tool(
            id="send",
            description="Send",
            actions=frozenset({Action.SEND}),
            base_risk=10,
            cost_usd=Decimal("0"),
        )


def test_assigned_document_must_carry_assigned_tag() -> None:
    with pytest.raises(ValidationError, match="assigned"):
        Document(
            id="job-1",
            customer_id="acme",
            title="Job 1",
            label="internal",
            assigned_user_ids=frozenset({"dana"}),
            body="...",
        )
    doc = Document(
        id="job-1",
        customer_id="acme",
        title="Job 1",
        label="internal",
        tags=frozenset({"assigned"}),
        assigned_user_ids=frozenset({"dana"}),
        body="...",
    )
    assert "dana" in doc.assigned_user_ids
