"""Checks on the committed seed data that later phases rely on."""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentgate.fixtures import find_canaries, load_fixtures
from agentgate.models import Role, UserStatus
from agentgate.store import Store

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
PREFIXES = {"kettle-creek-hvac": "kc-", "juniper-row-apothecary": "jr-"}


@pytest.fixture(scope="module")
def store() -> Iterator[Store]:
    with Store() as s:
        load_fixtures(s, FIXTURES)
        yield s


def test_both_customers_load(store: Store) -> None:
    assert [c.id for c in store.list_customers()] == sorted(PREFIXES)


@pytest.mark.parametrize("customer_id", sorted(PREFIXES))
def test_customer_covers_every_role_and_a_suspended_user(store: Store, customer_id: str) -> None:
    users = store.list_users(customer_id)
    assert {u.role for u in users} == set(Role)
    assert any(u.status is UserStatus.SUSPENDED for u in users)


@pytest.mark.parametrize("customer_id", sorted(PREFIXES))
def test_ids_carry_the_customer_prefix(store: Store, customer_id: str) -> None:
    prefix = PREFIXES[customer_id]
    assert all(u.id.startswith(prefix) for u in store.list_users(customer_id))
    assert all(d.id.startswith(prefix) for d in store.list_documents(customer_id))


@pytest.mark.parametrize("customer_id", sorted(PREFIXES))
def test_customer_has_every_label(store: Store, customer_id: str) -> None:
    labels = {d.label for d in store.list_documents(customer_id)}
    assert labels == {"public", "internal", "confidential", "restricted"}


@pytest.mark.parametrize("customer_id", sorted(PREFIXES))
def test_contractor_has_assigned_and_unassigned_work(store: Store, customer_id: str) -> None:
    contractor = next(u for u in store.list_users(customer_id) if u.role is Role.CONTRACTOR)
    assigned = [d for d in store.list_documents(customer_id) if "assigned" in d.tags]
    mine = [d for d in assigned if contractor.id in d.assigned_user_ids]
    others = [d for d in assigned if contractor.id not in d.assigned_user_ids]
    assert mine, "contractor needs work they may read"
    assert others, "contractor needs assigned work they may not read"


@pytest.mark.parametrize("customer_id", sorted(PREFIXES))
def test_untrusted_doc_targets_an_outside_example_address(store: Store, customer_id: str) -> None:
    customer = store.get_customer(customer_id)
    assert customer is not None
    untrusted = [d for d in store.list_documents(customer_id) if "untrusted" in d.tags]
    assert untrusted
    for doc in untrusted:
        assert doc.label in {"public", "internal"}
        assert not find_canaries(doc.body)
        domains = set(re.findall(r"@([a-z0-9.-]+\.example)\b", doc.body))
        assert domains - set(customer.domains), "injection needs an outside address"
