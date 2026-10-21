"""Core domain types: customers, users, roles, tools and documents.

All types are immutable and reject unknown fields, so a typo in a fixture or a
policy-adjacent file fails loudly instead of silently granting or hiding access.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64)]
ToolId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$", max_length=64)]
Label = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=32)]
Tag = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9:_-]*$", max_length=64)]
Domain = Annotated[str, Field(pattern=r"^[a-z0-9-]+(\.[a-z0-9-]+)+$", max_length=253)]


class Role(StrEnum):
    OWNER = "owner"
    MANAGER = "manager"
    STAFF = "staff"
    CONTRACTOR = "contractor"


class Action(StrEnum):
    READ = "read"
    WRITE = "write"
    SEND = "send"
    SPEND = "spend"


class UserStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Customer(_Model):
    id: Slug
    name: str = Field(min_length=1, max_length=200)
    domains: tuple[Domain, ...] = Field(min_length=1)
    timezone: str

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value


class User(_Model):
    id: Slug
    customer_id: Slug
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(pattern=r"^[^@\s]+@[a-z0-9-]+(\.[a-z0-9-]+)+$", max_length=254)
    role: Role
    status: UserStatus = UserStatus.ACTIVE

    @property
    def email_domain(self) -> str:
        return self.email.rsplit("@", 1)[1]


class Tool(_Model):
    id: ToolId
    description: str = Field(min_length=1, max_length=500)
    actions: frozenset[Action] = Field(min_length=1)
    customer_facing: bool = False
    base_risk: int = Field(ge=0, le=100)
    cost_usd: Decimal = Field(ge=0, max_digits=12, decimal_places=6)


class Document(_Model):
    id: Slug
    customer_id: Slug
    title: str = Field(min_length=1, max_length=300)
    label: Label
    tags: frozenset[Tag] = frozenset()
    assigned_user_ids: frozenset[Slug] = frozenset()
    body: str

    @model_validator(mode="after")
    def _assigned_doc_is_tagged(self) -> "Document":
        # Assignment is an access grant, so it must be visible in the tags too.
        if self.assigned_user_ids and "assigned" not in self.tags:
            raise ValueError("documents with assigned_user_ids must carry the 'assigned' tag")
        return self
