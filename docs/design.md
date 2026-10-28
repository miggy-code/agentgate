# Design and reference

The README covers what agentgate is and how a call flows through it. This page has the detail: each subsystem, integration, the MCP gateway, the policy format, the CLI, latency and the optional real model.

## Permission-aware retrieval

Search filters before it ranks. Each search asks the policy engine about every one of the customer's documents, from metadata only, never the body. Only documents the engine allows outright are searchable. BM25F then ranks the permitted set, with term statistics computed from that set alone, so forbidden documents do not shift scores. Each ranked result is checked again before it is returned; a disagreement is an invariant violation and the search returns nothing.

A denied read looks like a missing document. Whether the id does not exist, belongs to another customer, or names a document the user may not read, the model is told `not_found`. The audit log keeps the real reason. `audit show`, which the customer's own owners and managers read, prints a denial naming another customer's document exactly as it prints a missing one (`document_not_found`, the id as an unknown reference, no args hash), so it cannot be used to find which ids exist elsewhere.

## Approvals

- An approval is bound to the exact tool, action and a hash of the arguments. Executing with changed arguments is denied `args_mismatch`.
- It runs at most once. A replay is denied `already_executed`.
- It expires after the policy's TTL (60 minutes in both shipped policies).
- At execution the gate decides again with the current policy and the run's current highest label. A deny wins over the approval.
- Only the policy's approver roles (owner and manager in both policies) can decide, and a requester cannot approve their own request. An approval from another customer, or from another run of the same user, reads as `not_found`.
- Every state change (created, approved, rejected, expired, claimed) is recorded inside the transaction that makes it, with the approval's own args hash, so `agentgate audit check-store` can tell an approval row rewritten after the fact.

## Spend caps

Each policy sets three caps: USD per run, USD per customer per day, and tool plus model calls per run. The day is the customer's local day, from the customer's timezone. The gate reserves before every tool call, and the reference agent loop reserves before every model call. Reservations are settled at the actual cost. Calls that are denied or sent for approval give their reservation back and do not count toward the call cap; every model call counts, including one that failed.

A reservation is a lease. One still open 15 minutes after it was made, such as one left by a host that died between reserve and settle, is released on the next `open_world` of that database (the approvals, users and gateway commands each do one), or by `agentgate spend release-stale`. A released lease no longer counts toward the run or the customer day, but still counts toward the run's call cap. If the host was only slow and settles later, the actual cost is recorded.

## Recipients

`src/agentgate/policy/addresses.py` is the one address rule the policy engine, the gate and the email tool share. A recipient is one plain address only if every character is printable ASCII (checked before any case mapping, so control, bidi and zero-width characters and look-alikes such as the Kelvin sign are refused), it has exactly one `@`, its local part is dot-separated runs of letters, digits and `_ + - '` (so no `%` relay or bang-path forms, no quotes and no backslash), and its domain has two or more labels. It is stored and sent lowercased, so the address the policy checked is the address in the outbox. Anything else is malformed: the engine treats it as outside the company and on no list, and the email tool refuses to store or send it even if a policy or an approver allowed it.

## Audit

The default sink is [Panoptes](https://pypi.org/project/panoptes-observe/) (`panoptes-observe`): a hash-chained SQLite log in which each row's hash covers the row and the previous hash. Every decision is recorded before anything runs. Events carry ids, labels, counts, reason codes, rule ids and the policy version hash, never document bodies, message text, notes or search queries. If a decision cannot be recorded, the action does not happen (see [Limits](limits.md#audit) for the events recorded after their change).

So that an incident can be reconstructed from the log, decision events also carry values derived from the arguments, never the text: the args hash of a call whose arguments validated (sha256 over the tool, the action and the validated arguments, the hash approvals are bound to), for a send the number of recipients inside and outside the company, for a spend its amount, and for an executed call the id of the row it wrote (`effect_ref`). Approval events carry the approval's own args hash. A search carries no args hash, since its query is free text.

A value a caller chose that names nothing the store knows (an invented document slug, an unknown approver, a claimed run id), or that Panoptes would refuse (an id shaped like an API key), is recorded as a keyed reference: `ref:` plus the first 16 hex characters of an HMAC-SHA256 whose key is kept in the store. The same value always gives the same reference, so repeated probes can be counted, and the log never holds the value itself. A call with a key-shaped id is denied and audited like any other; if Panoptes still refuses an event for its content, a copy with every caller value as a reference is recorded instead.

Every open of a world stores the canonical bundle behind each customer's policy version in the store's `policy_snapshots` table, and a `policy_loaded` event records the version and the bundle's sha256, so `agentgate audit policy` can print the rules behind any version an event names. Operator actions are `operator_action` events: a fixture sync that changed something, a suspension, a reinstatement, and a stale spend sweep (on open, only one that released anything; `spend release-stale` always records its count).

How much of an open goes on the chain depends on who opens it (`OpenRecord` in `src/agentgate/world.py`). CLI commands that change state (`approvals approve`, `reject` and `expire`, `users suspend` and `reinstate`, `spend release-stale`) record everything, including a `policy_loaded` event per customer. The read-only `approvals list`, `approvals show` and `users list`, and `gateway serve`, record only what the open changed: a policy version stored for the first time, a fixture sync, a sweep. `audit show`, `audit policy` and `audit check-store` open the store read-only and record nothing. `open_world` records nothing about the open unless the host passes `record_open`, and in-memory worlds (`open_memory_world`, used by `agent run`, the demo and the red-team suite) record nothing about the open.

## Integration

agentgate runs in one of two ways: as in-process middleware, a Python library the host application calls, or as an MCP gateway in its own process.

In-process, the host builds a world (store, fixtures, index, policies, audit sink, gate) with `open_world`, starts a run for a user it has already authenticated, and sends every tool call the model proposes through `call_tool`:

```python
from pathlib import Path

from agentgate.models import Action
from agentgate.world import open_world

with open_world(
    db_path=Path("data/agentgate.db"),
    audit_path=Path("data/audit.db"),
    policies=Path("policies"),
    fixtures=Path("fixtures"),
) as world:
    # The host has already authenticated this user. agentgate does not.
    ctx = world.gate.start_run("kettle-creek-hvac", "kc-staff-nadine")

    # Every tool call the model proposes goes through the gate.
    outcome = world.gate.call_tool(
        ctx, "docs.read", Action.READ, {"document_id": "kc-payroll-summary"}
    )
    print(outcome)  # Denied(reason='not_found'): staff may not read restricted documents
```

`call_tool` returns `Executed`, `Denied`, `PendingApproval` or `CapReached`. An approved call runs through `gate.execute_approved(ctx, approval_id, tool, action, args)`. Model spend goes through `gate.charge_model` and `gate.settle_model`. `src/agentgate/agent/loop.py` is a reference loop that does all of this. `open_memory_world` builds the same world in memory with an in-memory audit sink.

Opening a world on disk syncs the fixtures into the store (new customers, users and documents are added and changed ones updated; nothing is deleted, and a user no longer in the fixtures is suspended), so a suspension, role change or relabelling in the fixtures takes effect on the next open. It also releases stale spend reservations and refuses a store path that is the audit log or the same file as it. `open_world` records nothing about the open itself on the audit chain unless `record_open` is passed (see [Audit](#audit)).

### The MCP gateway

In-process middleware only protects tools that are reachable solely through the gate. The gateway moves the tools, the database and their side effects into their own process, so an agent gets nothing but an MCP connection. Code is in `src/agentgate/gateway/server.py`.

`agentgate gateway serve --as USER` serves MCP over stdio. It opens the same on-disk world as the CLI (`data/agentgate.db` and `data/audit.db` by default) and starts one run for that user before it serves anything; every call on the connection belongs to that run. The identity is bound at launch: the MCP client never names a user, role or customer, no tool takes an identity argument, and an extra key such as `user_id` or `role` is denied `invalid_args`. `--as` is the same demo identity as on the CLI: it names a fixture user, it is not a login, and the gateway prints the same note on stderr. An unknown or suspended user exits 1 before anything is written to stdout. One gateway process serves one connection and one run.

The gateway lists the run customer's tools under MCP-safe names (`docs.search` becomes `docs_search`), plus `approvals_execute`. For a Kettle Creek user such as `kc-staff-nadine` the list is `docs_read`, `docs_search`, `email_send`, `notes_write`, `parts_order` and `approvals_execute`. The action of each call is taken from that tool's entry in the customer's tool list, never from the client. A name the gateway does not list goes to the gate unchanged and is denied there as an unknown tool.

`approvals_execute` runs a call a person has already approved, with exactly the approved tool and arguments, through `gate.execute_approved`, which checks the run, the arguments hash, the expiry and the current policy again. It cannot approve anything: approvals are decided out of band with `agentgate approvals approve`. An approval belongs to the run that requested it, so it is executed on the same gateway connection.

Every call and every refusal goes through the gate and is recorded there with the bound run's identity; the gateway refuses nothing itself. An `approvals_execute` request whose fields cannot be read (a missing field, a wrong type, an extra key) goes to `gate.refuse`, which checks the run and user like any call and records an `invalid_args` denial. Calls are decided one at a time, in order. The client gets the outcome the reference loop shows its model, as JSON text: the tool's content, `{"error": reason}`, a pending approval's id and expiry, or the cap that was reached. Two kinds of request end without a gate record; see [Limits](limits.md).

The tools, the database and every side effect live in the gateway process, as would the credentials of real tools. An agent that holds only the MCP connection has no path to a tool except through the gate.

An example MCP client configuration, in the `mcpServers` format many clients use for stdio servers (replace the path with your checkout):

```json
{
  "mcpServers": {
    "agentgate": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/agentgate", "agentgate", "gateway", "serve", "--as", "kc-staff-nadine"]
    }
  }
}
```

`uv run --directory` runs the command from the checkout, so the default `data/`, `policies/` and `fixtures/` paths resolve there. At commit `b9e650b` this command line, launched from another directory with `--db` and `--audit-db` added to point at a scratch folder, answered `initialize`, `tools/list` and a `docs_read` over stdio and wrote 7 audit events: a `policy_loaded` and a `fixture_sync` event for each of the two customers on the first open, then `run_started`, `allow` and `executed`. `tests/test_gateway.py::test_cli_serves_over_stdio` launches `agentgate gateway serve --as kc-staff-nadine` as a subprocess and checks the same on every test run.

## Policy format

Each customer has a folder under `policies/` with three files: `policy.yaml` (rules, caps, approvals, lists), `tools.yaml` (the tools the customer has) and `tests.yaml` (expected decisions). A trimmed excerpt of `policies/kettle-creek-hvac/policy.yaml`:

```yaml
# Access policy for Kettle Creek Heating & Air. Deny by default: anything no rule allows is denied.
# Precedence: any matching deny wins, then require_approval, then allow.
apiVersion: agentgate/v1
customer: kettle-creek-hvac

# Least to most sensitive.
labels: [public, internal, confidential, restricted]
default: deny

# Platform cost caps for agent runs.
spend:
  per_run_usd: "0.50"
  per_customer_day_usd: "20.00"
  max_calls_per_run: 40

approvals:
  approver_roles: [owner, manager]
  ttl_minutes: 60
  risk_threshold: 50

lists:
  # Company mailboxes only. Exact domain, so subdomains and look-alikes do not match.
  internal_addresses:
    - "@kettlecreek.example"
  # Exact customer addresses from the customer list. Senders seen only in untrusted inbox
  # mail are not added, and no shared consumer domain is listed with @domain.
  customer_directory:
    - beatrice.l@mail-sample.example
    - ruben.c@mail-sample.example

rules:
  - id: contractor-read-assigned
    effect: allow
    description: Contractors may read internal documents assigned to them, such as their job tickets.
    roles: [contractor]
    tools: [docs.search, docs.read]
    actions: [read]
    labels: {max: internal}
    where:
      doc_assigned_to_user: true

  - id: email-customers
    effect: require_approval
    description: Emailing customers in the directory needs approval.
    roles: [owner, manager, staff]
    tools: [email.send]
    actions: [send]
    where:
      recipient_in: customer_directory

  - id: no-external-after-confidential
    effect: deny
    description: No run may email outside the company once it has read confidential or restricted data.
    tools: [email.send]
    actions: [send]
    where:
      run_max_label_gte: confidential
      recipient_external: true
```

The full file has 7 directory addresses and 14 rules.

### Rule semantics

- A matching `deny` beats `require_approval`, which beats `allow`. A request no rule matches is denied (`default: deny`).
- `allow` and `require_approval` rules must list `roles`. Only a `deny` rule may omit them (it then applies to every role), so a missing line can never widen access.
- A rule matches on `roles`, `tools`, `actions` and `labels: {max: ...}` (the document's label is at or below the bound), plus optional `where` conditions.
- A condition whose fact is missing never holds. A condition on an action that never carries its fact (a recipient on a read) is a validation error.
- There is no expression language. The conditions are a closed set:

| Condition | Holds when |
| --- | --- |
| `recipient_in: <list>` | every recipient is on the named list (send) |
| `recipient_external: true` | at least one recipient is outside the customer's domains (send) |
| `amount_usd_gt: "<usd>"` | the amount is above the value (spend) |
| `amount_usd_lte: "<usd>"` | the amount is at or below the value (spend) |
| `run_max_label_gte: <label>` | the run has already been shown a document at or above this label (any action) |
| `doc_tags_any: [<tag>]` | the document has any of the tags (read, write) |
| `doc_assigned_to_user: true` | the document is assigned to the acting user (read, write) |

### tools.yaml

Each shipped tool has exactly one action, a base risk for the risk score, an optional `customer_facing` flag, and a per-call platform cost counted against the spend caps. From `policies/kettle-creek-hvac/tools.yaml`:

```yaml
- id: email.send
  description: Send an email from the company mailbox.
  actions: [send]
  customer_facing: true
  base_risk: 40
  cost_usd: "0.001"
```

The risk score is the tool's `base_risk`, plus 25 if any recipient is outside the company, plus 5 per full 100 USD of the amount, up to 40.

### tests.yaml

Each case names a fixture user (role and customer come from the fixtures, never from the case), a request, the expected effect, and the deciding rule ids. One case from `policies/kettle-creek-hvac/tests.yaml`:

```yaml
- name: owner emails the injection target after reading confidential data
  user: kc-owner-dale
  tool: email.send
  action: send
  recipients: [records-review@exfil-drop.example]
  run_max_label: confidential
  expect: deny
  rules: [no-external-after-confidential]
```

### Policy version

A policy's version is a content hash of the validated bundle (policy and tools), so comments and key order do not change it and any change in meaning does. Every audit event records it, and the bundle behind it is kept in the store (see [Audit](#audit)).

```console
$ uv run agentgate policy lint
juniper-row-apothecary  sha256:091571d97a0826e8  15 rules  5 tools  git 87a4801
kettle-creek-hvac  sha256:66ca6bb217693239  14 rules  5 tools  git 87a4801

$ uv run agentgate policy test
juniper-row-apothecary  sha256:091571d97a0826e8  59/59 passed
kettle-creek-hvac  sha256:66ca6bb217693239  67/67 passed
total  126/126 passed
```

Both outputs are from commit `87a4801`.

## Latency

From `uv run agentgate bench` at commit `05456e9` (2000 timed iterations after 200 warmup, times in microseconds). Machine: Apple M4 Pro, 12 CPUs, macOS 15.7.3, Python 3.12.12, with desktop apps running (load average about 4). The store is in-memory SQLite.

| Measurement | p50 | p95 |
| --- | ---: | ---: |
| `engine.decide`, allowed read (decision only) | 2.5 | 2.6 |
| Gated `docs.read`, added over a direct call, MemorySink | +161.3 | +178.7 |
| Gated `docs.read`, added over a direct call, PanoptesSink | +3209.5 | +4659.0 |
| Gated `docs.search`, added over a direct index search | +177.8 | +216.2 |
| Denied call through the gate | 62.9 | 73.9 |
| Gated `docs.read` on a busy day (1000 earlier reservations) | 179.4 | 207.1 |

"Added" is the median minus the median (and p95 minus p95). The durable Panoptes audit dominates the cost: an allowed read writes two hash-chained events to a SQLite file. These numbers are machine-dependent; compare them only with runs on the same machine.

## Approvals, users and audit from the CLI

```bash
uv run agentgate approvals list --as kc-mgr-terrence [--status pending]
uv run agentgate approvals show <ID> --as kc-mgr-terrence
uv run agentgate approvals approve <ID> --as kc-mgr-terrence --note "ok to send"
uv run agentgate approvals reject <ID> --as kc-mgr-terrence
uv run agentgate approvals expire --as kc-owner-dale
uv run agentgate users list --as kc-owner-dale
uv run agentgate users suspend kc-staff-nadine --as kc-owner-dale --reason investigation
uv run agentgate users reinstate kc-staff-nadine --as kc-owner-dale --reason resolved
uv run agentgate spend release-stale --as kc-owner-dale [--older-than 15]
uv run agentgate audit show --as kc-mgr-terrence [--run <RUN_ID>] [--kind deny] [--limit 20] [--json]
uv run agentgate audit policy sha256:66ca6bb217693239 --as kc-mgr-terrence
uv run agentgate audit check-store --as kc-owner-dale
uv run agentgate audit verify [--db data/audit.db]
```

`--as` is a demo identity: it names a fixture user and is not a login. Every command that takes `--as` prints `note: --as is a demo identity; agentgate does not authenticate users`. The commands use `data/agentgate.db` (created and seeded from the fixtures if missing, and synced from them on every open) and `data/audit.db` by default. The approvals, users, spend and gateway commands open the world. On stderr they say what the open changed: a fixture sync, released stale reservations, or a store file other users could read, which is tightened to 0600 with a warning. They refuse a `--db` that is a Panoptes audit log or the same file as `--audit-db`, and an `--audit-db` that holds an agentgate store. A database error is one line, `error: database error: ...`, with exit code 1.

`approvals list` and `show` only show the acting approver's customer, and `show` prints the exact arguments that would run. Neither changes an approval: one past its expiry shows as `expired (pending sweep)`, and counts as expired for `--status`, until `approvals expire` marks it. A staff user who tries to approve gets `approver_not_allowed: role 'staff' may not decide approvals`. `expire` expires every overdue approval for all customers and prints only the count; the acting user must be an active owner.

`users list` prints the acting user's customer's users with role and status; a user suspended with `users suspend` shows as `suspended (by operator)`. `users suspend` and `users reinstate` are for owners, and an owner cannot suspend themselves. A suspension takes effect on the user's next gate call in any process that uses the same database, a live run or gateway included. It is sticky: a fixture sync does not reactivate the user until `users reinstate`, after which the fixtures decide the user's status again. Both record the acting owner, the user and the `--reason` code. `spend release-stale` releases reservations open longer than `--older-than` minutes (default 15) for all customers and prints only the count; the acting user must be an active owner.

`audit show` prints the acting user's customer's events, oldest first, one per line, for owners and managers. A value the log holds as a reference prints as its `ref:` and why (`unknown value`: it named nothing the store knew). `audit policy` prints the policy bundle stored for a version and checks its sha256 against the snapshot, the version and every `policy_loaded` event for it, for owners and managers. `audit check-store`, for owners, verifies the chain, then checks each of the customer's approval, outbox, parts order, refund and note rows against the hashes its events carry. Both checks exit 1 on any mismatch. The three `audit` commands open the store read-only and change neither file.

No approvals command queues an approval. A pending approval comes from an agent run against an on-disk world: the host calls `call_tool` as in the integration sample and a call that needs approval returns `PendingApproval`, or an MCP client calls a tool through `agentgate gateway serve`, which uses the same files by default. `agentgate agent run` uses a fresh in-memory world that is discarded when it exits, so an approval it creates cannot be decided afterwards. After approval, the requester's run executes the call through `gate.execute_approved`, or over MCP through `approvals_execute` on the same gateway connection.

`agentgate audit verify` checks the Panoptes hash chain and prints `<path>  N events  chain intact`, or `chain BROKEN: ...` and exits 1. An agentgate log is chained from its first event, so events outside the chain (a log whose hashes were nulled and chain metadata deleted) also fail. It verifies a private snapshot of the file and never writes to it. It has no checkpoint option; see [Limits](limits.md).

## Optional real model

The demo, the red-team suite and CI use mock models. To run the reference agent with Claude:

```bash
uv sync --extra anthropic
export AGENTGATE_MODEL=anthropic
export ANTHROPIC_API_KEY=...
uv run agentgate agent run --as kc-staff-nadine --task "Please email beatrice.l@mail-sample.example about her tune-up." --model anthropic
```

`.env.example` lists the variables. agentgate does not read `.env` files itself, so export them in your shell. The default model is `claude-opus-5-5`; `AGENTGATE_ANTHROPIC_MODEL` changes it. `agent run` prints every gate decision for one task in an in-memory world. Without `--model anthropic` it uses the offline mock model.

Two caveats. The spend caps in the shipped policies are sized for the mock model's synthetic prices, not for real model prices. The adapter is tested only against a fake client built from the SDK's types; no test, CI job or report in this repository has exercised it against the live API.

## Red-team suite over MCP

The same suite through the MCP gateway, with every agent-side tool call sent through an MCP client session to a gateway bound to the run:

| Category | Scenarios | Passed | Pass rate |
| --- | ---: | ---: | ---: |
| cross_tenant | 14 | 14 | 14/14 (100%) |
| privilege_escalation | 15 | 15 | 15/15 (100%) |
| prompt_injection_exfil | 16 | 16 | 16/16 (100%) |
| approval_bypass | 15 | 15 | 15/15 (100%) |
| spend_runaway | 14 | 14 | 14/14 (100%) |
| identity_spoofing | 16 | 16 | 16/16 (100%) |
| fail_closed | 17 | 17 | 17/17 (100%) |
| audit_integrity | 19 | 19 | 19/19 (100%) |
| **attacks total** | 126 | 126 | 126/126 (100%) |
| benign_control | 20 | 20 | 20/20 (100%) |

Produced by `uv run agentgate redteam --transport mcp` at commit `b9e650b`: 146 scenarios, 1552 checks, baseline check 0 problems, calls over MCP: 1361, direct: 58. A call goes to the gate directly only when MCP cannot express it: 29 use a forged run context (an MCP client cannot pass a context at all), 23 use an action the tool does not have (over MCP the action comes from the gateway's own tool list), and 6 belong to a run whose customer has no policy (the gateway will not launch for it). 13 of the 126 attack scenarios make no MCP call: id-005, id-006, id-007, fc-002, fc-003 and pe-014 send only such direct calls; id-001, xt-013, sr-003, sr-006 and sr-007 test only run starts and model spend, which stay in-process because a gateway's identity is bound at launch and it never sees model calls; and fc-016 and fc-017 make no tool call at all, since they test how the store file is opened. Every benign_control scenario makes at least one MCP call. The suite's sessions use the MCP SDK's in-memory transport, one gateway per run; the stdio launch is covered by a test (see [Integration](#integration)).

## Project layout

| Path | Contents |
| --- | --- |
| `src/agentgate/gate.py` | the gate: check, decide, record, then run every tool call |
| `src/agentgate/context.py` | `RunContext`, the identity of one run, created only by the gate |
| `src/agentgate/policy/` | policy schema, engine (deny by default), `tests.yaml` runner, version hash |
| `src/agentgate/policy/addresses.py` | the one recipient address rule, shared by the engine, the gate and the email tool |
| `src/agentgate/retrieval.py` | permission-aware search: filter, BM25F over the permitted set, re-check |
| `src/agentgate/approvals.py` | approval queue bound to tool, action and args hash |
| `src/agentgate/spend.py` | spend ledger: per-run calls and USD, per-customer-day USD |
| `src/agentgate/audit/` | audit event type and keyed references, memory sink, Panoptes sink |
| `src/agentgate/tools/` | tool adapter interface and the built-in tools |
| `src/agentgate/agent/` | model protocol, reference agent loop, mock models, optional Anthropic adapter |
| `src/agentgate/store.py`, `world.py` | SQLite store; `open_world` and `open_memory_world`, user suspension, `check_store` |
| `src/agentgate/fixtures.py` | fixture loading, and the sync applied on every open |
| `src/agentgate/gateway/` | MCP gateway: one run per process, every tool call through the gate |
| `src/agentgate/redteam/` | scenario registry, leak detector, file worlds, baseline check, reports, benchmark, MCP transport |
| `src/agentgate/demo.py`, `cli.py` | `make demo` and the `agentgate` CLI |
| `policies/` | one folder per customer: `policy.yaml`, `tools.yaml`, `tests.yaml` |
| `fixtures/` | the synthetic customers, users and documents |
| `redteam/baseline.json` | scenarios that must exist and pass |
| `tests/` | pytest suite, including mutation tests for the red-team scenarios |
