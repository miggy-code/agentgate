# Design and reference

The README says what agentgate is and how a call goes through it. This page gives the details: each part of the system, integration, the MCP gateway, the model gateway, the policy format, the CLI, latency and the optional real model.

## Permission-aware retrieval

Search filters first and ranks second.

1. For each search, the policy engine decides every document of the customer. It uses only metadata, never the body.
2. Only the documents that the engine allows outright are searchable.
3. BM25F ranks the permitted documents. It computes term statistics from the permitted set only, so forbidden documents do not change the scores.
4. The engine checks each ranked result again before search returns it. If the two decisions disagree, that is an invariant violation, and the search returns nothing.

### Source ACLs

A document can carry the ACL of the system it came from. The `acl` field lists the principals that a Drive file, mailbox or channel is shared with. A user holds principals too: their `memberships`, plus `anyone`, `user:<email>` and `domain:<email domain>`, which come from their address. Principals are opaque strings. agentgate compares them only for equality.

When a document has an ACL, the engine checks it before any rule. If the user holds none of its principals, the engine denies the request with `acl_denied`. The policy cannot change this. So the source ACL is the ceiling, and a policy can only narrow access below it. An empty ACL shares the document with nobody. A document with no `acl` field is decided by the policy alone.

The gate and search take the user's principals from the store's copy of the user. They read it again for every call. Code that calls `PolicyEngine.decide` directly must pass the principals in `Request.principals`. If that field is empty, no document with an ACL is readable.

A denied read looks the same as a missing document. The model gets `not_found` in all of these cases: the id does not exist, the document belongs to another customer, or the user may not read it. The audit log keeps the real reason. The customer's owners and managers read the log with `audit show`. It prints a denial for another customer's document exactly like a missing one: `document_not_found`, the id as an unknown reference, and no args hash. So `audit show` cannot show which ids exist in another customer.

## Approvals

- An approval is bound to the exact tool, the action and a hash of the arguments. If you execute it with changed arguments, the gate denies it with `args_mismatch`.
- An approval runs one time only. The gate denies a replay with `already_executed`.
- An approval expires after the policy's TTL. Both shipped policies use 60 minutes.
- At execution, the gate decides again with the current policy and the run's current highest label. A deny wins over the approval.
- Only the policy's approver roles can decide. In both shipped policies, these are owner and manager. A requester cannot approve their own request. An approval of another customer, or of another run of the same user, reads as `not_found`.
- The gate records each state change (created, approved, rejected, expired, claimed) in the same transaction that makes the change. The event carries the approval's own args hash. So `agentgate audit check-store` can find an approval row that someone changed later.

## Spend caps

Each policy sets three caps:

- USD per run
- USD per customer per day, where the day is the customer's local day in the customer's timezone
- tool calls plus model calls per run

The gate reserves spend before every tool call. The reference agent loop reserves before every model call. agentgate settles each reservation at the actual cost.

- A denied call gives its reservation back. It does not count toward the call cap.
- A call sent for approval gives its money back, but it counts as one call. So a run cannot queue more approvals than its call cap. Running the approved call later is another call.
- Every model call counts, also one that failed.

A reservation is a lease. Sometimes a host dies between reserve and settle, and the reservation stays open. agentgate releases a reservation that is still open 15 minutes after it was made. This happens on the next `open_world` of that database, or when you run `agentgate spend release-stale`. The approvals, users and gateway commands each call `open_world`. A released lease no longer counts toward the run or the customer day. It still counts toward the run's call cap. If the host was only slow and settles later, agentgate records the actual cost.

## Recipients

`src/agentgate/policy/addresses.py` holds the one address rule. The policy engine, the gate and the email tool all use it. A recipient is a valid plain address only if all of these are true:

- Every character is printable ASCII. The rule checks this before any case mapping. So it refuses control, bidi and zero-width characters, and look-alikes such as the Kelvin sign.
- It has exactly one `@`.
- The local part is runs of letters, digits and `_ + - '`, separated by dots. So there are no `%` relay or bang-path forms, no quotes and no backslash.
- The domain has two or more labels.

agentgate stores and sends the address in lowercase. So the address the policy checked is the address in the outbox. Any other address is malformed. The engine treats a malformed address as outside the company and on no list. The email tool refuses to store or send it, even if a policy or an approver allowed it.

## Audit

The default sink is [Panoptes](https://pypi.org/project/panoptes-observe/) (`panoptes-observe`). It is a hash-chained SQLite log. Each row's hash covers the row and the previous hash. agentgate records every decision before anything runs. Events carry ids, labels, counts, reason codes, rule ids and the policy version hash. They never carry document bodies, message text, notes or search queries. If agentgate cannot record a decision, the action does not happen. A few changes are recorded after they happen; see [Limits](limits.md#audit).

agentgate stores each event as a Panoptes schema-v5 event:

- The type is `agentgate.<kind>`, and the agent type is `agentgate`.
- The agentgate fields go in `attributes`. The run and the user go in Panoptes' own fields.
- For a model request through the proxy, `exchange_id` goes in `attributes`, and the exchange's `correlation_id` goes on the event.
- An event with no run has no run id, and its session is `customer:<id>`. An event with no user has no user.
- agentgate records no auth method, because it never sees a credential.

agentgate writes identifiers with strict ingest. If Panoptes would redact an identifier, it refuses the event. It never stores a changed value. Logs written with schema v4 (the same fields under `context.agentgate`) still read and verify, and new events extend their chain.

### Values derived from arguments

The log must be enough to rebuild an incident. So decision events also carry values derived from the arguments, but never the text:

- the args hash of a call whose arguments are valid: sha256 over the tool, the action and the validated arguments. Approvals are bound to the same hash.
- for a send, the number of recipients inside and outside the company
- for a spend, the amount
- for an executed call, the id of the row it wrote (`effect_ref`)

Approval events carry the approval's own args hash. A search carries no args hash, because its query is free text.

### Keyed references

Some caller values name nothing the store knows: an invented document slug, an unknown approver, a claimed run id. Others are values that Panoptes would refuse, such as an id shaped like an API key. agentgate records these as a keyed reference: `ref:` plus the first 16 hex characters of an HMAC-SHA256. The key is in the store. The same value always gives the same reference, so you can count repeated probes. The log never holds the value itself.

A call with a key-shaped id is denied and audited like any other call. If Panoptes still refuses an event because of its content, agentgate records a copy with every caller value as a reference.

### Policy snapshots and operator actions

Each time a world opens, agentgate stores the canonical bundle behind each customer's policy version. It goes in the store's `policy_snapshots` table. A `policy_loaded` event records the version and the bundle's sha256. So `agentgate audit policy` can print the rules behind any version an event names.

Operator actions are `operator_action` events. These are a fixture sync that changed something, a suspension, a reinstatement, and a stale spend sweep. On open, agentgate records a sweep only if it released something. `spend release-stale` always records its count.

### What an open records

What an open puts on the chain depends on who opens the world (`OpenRecord` in `src/agentgate/world.py`):

| Who opens | What goes on the chain |
| --- | --- |
| CLI commands that change state: `approvals approve`, `reject` and `expire`, `users suspend` and `reinstate`, `spend release-stale` | everything, including a `policy_loaded` event per customer |
| `approvals list`, `approvals show`, `users list` and `gateway serve` | only what the open changed: a new policy version, a fixture sync, a sweep |
| `audit show`, `audit policy` and `audit check-store` | nothing; they open the store read-only |
| `open_world` from a host | nothing, unless the host passes `record_open` |
| `open_memory_world` (used by `agent run`, the demo and the red-team suite) | nothing |

## Integration

You can run agentgate in two ways. It can be in-process middleware, a Python library that the host calls. Or it can be an MCP gateway in its own process.

### In-process

The host builds a world with `open_world`. A world is the store, fixtures, index, policies, audit sink and gate. The host starts a run for a user that it has already authenticated. Then it sends every tool call the model proposes through `call_tool`:

```python
from pathlib import Path

from agentgate.domain import Action
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

- `call_tool` returns `Executed`, `Denied`, `PendingApproval` or `CapReached`.
- An approved call runs through `gate.execute_approved(ctx, approval_id, tool, action, args)`.
- Model spend goes through `gate.charge_model` and `gate.settle_model`.
- `src/agentgate/agent/loop.py` is a reference loop that does all of this.
- `open_memory_world` builds the same world in memory, with an in-memory audit sink.

When a world opens on disk, agentgate does these things:

- It syncs the fixtures into the store. It adds new customers, users and documents, and it updates changed ones. It deletes nothing. It suspends a user who is no longer in the fixtures. So a suspension, a role change or a relabeling in the fixtures takes effect on the next open.
- It releases stale spend reservations.
- It refuses a store path that is the audit log, or that is the same file as the audit log.

`open_world` records nothing about the open on the audit chain unless you pass `record_open` (see [Audit](#audit)).

### The MCP gateway

In-process middleware protects only the tools that the gate alone can reach. The gateway moves the tools, the database and their side effects into a separate process. The agent gets only an MCP connection. The code is in `src/agentgate/gateway/server.py`.

`agentgate gateway serve --as USER` serves MCP over stdio. It opens the same on-disk world as the CLI, which is `data/agentgate.db` and `data/audit.db` by default. Before it serves anything, it starts one run for that user. Every call on the connection belongs to that run.

The identity is bound at launch:

- The MCP client never names a user, role or customer.
- No tool takes an identity argument. The gate denies an extra key such as `user_id` or `role` with `invalid_args`.
- `--as` is the same demo identity as on the CLI. It names a fixture user. It is not a login, and the gateway prints the same note on stderr.
- For an unknown or suspended user, the gateway exits with code 1 before it writes anything to stdout.
- One gateway process serves one connection and one run.

The gateway lists the customer's tools under MCP-safe names, so `docs.search` becomes `docs_search`. It also lists `approvals_execute`. For a Kettle Creek user such as `kc-staff-nadine`, the list is `docs_read`, `docs_search`, `email_send`, `notes_write`, `parts_order` and `approvals_execute`. The action of each call comes from that tool's entry in the customer's tool list, never from the client. If a name is not on the list, the gateway sends it to the gate unchanged, and the gate denies it as an unknown tool.

`approvals_execute` runs a call that a person has already approved, with exactly the approved tool and arguments. It goes through `gate.execute_approved`, which checks the run, the arguments hash, the expiry and the current policy again. `approvals_execute` cannot approve anything. Approvals are decided separately with `agentgate approvals approve`. An approval belongs to the run that requested it, so you execute it on the same gateway connection.

The gateway refuses nothing itself. Every call and every refusal goes through the gate, which records it with the bound run's identity. Sometimes the gateway cannot read the fields of an `approvals_execute` request: a field is missing, a type is wrong, or there is an extra key. Then the request goes to `gate.refuse`. This checks the run and user like any call and records an `invalid_args` denial.

The gateway decides calls one at a time, in order. The client gets the same outcome that the reference loop shows its model, as JSON text. This is the tool's content, `{"error": reason}`, a pending approval's id and expiry, or the cap that was reached. Two kinds of request end with no gate record; see [Limits](limits.md#mcp-gateway).

The tools, the database and every side effect are in the gateway process. The credentials of real tools would be there too. So an agent that holds only the MCP connection can reach a tool only through the gate.

This is an example MCP client configuration. It uses the `mcpServers` format that many clients use for stdio servers. Replace the path with your checkout:

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

`uv run --directory` runs the command from the checkout, so the default `data/`, `policies/` and `fixtures/` paths resolve there. `tests/test_gateway.py::test_cli_serves_over_stdio` starts `agentgate gateway serve --as kc-staff-nadine` as a subprocess on every test run. It checks that the gateway answers `initialize`, `tools/list` and a `docs_read` over stdio.

## Model gateway

agentgate also decides what a run may ask of a model. The parts are in `src/agentgate/model/` and in `Gate.admit_model`. Two proxies put these parts in front of a provider: `model-proxy serve` for one run, and `model-proxy server` for many runs. The sections after this one describe the proxies.

- **Keys** (`model/keys.py`). agentgate loads the provider keys one time, from an owner-only JSON file (`{"openai_compatible": "..."}`) or from `AGENTGATE_<PROVIDER>_API_KEY` variables. Only the agentgate process holds them. An error names the provider and the path, never a value, and agentgate never chains a parser error to it. The key object prints without its values, and you cannot pickle it. `redact` removes a key from outgoing text. Before agentgate adds the provider's key, it removes every client credential header: `Authorization`, `x-api-key`, `x-goog-api-key`, `api-key`, `Cookie` and `Proxy-Authorization`. Panoptes removes credential query parameters (`key`, `api_key`, `apikey`, `access_token`) when the proxy hook calls `exchange.drop_query_credentials()`.
- **Tokens** (`model/tokens.py`). An agent holds a token, never a key. A token is bound to one agentgate run. agentgate stores only its hash. agentgate refuses a token in these cases: it is unknown, revoked or expired; it belongs to a run other than the one served; or its run's user is unknown, of another customer or suspended. A token dies with its run, or at the end of its lifetime. The default lifetime is 8 hours, and the maximum is 30 days. `end_run` revokes every token of a run. It records each revocation with reason `run_ended` and no actor, in the same transaction. A run holds at most 16 live tokens. agentgate refuses a request with two different credentials as ambiguous.
- **The decision** (`Gate.admit_model`). The steps are, in order:
  1. Check the run and its user, as for a tool call.
  2. Read the body strictly (`model/admission.py`): UTF-8, one JSON object, no repeated key, whole-number limits. The largest declared output limit counts. A body that cannot be read this way is denied with `model_request_invalid`. A body with a top-level field that is not on the provider's list is denied with `model_unknown_field`.
  3. Decide with the policy's `models:` entry.
  4. Price the reservation.
  5. Check the caps.
  6. Record the allow, then return the reservation. If agentgate cannot record the allow, it releases the reservation and denies the call with `audit_unavailable`.

  A model name that the policy does not list is recorded only as a keyed reference. A correlation id from the client is recorded as `asserted_correlation_id` only if it has a plain identifier shape. Otherwise it is recorded as a reference. It never decides who the run is.
- **The reservation** (`model/pricing.py`). The worst case of a call is its declared `max_tokens` at the output rate, plus the prompt. The prompt counts as one token per byte of body, at the higher of the input and cached-input rates. agentgate rounds the result up to a micro-dollar. Rates come from a versioned JSON table. Panoptes uses the same format to price observed responses: `{"version", "source", "currency": "USD", "models": {name: {input_per_million, cached_input_per_million, output_per_million}}}`. A model that the table does not name has no price. agentgate denies its calls with `model_unpriced`, unless the policy sets a fixed `reserve`. A fixed `reserve` replaces a missing price, and it is a floor under a price the table gives. The reservation goes through the same ledger as every tool call. So tool spend and model spend share the run and customer-day caps, and a model call counts toward `max_calls_per_run`.
- **Caps** (`Caps`). A process that serves one run, or a run started on the multi-run server, can lower the run's USD cap and call cap. It can never raise them. The policy's value is the ceiling.
- **After the call** (`Gate.finish_model`). agentgate releases the reservation only in two cases: the request never reached the provider, or the provider answered 4xx or 5xx with no output. Every other ending is charged. If the reply is complete and priced, agentgate charges the priced cost. Otherwise it charges the full reservation. This includes a client that drops the connection and a stream that breaks, because the provider may have billed them. If a killed process left a reservation open, the stale sweep releases it the next time a world opens.

### The model proxy

```bash
agentgate model-proxy serve --as USER --upstream openai_compatible=https://api.deepseek.com --rate-table rates.json
```

This command serves the model APIs for one run. It is the Panoptes reverse proxy, with agentgate's decision in its two hooks (`panoptes.hooks`; see `docs/hooks.md` in Panoptes). Panoptes forwards, parses, prices and records what it observed. agentgate decides and never touches the body. Both write to one audit file, through one `AuditDB`, on one hash chain.

`before_forward` does these steps in order. Any step can end the request with a status and a reason code. The reason code never shows a policy detail or a key.

1. It accepts only a `POST` to a generation endpoint: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `.../models/<model>:generateContent` or `:streamGenerateContent`. It refuses every other route or method with 403 `route_not_allowed`. This includes `/v1/models`, `count_tokens` and `/v1/responses/<id>`. It reads a Gemini model from the path as received, and never percent-decodes it.
2. Panoptes names the wire protocol. The operator's `--upstream` names the policy provider for that protocol. `openai_compatible` is an upstream that uses the OpenAI wire protocol. The proxy refuses a protocol it does not serve.
3. It reads the raw request body with agentgate's strict parser (`parse_model_call`), not with Panoptes' `request_json`.
4. It reads the token from `Authorization`, `x-api-key`, `x-goog-api-key`, `api-key` or a `?key=` query value. It refuses two different credentials as ambiguous. It finds the run that the token belongs to. A bad token gets 401 `token_refused`. If a bad token names a run, the refusal and its reason go on that run's record.
5. It builds and validates the identity. The `session_id` and `run_id` are the run id, and the user is the run's user. It also does every other step that can raise before a reservation exists.
6. `Gate.admit_model` decides (policy, worst-case reservation, caps) and records. A denial is a 403 with its reason code. A reached cap is a 429 with the cap. A failure to decide is a 503.
7. It drops the client's query credentials and removes every client credential header. agentgate's list is wider than Panoptes' list: it adds `api-key`, `cookie` and `proxy-authorization`. The provider's key is the only credential sent.

`after_exchange` runs one time for every exchange that entered `before_forward`. It ends the reservation through `Gate.finish_model`:

- It releases the reservation if the request never reached the provider, or if the provider answered 4xx or 5xx with no output.
- It settles at the priced cost if the reply is complete and priced.
- Otherwise it settles at the full reservation.

The cost is Panoptes' `estimated_usd_micros`, as `Decimal(micros).scaleb(-6)`. A reply that reports no usage is unpriced (`None`), so it is charged the full reservation. The decision events of an exchange carry its `exchange_id`. This is the key to join them with Panoptes' events for the same exchange.

The process lifecycle is:

- At startup, the process starts the run and issues its one token. It prints the token one time, or writes it with mode 0600 to `--token-file`.
- Every minute, it sweeps reservations that are older than the lease.
- At shutdown, it stops accepting requests and lets in-flight exchanges finish and settle. Then it ends the run and revokes its token.

At startup, the process refuses to serve unless all of these are true:

- The policy lists a model for each upstream.
- Each upstream has a provider key.
- The reservation's rate table (`RateTable`) and the settlement's table (Panoptes' `PricingTable`) read one file and get the same version, source and rates for every policy model.

`agentgate doctor model-proxy` runs these checks without serving.

#### The ready file

`--ready-file PATH` tells a harness that the proxy is up. The harness does not need to parse stderr, but the `model proxy: run <id> for <user>, http://...` line is still printed. After the port is listening and the token file is written, the process creates PATH as a new 0600 file. If PATH exists, the process refuses to start. It publishes the file atomically, so a reader that sees the file sees all of it:

```json
{
  "url": "http://127.0.0.1:53017",
  "host": "127.0.0.1",
  "port": 53017,
  "pid": 41230,
  "run_id": "6f1c2a8e-...",
  "user": "lab-runner",
  "customer": "lab",
  "max_usd": "0.25",
  "max_calls": 40,
  "per_customer_day_usd": "5",
  "token_file": "/path/to/run.token"
}
```

- `max_usd` and `max_calls` are the caps that the proxy enforces. They are the policy's `per_run_usd` and `max_calls_per_run`, lowered by `--max-usd` and `--max-calls`, and never raised. Each admission computes them the same way.
- Amounts are decimal strings, as in the policy.
- At shutdown, the process removes the file and the token file. It installs the SIGTERM handler before it publishes the file. Removal runs even if shutdown fails.
- The process removes the file only if it still names this process's `pid` and `run_id`. So it leaves alone a replacement that another process wrote.
- A process that is killed with SIGKILL, or whose machine crashes, cannot remove its file. So treat the file as a hint. Before you rely on it, check that `pid` is alive and that `url` answers.

#### Inside the MCP gateway

```bash
agentgate gateway serve --model-proxy --upstream ... --rate-table ... --token-file ...
```

This starts the same proxy inside the MCP gateway's run. There is one run id, so model calls and tool calls reserve against the same per-run USD cap, per-day cap and call count. The token goes only to `--token-file`, because stdout is the MCP stream. `--max-usd` and `--max-calls` tighten the run's caps for model calls. They can never raise the policy's caps.

### The multi-run model proxy

```bash
agentgate model-proxy server --upstream openai_compatible=https://api.deepseek.com \
  --rate-table rates.json --keys-file keys.json \
  [--control-socket PATH] [--max-token-ttl 3600] [--ready-file F]
```

This is one long-running process that serves model calls for many runs of one world. The model listener is the same Panoptes proxy on a loopback port, with the same `before_forward` and `after_exchange` steps as `serve`.

The difference is where a token's run comes from (`RunSource` in `model_proxy/proxy.py`):

- `serve` serves the one run it started. It refuses a token of any other run as `wrong_run`.
- The server (`ServedRuns` in `model_proxy/server.py`) accepts a token for the run that the token names, if this server instance started that run. It refuses any other run as `not_served`. This includes a run of a `serve` process on the same world and a run of another server.
- Like `serve`, the server refuses a run that has ended, or whose user is unknown, suspended or of another customer.

A world can hold several customers, and the server serves all of them. A host with several worlds runs one server for each world.

#### The control socket

The control listener is a Unix domain socket. The server creates a new directory for it with mode 0700, and creates the socket with mode 0600. So only processes of the user who runs the server can reach it. Filesystem permission is the authentication. This is the same trust as `--as`.

- By default, the socket is in a new private directory under the system temporary directory.
- With `--control-socket PATH`, the directory of PATH must not exist. The server refuses an existing directory, symlink or socket. It does not reuse one.
- There is no TCP control port.

Requests are small JSON over HTTP/1.1, one per connection. agentgate's own handler handles them, at most 16 at a time. Above that, the server answers 503 `busy`.

| Request | Answer |
| --- | --- |
| `POST /runs` `{"customer_id", "user_id", "ttl_seconds"?, "max_usd"?, "max_calls"?}` | `201 {"run_id", "customer_id", "user_id", "token", "expires_at", "max_usd", "max_calls", "per_customer_day_usd"}` |
| `POST /runs/{run_id}/end` | `200 {"run_id", "status": "ended"}`, also for a run that already ended; 404 `unknown_run` for a run that this server did not start |
| `GET /health` | `200 {"status": "ok", "model_url", "runs_live"}` |

How `POST /runs` works:

- The server starts the run through `Gate.start_run`. So it refuses an unknown user, a suspended user, or a user of a customer other than `customer_id`. The refusal is a 403 with the gate's reason, and the gate records it.
- The server issues the run's one token and records `token_issued`. The token appears only in the answer.
- `max_usd` (a decimal string) and `max_calls` can only tighten the policy's caps. The answer gives the caps in effect.
- `ttl_seconds` is cut to `--max-token-ttl`. That option is one hour by default, and it is also the default lifetime.
- The server stores the caps in `run_limits`, and which instance and process started the run in `served_runs`. Both writes are in the transaction that starts the run. Each admission reads the caps of the run it is serving.

A malformed request gets a 400 with a reason code. Malformed means it is not one JSON object, or it has a repeated or unknown field, or a wrong type. Another route gets 404, and another method gets 405. The server never sends a traceback.

`agentgate.model_proxy.client.ControlClient(socket_path)` wraps the three requests over httpx's Unix-socket transport. It refuses a socket path that is a symlink or not a socket. It also refuses a socket in a directory that is a symlink, that another user owns, or that group or others can open.

#### Lifecycle

- **Ready.** When both listeners are bound, the server writes `--ready-file`. This is a new 0600 JSON file with `model_url`, `control_socket`, `pid` and `instance_id`. The server publishes it the same way as `serve`.
- **Ending a run.** A request to end a run records `run_ended` with reason `requested`.
- **Shutdown.** On SIGTERM or an interrupt, the server stops accepting on both listeners. It lets in-flight control requests and exchanges finish, and it settles each exchange. It ends every run it started, with `run_ended` reason `shutdown` and a `token_revoked` for each token. Then it removes the socket, its directory and the ready file.
- **Recovery.** A server that dies without a clean stop leaves its runs' tokens live. The next server that starts on the world ends every live run whose server process is gone, with `run_ended` reason `recovered`. Its stale-reservation sweep runs the same as in `serve`.
- **Startup checks.** The server checks the upstreams and keys. It checks that some customer's policy lists a model for the served providers. It also checks the rate table both ways, as `serve` does.

## Policy format

Each customer has a folder under `policies/` with up to three files:

- `policy.yaml`: rules, caps, approvals, lists and models
- `tools.yaml`: the tools the customer has
- `tests.yaml`: expected decisions

`rules`, `approvals`, `lists` and `models` are each optional, with these limits:

- A policy must have rules or a `models:` section. agentgate rejects a policy with neither, because it allows nothing.
- `approvals` is required when any rule is `allow` or `require_approval`, because an allowed call can still escalate to approval.
- `tools.yaml` is required when there are rules.

See [Models-only policies](#models-only-policies). This is a shortened part of `policies/kettle-creek-hvac/policy.yaml`:

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

- A matching `deny` wins over `require_approval`, and `require_approval` wins over `allow`. If no rule matches, the request is denied (`default: deny`).
- `allow` and `require_approval` rules must list `roles`. Only a `deny` rule can leave out `roles`, and then it applies to every role. So a missing line can never widen access.
- If a document has a source ACL and the user holds none of its principals, the request is denied (`acl_denied`) before any rule (see [Source ACLs](#source-acls)). No rule can grant past the ACL.
- A rule matches on `roles`, `tools`, `actions` and `labels: {max: ...}`, plus optional `where` conditions. `labels: {max: ...}` matches when the document's label is at or below the bound.
- A condition whose fact is missing never holds. A condition on an action that never carries its fact, such as a recipient on a read, is a validation error.
- There is no expression language. The conditions are a closed set:

| Condition | Holds when |
| --- | --- |
| `recipient_in: <list>` | every recipient is on the named list; on a `deny` rule, any recipient is (send) |
| `recipient_external: true` | at least one recipient is outside the customer's domains (send) |
| `amount_usd_gt: "<usd>"` | the amount is above the value (spend) |
| `amount_usd_lte: "<usd>"` | the amount is at or below the value (spend) |
| `run_max_label_gte: <label>` | the run has already been shown a document at or above this label (any action) |
| `doc_tags_any: [<tag>]` | the document has any of the tags (read, write) |
| `doc_assigned_to_user: true` | the document is assigned to the acting user (read, write) |

### Models

`policy.yaml` can also list the models that a run may call through the model gateway. Like the rules, the list denies by default. A model that is not listed is denied (`model_not_allowed`). Under a policy with no `models:` section, every model request is denied. From `policies/kettle-creek-hvac/policy.yaml`:

```yaml
models:
  allow:
    - id: deepseek-flash
      provider: openai_compatible
      model: deepseek-flash
      roles: [owner, manager, staff]
      max_output_tokens: 8000
      max_request_bytes: 120000
  reserve: computed
```

- `provider` is the wire protocol that the request must use: `anthropic`, `openai`, `gemini` or `openai_compatible`.
- `model` must equal the request body's `model` field exactly. It is never a prefix or a pattern.
- Two entries cannot name the same provider and model.
- Every field of an entry is required. `roles` is never optional, so a missing line cannot widen access.

A request is allowed only if it fits its entry. Each way it can miss has its own reason code, and the caller sees only that code:

| Reason | The request |
| --- | --- |
| `model_not_allowed` | names a model or provider that no entry lists |
| `model_role_denied` | comes from a role that the entry does not list |
| `model_max_tokens_required` | declares no output limit, or a limit below 1 |
| `model_max_tokens_exceeded` | declares a limit above `max_output_tokens` |
| `model_n_not_one` | asks for a number of completions other than one |
| `model_request_too_large` | has a body above `max_request_bytes` |
| `model_server_feature` | starts provider-side state, a provider-run tool or a prompt-cache write, and the entry does not set `allow_server_features: true` |

- A request with no `max_tokens` is denied, even when `reserve` is a fixed amount. The output limit is part of the envelope. Without it, agentgate cannot check the entry's cap.
- `reserve` is what one model call holds against the spend caps before agentgate sends it. `computed` prices the worst case from the request and the rate table. A fixed USD string such as `"0.02"` is the reservation for a model that the rate table does not price. For a model that the table prices, the fixed value is a floor under the computed amount, and agentgate reserves the larger one. So a very small fixed value cannot lower the worst case.
- A `tests.yaml` case for a model call gives `model: {provider, name, max_tokens, n, request_bytes}` in place of `tool` and `action`. It can state the decision's `reason` and the `rules` (the entry id).

### Models-only policies

Some experiments only need this: "this user may call this model within these caps." For these, a policy can leave out `tools.yaml`, `rules` and `approvals`:

```yaml
apiVersion: agentgate/v1
customer: lab
labels: [public]
default: deny
spend:
  per_run_usd: "0.50"
  per_customer_day_usd: "5"
  max_calls_per_run: 40
models:
  allow:
    - id: deepseek-flash
      provider: openai_compatible
      model: deepseek-flash
      roles: [owner]
      max_output_tokens: 8000
      max_request_bytes: 4000000
  reserve: computed
```

The tool gate still denies by default. With no `tools.yaml`, every tool call is `unknown_tool`. With a `tools.yaml` but no rules, every call is `default_deny`. A customer with no `approvals` has no approvers, so `approvals approve` and `reject` refuse its users (`approver_not_allowed`).

`agentgate init-experiment DIR --model PROVIDER:NAME [--model ...] --max-usd USD --max-calls N` writes exactly this. It writes `DIR/fixtures/lab/` with one active owner, `lab-runner`, and it writes `DIR/policies/lab/policy.yaml`. The options are:

| Option | Sets | Default |
| --- | --- | --- |
| `--day-usd` | the day cap | `--max-usd` |
| `--max-output-tokens` | `max_output_tokens` of every entry | 8000 |
| `--max-request-bytes` | `max_request_bytes` of every entry | 4000000 |
| `--customer`, `--user` | the ids | `lab`, `lab-runner` |
| `--allow-server-features` | `allow_server_features: true` on every entry | off |

Use `--allow-server-features` for agents whose requests use provider-side features, such as prompt-cache writes or tool types that are not functions. It also lets provider-run tools and provider-side state (background runs, stored conversations) through. Work that the provider does on its side can cost more than the reservation computed from the request (see [Limits](limits.md#model-gateway)).

`init-experiment` validates everything before it writes anything. It creates each file exclusively and never through a symlink. So it never overwrites an existing file, even one that appears while the command runs. If that happens, it removes the files that this call already wrote. It does not remove a file that another process replaced in the meantime; it checks this by inode.

Serve the experiment with `model-proxy serve --as lab-runner --policies DIR/policies --fixtures DIR/fixtures ...`.

### tools.yaml

Each shipped tool has these fields:

- exactly one action
- a base risk for the risk score
- an optional `customer_facing` flag
- a platform cost for each call, which counts against the spend caps

From `policies/kettle-creek-hvac/tools.yaml`:

```yaml
- id: email.send
  description: Send an email from the company mailbox.
  actions: [send]
  customer_facing: true
  base_risk: 40
  cost_usd: "0.001"
```

The risk score is the tool's `base_risk`, plus 25 if any recipient is outside the company, plus 5 for each full 100 USD of the amount, up to 40.

An allowed call still needs approval in these cases:

- A `customer_facing` tool reaches someone outside the company.
- The risk score reaches `risk_threshold`.
- The run has already been shown a document, and an internal recipient is not an active user of the customer whose role may read a document at that label (`escalated_recipient_clearance`). An internal address that names no user counts as not cleared.

### tests.yaml

Each case names a fixture user, a request, the expected effect and the ids of the deciding rules. The role and customer come from the fixtures, never from the case. One case from `policies/kettle-creek-hvac/tests.yaml`:

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

A policy's version is a content hash of the validated bundle (the policy and its tools). Comments and key order do not change it. Any change in meaning does. Every audit event records the version, and the store keeps the bundle behind it (see [Audit](#audit)).

```console
$ uv run agentgate policy lint
juniper-row-apothecary  sha256:091571d97a0826e8  15 rules  5 tools  git f5b8eba
kettle-creek-hvac  sha256:0505f513f2ee6853  14 rules  5 tools  1 models  git 329c4d3

$ uv run agentgate policy test
juniper-row-apothecary  sha256:091571d97a0826e8  60/60 passed
kettle-creek-hvac  sha256:0505f513f2ee6853  78/78 passed
total  138/138 passed
```

Both outputs are from commit `99accb1`. The version covers the `models:` section too. A policy with no `models:` section keeps the version it had before that section existed.

## Approvals, users, tokens and audit from the CLI

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
uv run agentgate token issue --as kc-staff-nadine --run <RUN_ID> [--ttl-minutes 480] [--token-file run.token]
uv run agentgate token list --as kc-staff-nadine [--run <RUN_ID>]
uv run agentgate token revoke <TOKEN_ID> --as kc-staff-nadine
uv run agentgate audit show --as kc-mgr-terrence [--run <RUN_ID>] [--kind deny] [--limit 20] [--json]
uv run agentgate audit policy sha256:66ca6bb217693239 --as kc-mgr-terrence
uv run agentgate audit check-store --as kc-owner-dale
uv run agentgate audit verify [--db data/audit.db]
```

### Common behavior

- `--as` is a demo identity. It names a fixture user. It is not a login. Every command that takes `--as` prints `note: --as is a demo identity; agentgate does not authenticate users`.
- By default, the commands use `data/agentgate.db` and `data/audit.db`. If `data/agentgate.db` is missing, agentgate creates it and seeds it from the fixtures. It syncs from the fixtures on every open.
- The approvals, users, spend and gateway commands open the world. On stderr, they say what the open changed: a fixture sync, released stale reservations, or a store file that other users could read. agentgate tightens such a file to 0600 and prints a warning.
- The commands refuse a `--db` that is a Panoptes audit log or the same file as `--audit-db`. They also refuse an `--audit-db` that holds an agentgate store.
- A database error is one line, `error: database error: ...`, with exit code 1.

### Approvals

- `approvals list` and `show` show only the acting approver's customer. `show` prints the exact arguments that would run.
- Neither command changes an approval. An approval past its expiry shows as `expired (pending sweep)` until `approvals expire` marks it. It counts as expired for `--status`.
- A staff user who tries to approve gets `approver_not_allowed: role 'staff' may not decide approvals`.
- `expire` expires every overdue approval for all customers and prints only the count. The acting user must be an active owner.

No approvals command creates an approval. A pending approval comes from an agent run against an on-disk world, in one of two ways:

- The host calls `call_tool` as in the integration example, and a call that needs approval returns `PendingApproval`.
- An MCP client calls a tool through `agentgate gateway serve`, which uses the same files by default.

`agentgate agent run` uses a new in-memory world and discards it when it exits. So you cannot decide its approvals afterward. After approval, the requester's run executes the call through `gate.execute_approved`, or over MCP through `approvals_execute` on the same gateway connection.

### Users and spend

- `users list` prints the users of the acting user's customer, with role and status. A user suspended with `users suspend` shows as `suspended (by operator)`.
- `users suspend` and `users reinstate` are for owners. An owner cannot suspend themselves.
- A suspension takes effect on the user's next gate call, in any process that uses the same database. This includes a live run or gateway.
- A suspension is sticky. A fixture sync does not reactivate the user. After `users reinstate`, the fixtures decide the user's status again.
- Both commands record the acting owner, the user and the `--reason` code.
- `spend release-stale` releases reservations that have been open longer than `--older-than` minutes (default 15), for all customers. It prints only the count. The acting user must be an active owner.

### Tokens

- `token issue` makes a model-gateway token for a run of the acting user. An owner can issue a token for any run of their customer. The command prints the token one time on stdout. Or it writes the token to `--token-file`, a new file with mode 0600 that it never overwrites.
- agentgate stores only the token's sha256.
- `token list` shows ids, runs, status and times. It never shows a token.
- `token revoke` revokes one token by id.
- A run or token of another customer reads like a missing one.
- agentgate records `issue` and `revoke` in the audit log as operator actions, in the transaction that makes them. So a token whose issue cannot be recorded does not exist.

### Audit

- `audit show` prints the events of the acting user's customer, oldest first, one per line. It is for owners and managers. A value that the log holds as a reference prints as its `ref:` and the reason. `unknown value` means it named nothing that the store knew.
- `audit policy` prints the policy bundle stored for a version. It checks the bundle's sha256 against the snapshot, the version and every `policy_loaded` event for that version. It is for owners and managers.
- `audit check-store` is for owners. It verifies the chain. Then it checks each of the customer's approval, outbox, parts order, refund and note rows against the hashes in their events.
- `audit policy` and `audit check-store` exit with code 1 on any mismatch.
- The three commands above open the store read-only. They change neither file.
- `agentgate audit verify` checks the Panoptes hash chain. It prints `<path>  N events  chain intact`, or it prints `chain BROKEN: ...` and exits with code 1. An agentgate log is chained from its first event, so events outside the chain also fail. An example is a log whose hashes were set to null and whose chain metadata was deleted. The command opens the file read-only through Panoptes' read API. So it never migrates the file, starts a chain or writes to it. It has no checkpoint option; see [Limits](limits.md#audit).

## Latency

These numbers are from `uv run agentgate bench` at commit `05456e9`: 2000 timed iterations after 200 warmup iterations, with times in microseconds. The machine was an Apple M4 Pro with 12 CPUs, macOS 15.7.3 and Python 3.12.12, with desktop apps running (load average about 4). The store is in-memory SQLite.

| Measurement | p50 | p95 |
| --- | ---: | ---: |
| `engine.decide`, allowed read (decision only) | 2.5 | 2.6 |
| Gated `docs.read`, added over a direct call, MemorySink | +161.3 | +178.7 |
| Gated `docs.read`, added over a direct call, PanoptesSink | +3209.5 | +4659.0 |
| Gated `docs.search`, added over a direct index search | +177.8 | +216.2 |
| Denied call through the gate | 62.9 | 73.9 |
| Gated `docs.read` on a busy day (1000 earlier reservations) | 179.4 | 207.1 |

"Added" is the gated median minus the direct median, and the same for p95. The durable Panoptes audit log is most of the cost: an allowed read writes two hash-chained events to a SQLite file. These numbers depend on the machine. Compare them only with runs on the same machine.

## Optional real model

The demo, the red-team suite and CI use mock models. To run the reference agent with Claude:

```bash
uv sync --extra anthropic
export AGENTGATE_MODEL=anthropic
export ANTHROPIC_API_KEY=...
uv run agentgate agent run --as kc-staff-nadine --task "Please email beatrice.l@mail-sample.example about her tune-up." --model anthropic
```

`.env.example` lists the variables. agentgate does not read `.env` files, so export the variables in your shell. The default model is `claude-opus-5-5`. Set `AGENTGATE_ANTHROPIC_MODEL` to change it. `agent run` prints every gate decision for one task in an in-memory world. Without `--model anthropic`, it uses the offline mock model.

There are two caveats:

- The spend caps in the shipped policies are sized for the mock model's synthetic prices, not for real model prices.
- The adapter is tested only against a fake client built from the SDK's types. No test, CI job or report in this repository has run it against the live API.

## Red-team suite over MCP

The README table gives the results for both transports. In the MCP run, the suite sends every agent-side tool call through an MCP client session to a gateway bound to the run.

These counts are from `uv run agentgate redteam --transport mcp` at commit `99accb1`: 177 scenarios, 1989 checks, all passed, and 0 baseline problems. The suite made 1359 calls over MCP and 58 calls directly to the gate.

A call goes directly to the gate only when MCP cannot express it:

- 29 calls use a forged run context. An MCP client cannot pass a context at all.
- 23 calls use an action that the tool does not have. Over MCP, the action comes from the gateway's own tool list.
- 6 calls belong to a run whose customer has no policy. The gateway does not start for such a run.

42 of the 156 attack scenarios make no MCP call:

- The 29 model_boundary scenarios call the gate, the token store and the key helpers directly, because MCP does not carry the model gateway.
- id-005, id-006, id-007, fc-002, fc-003 and pe-014 send only direct calls of the kinds above.
- id-001, xt-013, sr-003, sr-006 and sr-007 test only run starts and model spend. These stay in-process, because a gateway's identity is bound at launch and the gateway never sees model calls.
- fc-016 and fc-017 make no tool call. They test how the store file is opened.

Every benign_control scenario makes at least one MCP call. The suite's sessions use the MCP SDK's in-memory transport, with one gateway per run. A test covers the stdio launch (see [The MCP gateway](#the-mcp-gateway)).

## Architecture

The code has six layers. A module imports only from layers below its own. [import-linter](https://import-linter.readthedocs.io/) checks this in `make lint`, so CI fails on an import that goes the wrong way. The contracts are in `pyproject.toml`.

| Layer | Packages | Role |
| --- | --- | --- |
| Entry points | `cli`, then `demo`, `redteam`, `experiment` | Commands and reports. Nothing imports them. |
| Adapters | `gateway`, `model_proxy`, `agent` | Put the gate behind MCP or HTTP, or drive it with a model. |
| Composition | `world` | Opens a store, fixtures, policies, an audit sink and a gate in one call. |
| Gate | `gate` | Decides, records and runs every call. |
| Services | `approvals`, `spend`, `tools`, `audit`, `model`, `retrieval`, `policy` | One job each. None of them knows about the gate. |
| Foundation | `fixtures`, `store`, `context`, `clock`, `domain` | Data and the types everything shares. |

The contract orders the packages inside a row too. For example, `tools` may import `retrieval`, and `retrieval` may import `policy`, but not the other way around.

`model` holds what the gate needs to decide a model call: strict request reading, prices, run tokens and provider keys. `model_proxy` is the HTTP server in front of a provider. It uses the gate, so it sits two layers above `model`.

### Inside the gate

`Gate` is the whole interface of the `gate` package. A second import-linter contract stops any other package from importing the modules behind it. `Gate` builds these parts and passes each one the others it needs:

| Module | Part | Job |
| --- | --- | --- |
| `recorder.py` | `AuditRecorder` | Writes every event. Records caller values as keyed references. |
| `runs.py` | `RunRegistry` | Starts runs. Checks every call's run against the `runs` table. Tracks the max label. Reserves spend. |
| `tool_calls.py` | `ToolCalls` | The tool-call steps, from argument validation to running the tool. |
| `model_calls.py` | `ModelAdmission` | Admits model requests, and settles what they cost. |
| `decisions.py` | `ApprovalDecisions` | Approve, reject and expire, each recorded in its own transaction. |
| `operator_log.py` | `OperatorLog` | Policy loads, operator actions and the stale spend sweep. |
| `outcomes.py` | | `Executed`, `Denied`, `PendingApproval`, `CapReached` and the gate's reason codes. |

The tool path and the model path share the run check, the recorder and the spend reservation. So both paths fail closed in the same way.

## Project layout

| Path | Contents |
| --- | --- |
| `src/agentgate/gate/` | the gate: check, decide, record, then run every tool and model call (see [Inside the gate](#inside-the-gate)) |
| `src/agentgate/domain.py` | domain types: customers, users, roles, tools, documents |
| `src/agentgate/context.py` | `RunContext`, the identity of one run, created only by the gate |
| `src/agentgate/policy/` | policy schema, engine (deny by default), `tests.yaml` runner, version hash |
| `src/agentgate/policy/addresses.py` | the one recipient address rule, used by the engine, the gate and the email tool |
| `src/agentgate/retrieval.py` | permission-aware search: filter, BM25F over the permitted set, check again |
| `src/agentgate/approvals.py` | approval queue bound to tool, action and args hash |
| `src/agentgate/spend.py` | spend ledger: calls and USD per run, USD per customer day |
| `src/agentgate/model/` | model calls: provider keys, run tokens, strict request reading and caps, rate tables and the worst-case reservation |
| `src/agentgate/model_proxy/` | the model proxy: one run (`proxy.py`), many runs and their control socket (`server.py`, `client.py`) |
| `src/agentgate/audit/` | audit event type and keyed references, memory sink, Panoptes sink |
| `src/agentgate/tools/` | tool adapter interface and the built-in tools |
| `src/agentgate/agent/` | model protocol, reference agent loop, mock models, optional Anthropic adapter |
| `src/agentgate/store.py`, `world.py` | SQLite store; `open_world` and `open_memory_world`, user suspension, `check_store` |
| `src/agentgate/fixtures.py` | fixture loading, and the sync on every open |
| `src/agentgate/gateway/` | MCP gateway: one run per process, every tool call through the gate |
| `src/agentgate/redteam/` | scenario registry, leak detector, file worlds, baseline check, reports, benchmark, MCP transport |
| `src/agentgate/experiment.py` | `init-experiment`: a customer with one user and a models-only policy |
| `src/agentgate/demo.py` | `make demo` |
| `src/agentgate/cli/` | the `agentgate` CLI, one module per command group |
| `policies/` | one folder per customer: `policy.yaml`, `tools.yaml`, `tests.yaml` (`tools.yaml` is optional for a models-only policy) |
| `fixtures/` | the synthetic customers, users and documents |
| `redteam/baseline.json` | the scenarios that must exist and pass |
| `tests/` | pytest suite, including mutation tests for the red-team scenarios |
