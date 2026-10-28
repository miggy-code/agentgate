# Limits

What agentgate does not do, and where its guarantees stop. The README lists the most important of these.

## Identity and deployment

- `--as` on the CLI, `agentgate gateway serve` included, names a user; it is not a login. agentgate trusts the host application, or whoever launches the gateway, to authenticate users before a run starts.
- The run id is the run's only credential. `RunContext` is a plain dataclass, not signed. Altered fields, unknown run ids and look-alike objects are refused, but anything in the host process that holds a live run id and its fields can act as that run. Runs never expire; there is no call that ends one, and a run is bounded only by its spend caps.
- In-process, agentgate only protects tools that are reachable solely through the gate: code in the same process that holds a tool's credentials, or the database, can skip it. The MCP gateway closes that gap for an agent that holds only the MCP connection, since the tools and the database live in the gateway process. Code that can open the database or the tools directly can still skip the gate.
- State lives in one SQLite file per world, plus a separate Panoptes SQLite file for the audit log. Several processes on the same machine can share them (the approvals CLI does). There is no server database and no multi-host support.
- The store file is created 0600, and an existing one that other users can read is tightened to 0600 on open. Directories agentgate creates for it are 0700; a directory that already exists keeps its mode, and agentgate does not change the audit log file's mode.
- Fixture sync never deletes. A user removed from the fixtures is suspended, but a document removed from them stays in the store, readable as before, and is only reported. A fixture that moves a user or document to another customer is refused, and the open fails.
- A reservation still open 15 minutes after it was made is released on the next open, even if its call is still running. If that call then settles, its cost is recorded, so a customer day can end above its cap by that amount.
- Side effects are simulated. Email goes to an `outbox` table and is never sent. Parts orders and refunds are rows in the store; a refund does not check that the order exists or that the amount matches it.
- Every customer, person and document is invented. Some Juniper Row records are tagged `health-adjacent`; that data is made up too. agentgate has not been audited and makes no compliance claim of any kind.

## MCP gateway

- Stdio only. There is no HTTP transport; one needs real client authentication, which is not built.
- The gateway's `--as` is a demo identity, as on the rest of the CLI. One gateway process serves one user's run.
- Model spend is not metered through the gateway. It never sees the model's calls, so the per-run and per-customer-day USD caps count tool costs only. Tool costs and `max_calls_per_run` still apply to every tool call.
- Requests so malformed that the MCP SDK rejects them before any gateway handler runs may get no response at all. At commit `b9e650b`, raw JSON-RPC lines sent over stdio to `agentgate gateway serve --as kc-staff-nadine` got no reply for a malformed JSON line, for params sent as an array, and for arguments nested about 200 levels deep (200 got no reply, 195 did), so a client waiting for a reply can hang. A numeric tool name got error -32602. None of them is audited, and nothing runs for them.
- An exception inside the gate, for example the audit sink failing while an approval is created, reaches the client only as a generic `internal error`. The gateway logs the exception type on stderr. When the sink itself failed, nothing is recorded.

## Audit

- The audit log is a Panoptes hash chain in a local SQLite file. Editing, deleting or reordering a stored event fails `agentgate audit verify`. Someone with write access to the file can still delete the newest events and rewrite the chain metadata so that it passes `agentgate audit verify`, which has no checkpoint option. Catching that needs a checkpoint stored somewhere else: `panoptes checkpoint`, then `panoptes verify --checkpoint` (both installed with the dependency). agentgate does not take or store checkpoints itself.
- The audit log and the state database are separate SQLite files with no two-phase commit. "Recorded in the same transaction" means the event is written to Panoptes inside the agentgate transaction, before COMMIT, and a sink error rolls the change back. If the agentgate COMMIT then failed, the log would hold an event for a change that did not happen.
- The `allow` event is always recorded before a tool runs. The `executed` event is not guaranteed: if the sink fails after the tool has run, the ALLOW decision is on record, the side effect has happened, and the EXECUTED outcome event can be missing.
- Approvals store the full arguments, including email bodies, in the agentgate database so an approver can see exactly what will run. The audit log holds only hashes of them.
- Args hashes are unkeyed sha256 over the tool, the action and the arguments. Someone with only the audit log can confirm a guessed argument set when it has little entropy, such as a refund amount for a known order or a short internal email. Keyed references are only as private as the store's key: anyone who can read the store can compute the reference of a guessed value, and deleting the key leaves past references pointing at nothing.
- Three changes are made before their events are recorded. An operator suspension stands even if its event cannot be recorded, so an audit outage cannot keep a user active. A fixture sync and a stale spend sweep commit first; if the sink then fails, the open fails with the change made and no event for it.
- `agentgate audit check-store` trusts the chain, so someone who can rewrite the chain (above) can change a row and its hashes together. Rows written before the chain carried hashes are checked by id only.
- `open_world` puts nothing about the open on the chain unless the host passes `record_open`, so a host that leaves it out has no `policy_loaded` or `fixture_sync` events.

## Retrieval and policy

- Each search asks the policy engine about every one of the customer's documents, plus once more per returned result. That is fine at the fixture scale (20 documents per customer) and the first thing to change at scale. `notes.write` indexes only the new note, in one short transaction; the whole index is rebuilt only when the fixtures are loaded or a sync added or changed a document.
- The address rule is narrower than RFC 5322 (no quoted local parts, IP literals or non-ASCII addresses) and lowercases the local part, which RFC 5321 treats as case-sensitive.
- The risk threshold never changes a decision under the shipped policies. The highest-risk call they allow outright is a manager parts order of 500 USD (risk 45, threshold 50). Today the threshold is a backstop, not a deciding rule.
- Both shipped policies deny any email outside the company once a run has been shown a confidential or restricted document, through a read or a search result. That includes customers in the approved directory and approvals granted earlier in the run. It is a policy rule, it covers `email.send` only, and it is deliberately conservative: such a run has to end and a fresh run send the email.
- `recipient_in` on an allow or approval rule requires every recipient to be on the list, so a customer email that also copies a colleague matches no rule and is denied. On a deny rule one listed recipient is enough.
- Policies are loaded when a world is opened. A policy change takes effect in a new world.
- Calls sent for approval count toward `max_calls_per_run`, so a run can queue at most that many. Denied calls still do not count, so a run that only makes denied calls adds deny events to the log until its model calls hit the cap; a call naming a document is denied before spend and never counts.
- Any denial that depends on which document was named reads as `not_found`. Checks that run before the document is looked up (run, user, tool, arguments) return their own reason codes. For sends and spends the model sees the reason code, never the rule id.

## Testing

- Most attack scenarios assume a fully compromised model proposing calls directly. The leak detector looks for exact canary tokens; a model that paraphrased, re-encoded or split a canary would not be caught by it, and public and internal documents carry no canary.
- No test, CI job or report in this repository calls the live Claude API. The Anthropic adapter is tested only with a fake client, and no result here comes from a real model.
- Spend figures in the demo and the red-team suite come from a made-up mock price list (1 USD and 5 USD per million input and output tokens, 4 characters per token) and invented per-tool costs. They show the caps working, not what a real model would cost.
- The red-team suite's MCP run uses the MCP SDK's in-memory transport. The stdio launch is covered by one test in `tests/test_gateway.py`, not by the suite.
- CI runs lint, type checks, tests, policy tests and the red-team suite over both transports on Python 3.12 and 3.13. It does not run the benchmark.
