# Limits

This page lists what agentgate does not do and where its guarantees stop. The README lists the most important limits.

## Identity and deployment

- `--as` on the CLI names a user. It is not a login. This includes `agentgate gateway serve`. agentgate trusts the host application, or the person who starts the gateway, to authenticate users before a run starts.
- The run id is the run's only credential. `RunContext` is a plain dataclass and is not signed. The gate refuses altered fields, unknown run ids and look-alike objects. But any code in the host process that holds a live run id and its fields can act as that run.
- In-process, a run never expires, and the gate has no call that ends it. Only its spend caps bound it. The model gateway can end a run (see [Model gateway](#model-gateway)), but that only revokes the run's model tokens. The gate still accepts in-process tool calls for that run.
- In-process, agentgate protects only the tools that the gate alone can reach. Code in the same process that holds a tool's credentials, or the database, can skip the gate. The MCP gateway closes this gap for an agent that holds only the MCP connection, because the tools and the database are in the gateway process. Code that can open the database or the tools directly can still skip the gate.
- State is in one SQLite file for each world, plus a separate Panoptes SQLite file for the audit log. Several processes on the same machine can share these files (the approvals CLI does). There is no server database and no support for more than one host.
- agentgate creates the store file with mode 0600. If an existing store file is readable by other users, agentgate tightens it to 0600 on open. Directories that agentgate creates for the store are 0700. An existing directory keeps its mode. agentgate does not change the mode of the audit log file.
- Fixture sync never deletes. It suspends a user who was removed from the fixtures. But a document removed from the fixtures stays in the store and stays readable. The sync only reports it. If a fixture moves a user or a document to another customer, agentgate refuses it, and the open fails.
- agentgate releases a reservation that is still open 15 minutes after it was made, on the next open. It does this even if the call is still running. If that call settles later, agentgate records its cost. So a customer day can end above its cap by that amount.
- Side effects are simulated. Email goes to an `outbox` table and is never sent. Parts orders and refunds are rows in the store. A refund does not check that the order exists or that the amount matches it.
- Every customer, person and document is invented. Some Juniper Row records have the tag `health-adjacent`. That data is invented too. No one has audited agentgate, and it makes no compliance claim of any kind.

## MCP gateway

- The gateway uses stdio only. There is no HTTP transport, because that needs real client authentication, which is not built.
- The gateway's `--as` is a demo identity, as on the rest of the CLI. One gateway process serves the run of one user.
- A gateway started without `--model-proxy` does not meter model spend. It never sees the model's calls, so the run's USD caps count only tool costs. With `--model-proxy`, the proxy reserves and settles model calls against the same run. Then tool calls and model calls share the per-run cap, the per-day cap and `max_calls_per_run`. The caps are the policy's caps, tightened by `--max-usd` and `--max-calls`. These two options apply to model calls only.
- If the MCP SDK rejects a malformed request before any gateway handler runs, the request can get no response at all. At commit `b9e650b`, we sent raw JSON-RPC lines over stdio to `agentgate gateway serve --as kc-staff-nadine`. There was no reply for a malformed JSON line, for params sent as an array, or for arguments nested about 200 levels deep (200 levels got no reply, 195 did). So a client that waits for a reply can hang. A numeric tool name got error -32602. None of these requests is audited, and nothing runs for them.
- An exception inside the gate reaches the client only as a generic `internal error`. An example is the audit sink failing while the gate creates an approval. The gateway logs the exception type on stderr. When the sink itself failed, nothing is recorded.

## Model gateway

These limits apply to the model gateway: `Gate.admit_model`, provider keys, run tokens, the rate table and the proxy. The proxy has two modes. `agentgate model-proxy serve` serves one run. `agentgate model-proxy server` serves many runs of one world, which a host starts on a control socket.

### Trust and run lifetime

- The proxy serves one upstream for each wire protocol and one policy provider for each wire protocol. It listens on a loopback address.
- The token is a bearer secret. Whoever holds it can spend up to the run's caps. `--as` is a demo identity here too.
- The multi-run server's control socket trusts every process of the user who runs the server. Any of these processes can start a run for any active user of the world. This is the same trust as `--as`. Filesystem permission (a new 0700 directory and a 0600 socket) is the only authentication. No identity is verified when a run starts. One server serves one world.
- On the multi-run server, a run ends only when the host ends it (`POST /runs/{run_id}/end`) or when the server stops. A run that the host never ends lives until the server stops. Its token lives until then, or until its lifetime ends. The lifetime is at most `--max-token-ttl`, which is one hour by default.
- A multi-run server that is killed or crashes leaves its runs' tokens live. They stay live until their lifetime ends, or until the next server starts on the same world and ends them. The next server decides that the old server is gone by its process id. If the id was reused, the old server looks alive, and its runs live until their tokens expire. If a crash leaves a socket path given with `--control-socket`, you must remove it before the next start.
- With `serve`, a token dies when the serving process revokes its tokens, or when it expires (8 hours by default). A `serve` process that is killed leaves its tokens live until they expire. This matters only for a run that no process serves.

### What is decided

- Sometimes Panoptes refuses a request before the hook runs: malformed JSON, an oversized body, or no supported route. That request never reaches agentgate and has no decision event. The refusal is in Panoptes' events. Join on `exchange_id`.
- agentgate decides only the request. The proxy relays a streamed reply as it arrives. It does not inspect or hold back the reply. So the client must gate a tool call that a model emits in a reply. The MCP gateway does this for the tools it serves.
- A request with no declared `max_tokens` (or its equivalent) is denied, even when the policy sets a fixed `reserve`. The output limit is part of the envelope. Without it, agentgate cannot check the entry's `max_output_tokens`. A fixed reserve prices a model that the rate table does not price. For a model that the table prices, the fixed reserve is a floor under the table's price, and agentgate reserves the larger of the two.
- agentgate reads the output limit from the fields it knows: `max_tokens`, `max_completion_tokens`, `max_output_tokens`, and Gemini's `generationConfig` in both spellings. The largest one counts. A body with any other top-level field is denied (`model_unknown_field`), per provider. So a spelling that the code does not read cannot raise the limit. The cost is that agentgate refuses a field that a provider adds later, until someone adds it to the list in `model/admission.py`.
- agentgate checks only top-level fields. It does not check a field nested inside a known field. For example, it does not look at Gemini's `generationConfig.thinkingConfig` (a thinking budget) or `generationConfig.responseModalities` (image or audio output). It does not look inside Anthropic's `thinking` or `metadata` either.
- An Anthropic tool counts as a provider-run tool unless its `type` is exactly `custom` or absent. The match is case-sensitive. So if the provider accepts a `type` in another case, agentgate treats it as provider-run and denies it.
- A body must be UTF-8. Provider-side state, provider-run tools and prompt-cache writes are denied unless the policy entry sets `allow_server_features`.

### Pricing and charging

- The same rate table prices the reservation and the exchange, and agentgate trusts the table as written. A stale or wrong rate table moves every reservation with it. A reply with no usage, or a model that the table lacks, is charged the full reservation.
- The worst-case reservation bounds text, not everything a body can carry. agentgate counts the prompt as one token per byte of body. A tokenizer does not exceed that for text. But the provider prices an image, a file or an audio clip by its size or length, not by its bytes in the body. So a small body can still lead to a large bill.
- When a reply costs more than its reservation, the ledger records the actual cost. If that puts the run over its cap, agentgate refuses the next reservation. But N concurrent calls that each go over can pass the cap by N times the overage.
- agentgate releases a call only when the request never reached the provider, or when the provider refused it with a 4xx or 5xx and produced no output. So `max_calls_per_run` does not stop a client that sends requests the provider rejects. The provider's own rate limit is the only limit on those.
- Whether a request reached the provider is the proxy's report: `ModelExchange.request_sent`, from the Panoptes hook contract's `exchange.request_sent`. That value errs toward `True`. A proxy that reports it wrongly undercharges.
- Crash safety is a lease. If a process is killed between `before_forward` and `after_exchange`, its reservation stays open until the next sweep or open reclaims it (15 minutes by default). A sent call that never finished is then released, not charged.
- The proxy depends on `panoptes-observe>=0.2.0,<0.4`. The lock file and CI use 0.2.0. The hook contract belongs to Panoptes and can change. agentgate does not defend against a Panoptes that reports a wrong `request_sent`, `complete` or cost.

### Keys and audit

- Provider keys are plain strings in the agentgate process's memory. A key file or environment variable is only as private as the host account. A process that runs as the same user can read them.
- `redact` removes a key as written. It does not remove a key that was encoded again (base64, split or reversed). So it does not catch a provider that echoes a key in another form.
- A correlation id is the client's claim. agentgate records it as asserted and never uses it to decide who a run is. But a client can still choose which plain-shaped id its events carry, including the id of another run.
- agentgate records a model name that the policy does not list only as a reference. An operator who reads the log cannot see which model was requested, unless they check a guess with `audit_reference`.

## Audit

- The audit log is a Panoptes hash chain in a local SQLite file. If someone edits, deletes or reorders a stored event, `agentgate audit verify` fails. But a person with write access to the file can still delete the newest events and rewrite the chain metadata so that `agentgate audit verify` passes. That command has no checkpoint option. To catch this, you need a checkpoint stored in a different place: run `panoptes checkpoint`, then `panoptes verify --checkpoint`. Both come with the dependency. agentgate does not take or store checkpoints itself.
- The audit log and the state database are two SQLite files with no two-phase commit. "Recorded in the same transaction" means this: agentgate writes the event to Panoptes inside the agentgate transaction, before COMMIT, and a sink error rolls back the change. If the agentgate COMMIT then fails, the log holds an event for a change that did not happen.
- agentgate always records the `allow` event before a tool runs. The `executed` event is not guaranteed. If the sink fails after the tool runs, the ALLOW decision is on record and the side effect has happened, but the EXECUTED event can be missing.
- Approvals store the full arguments in the agentgate database, including email bodies, so that an approver can see exactly what will run. The audit log holds only hashes of the arguments.
- Args hashes are unkeyed sha256 over the tool, the action and the arguments. A person with only the audit log can confirm a guessed set of arguments when it has little entropy. Examples are a refund amount for a known order or a short internal email.
- Keyed references are only as private as the store's key. Anyone who can read the store can compute the reference of a guessed value. If you delete the key, past references point at nothing.
- Three changes happen before their events are recorded:
  - An operator suspension stays in effect even if agentgate cannot record its event. So an audit outage cannot keep a user active.
  - A fixture sync commits first. If the sink then fails, the open fails, with the change made and no event for it.
  - A stale spend sweep works the same way as a fixture sync.
- `agentgate audit check-store` trusts the chain. So a person who can rewrite the chain (see above) can change a row and its hashes together. For rows written before the chain carried hashes, the command checks only the id.
- `open_world` puts nothing about the open on the chain unless the host passes `record_open`. A host that leaves it out has no `policy_loaded` or `fixture_sync` events.

## Retrieval and policy

- Each search asks the policy engine about every document of the customer, plus one more time for each returned result. This is fine at fixture scale (20 documents per customer). At a larger scale, it is the first thing to change. `notes.write` indexes only the new note, in one short transaction. agentgate rebuilds the whole index only when it loads the fixtures, or when a sync adds or changes a document.
- Source ACLs are only as current as the fixtures, or whatever writes the store. agentgate does not sync from Drive, Gmail or any other source. A change at the source reaches decisions on the next fixture sync.
- agentgate compares principals as exact strings. So the host must spell a group, channel or team the same way in a document's ACL and in a user's memberships. agentgate does not expand nested groups.
- A host that calls `PolicyEngine.decide` directly must supply `Request.principals`. agentgate trusts it to compute them for the right user, as it trusts the host to authenticate that user.
- For email, recipient clearance checks whether the recipient may read the run's highest label. It does not check whether the recipient is on the ACL of each document the run has read.
- The address rule is narrower than RFC 5322. It does not allow quoted local parts, IP literals or non-ASCII addresses. It also lowercases the local part, which RFC 5321 treats as case-sensitive.
- Under the shipped policies, the risk threshold never changes a decision. The highest-risk call they allow outright is a manager parts order of 500 USD (risk 45, threshold 50). Today the threshold is a backstop, not a deciding rule.
- Both shipped policies deny any email outside the company after a run has been shown a confidential or restricted document, through a read or a search result. This includes customers in the approved directory and approvals granted earlier in the run. It is a policy rule, and it covers `email.send` only. It is conservative on purpose: such a run must end, and a new run must send the email.
- On an allow or approval rule, `recipient_in` requires every recipient to be on the list. So a customer email that also copies a colleague matches no rule and is denied. On a deny rule, one listed recipient is enough.
- agentgate loads policies when a world opens. A policy change takes effect in a new world.
- Mail to an internal recipient needs approval when the recipient may not read the most sensitive label the run has seen. The gate cannot see what a message says, so it decides by label only. So after a run reads a confidential document, an unrelated note to a staff member needs approval too. An internal address that names no user, such as a shared mailbox, always needs approval.
- Calls sent for approval count toward `max_calls_per_run`, so a run can queue at most that many. Denied calls do not count. So a run that makes only denied calls adds deny events to the log until its model calls reach the cap. A call that names a document is denied before spend and never counts.
- Any denial that depends on which document was named reads as `not_found`. Checks that run before the document lookup (run, user, tool, arguments) return their own reason codes. For sends and spends, the model sees the reason code, never the rule id.

## Testing

- Most attack scenarios assume a fully compromised model that proposes calls directly.
- The leak detector looks for exact canary tokens. It does not catch a model that paraphrased, encoded or split a canary. Public and internal documents carry no canary.
- The Anthropic adapter uses a default `max_tokens` of 2048, and thinking shares that budget with the answer. A turn cut off at that limit ends the run as `incomplete`, and its tool calls do not run. A higher default would make each call's spend reservation larger than the shipped per-run caps. The reservation is the full `max_tokens`, or twice that with fallbacks.
- No test, CI job or report in this repository calls the live Claude API. The Anthropic adapter is tested only with a fake client. No result here comes from a real model.
- Spend figures in the demo and the red-team suite come from an invented mock price list: 1 USD per million input tokens, 5 USD per million output tokens, and 4 characters per token. Per-tool costs are invented too. The figures show that the caps work. They do not show what a real model would cost.
- The red-team suite's MCP run uses the MCP SDK's in-memory transport. One test in `tests/test_gateway.py` covers the stdio launch. The suite does not.
- CI runs lint, type checks, tests, policy tests and the red-team suite over both transports, on Python 3.12 and 3.13. It does not run the benchmark.
- A second CI job runs the tests, the policy tests and the red-team suite against Panoptes `main` from git, on Python 3.12. It runs weekly and on manual dispatch, not on pushes, so a broken Panoptes `main` does not fail an agentgate pull request. The job fails if the installed Panoptes is not a git checkout.
