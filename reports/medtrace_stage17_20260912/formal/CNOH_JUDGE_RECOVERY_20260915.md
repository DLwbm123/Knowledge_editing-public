# C_NO_H priority Judge transport recovery

The first priority queue stopped at batch 26 on September 15 at 16:10 China time
after a response-stream transport decoding error. Batches 1–25 contain 1,250
accepted records. The failed batch produced no final response and no tool event.
The user explicitly approved one same-input recovery and continuing the remaining
unattempted batches, preserving the accepted prefix and original failure.

Recovery started at 18:51 China time using the unchanged gpt-6-astra/high Judge
lock, prompt, batch ordering, proxy transport and CLI 0.154.0-alpha.6.2. The
existing recovery preflight validated all 25 accepted batches and the failed
attempt. A separate recovery directory holds new evidence; the original failed
execution record remains unchanged. The first recovered batch passed all five
isolation probes and entered a fresh session. This is startup, not completion.

The campaign follower, accepted-result reuse and report reader now select the
latest supported recovery execution record. This prevents an old preserved
failure from blocking a later successful recovery. The recovered queue will
merge all 5,732 verdicts only after full format and ID coverage validation, then
produce the C_NO_H-only report and attempt sanitized public delivery. No accepted
batch is rejudged and no GPU work is restarted.

The user also authorized the existing half-hour monitor to resolve recoverable
network and engineering failures. Same-input transport recovery requires absent
final response, intact input/configuration/isolation bindings and preserved
accepted results; it is bounded to one dispatch per monitoring check. This does
not authorize semantic retries, model changes, changed references, replacement
scores or relaxed validation. Successful recovery and actionable failures are
reported; unchanged healthy progress stays quiet.

Validation: six campaign tests and the existing Judge recovery test pass. They
cover accepted-prefix preservation, transport-only recovery, scoped reporting,
and use of a completed recovery without overwriting the initial failure. Public
content excludes private raw questions, answers, tokens, bindings and credentials.
