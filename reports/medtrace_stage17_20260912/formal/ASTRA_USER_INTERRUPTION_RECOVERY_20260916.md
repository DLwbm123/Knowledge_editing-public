# User-authorized restart of the final scoring queue

The final queue accepted 62 of 68 batches. Batch 63 produced no final response
for more than four hours. On 2026-09-16 the user explicitly requested stopping
the wait and restarting from batch 63. The operator sent SIGTERM to that exact
child; the runner recorded exit code -15 and preserved its failed execution
record. Both child and parent exited, with no final response present. This is
a user-requested interruption, not a diagnosed network timeout.

The recovery validator now accepts this case only with an explicit, batch-bound
user authorization and interruption receipt confirming SIGTERM, process exit,
and absence of a final response before and after interruption. It still rejects
changed bindings, failed isolation, tool events, semantic errors, and existing
final responses. All 62 accepted batches are reused through the existing
predecessor chain. One fresh attempt starts at batch 63 and continues through
68 with the original model, reasoning effort, inputs, schema, and ordering.
No GPU experiment is restarted and no accepted verdict is rescored.

Validation: 10 existing/synthetic tests passed across judge recovery, campaign,
and report tests, including rejection cases for unauthorized interruption,
changed bindings, live processes, final responses, semantic errors, and
isolation/tool violations. An initial unittest module invocation could not
import the non-package tests directory; direct test-file execution passed.

This engineering note does not claim completion of the new attempt or final
campaign scoring. Raw answers and private operational receipts remain private.
