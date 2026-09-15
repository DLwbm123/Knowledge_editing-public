# BELoRA single completion — 2026-09-15

BELoRA single completed all 146 edits on the assigned 40 GiB A100 GPU3 at
approximately 11:31 China time. The existing worker then automatically started
BELoRA sequential; the completion check found it at prefix 50, with 237/578 panel
outputs and 49 fully completed edits. Sequential state remains active.

## Verified generation and lifecycle

- Strict phase binding and cleanup validation passed against the frozen dispatch.
- Every edit has its training and completion receipts, one native output, and its
  expected panel coverage: 146 native + 2378 panel = 2524 total outputs.
- The phase receipt records 7729.394 seconds (2 h 8 min 49 s). This is the recorded
  phase duration including generation; it is not a pure training-time benchmark.
- The cleanup receipt confirms deletion of 146 generated state files totaling
  519,493,948 bytes. A single low-cost check found no remaining single-phase
  `state.pt` files. No checkpoint backup is retained; reconstructing those weights
  requires retraining. Outputs, tokens, bindings and receipts are retained.

The scientific runtime remains commit
`63b4785233bac4bb05c2d4a46c8cd8cf60dee8b4`. BELoRA remains the disclosed independent
paper-spec V2 effect-repaired adaptation with 50 steps, not an author implementation
or a paper-exact five-step run. Cohort, order, seed, precision, routing and generation
settings did not change during this phase.

The GPU3 bridge waits for both real BELoRA phases and their cleanup before moving
small result files into the persistent-server acceptance queue. GRACE sequential
and the rented-host final-copy workflow were still running at this check. LoRA
BF16 remains paused. This is a generation/lifecycle milestone, not a scientific
performance conclusion or completed final judging/reporting.

Public content is limited to this aggregate completion report. Private raw outputs,
questions/answers, tokens, image paths, weights and credentials are excluded.
