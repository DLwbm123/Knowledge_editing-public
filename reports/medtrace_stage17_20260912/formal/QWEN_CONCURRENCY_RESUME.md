# Qwen sidecar concurrency amendment — 2026-09-13

The user authorized pausing the independent Qwen sidecar and increasing concurrency
from one to two, without rejudging saved decisions. Astra is unaffected.

The stopped run had 550 persisted decisions. These are validated for ordered IDs,
input/config lineage, model snapshot, Boolean labels and raw JSON consistency.
The successor resumes at record 551 of 2,378 in a separate output directory.
Original output, execution state and logs remain preserved, with a pause record.
Malformed, duplicate, mismatched or incomplete saved records stop recovery rather
than silently dropping evidence. The successor does not add previous labels to any
model prompt and does not perform semantic retries.

Only effective `max_num_seqs` changes from 1 to 2. Original input/config remains
immutable; an explicit resume amendment and per-record concurrency metadata identify
the new segment. Same Qwen3-32B-AWQ snapshot, non-thinking setting, prompt, individual
record context, token limits, temperature, seed and output constraint are retained.
This is a mixed execution-concurrency run, not a bitwise-equivalence claim.

Existing validation test covers prefix retention, duplicate/out-of-order IDs and
partial JSON rejection. No new dependencies, model download or historical training
is required. This document records a resume, not completed scoring or speedup.
