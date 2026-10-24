# Fixed training-only local-gradient diagnostic

Reuses the V54 teacher-forcing batches, objective, TT hook and runtime; no cache.
V55_ROOT/PARTITION/PARENT_RUN select the isolated run. V54_SOURCE points to the
two explicitly retained final deltas. Compare ZERO and V54_FINAL only, without
mutating the saved weights. Expected per worker24 forwards/32 backwards/0 updates/
0 generations, enforced in receipts. Launch under the independent1800-second
worker guard within the unchanged parent deadline.

Private gradient_evidence.pt contains small detached derivative vectors, not a
model checkpoint. Public summaries contain cosine matrices, current margins and
unit-descent directional derivatives. These are local coordinate diagnostics,
not actual canonical-RMS steps, global infeasibility certificates or medical
performance evidence. No new optimizer is implied by the result.
