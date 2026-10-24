# Required admission — not executed

No hook, tokenizer, torch forward, GPU process, or experiment was run during preparation. This is an acceptance specification. Passing the static validator is not real-model admission. Any failed dependency stops the finite protocol; do not weaken tolerances or select passing edits.

## 1. Source and tokenizer admission (all10 edits, all20 views)

- Use the unchanged V62 image preprocessing, chat template, generation configuration and model. Prepare q0/q1/q2 separately using the normal runtime. Do not alter assistant prefixes to amplify directions.
- Compare target token ID sequences, including EOS, across all three batches; require exact equality and exactly one terminal EOS. Reject context truncation, changed image-token layout, target-answer text in a derived prompt, or scope-changing rewrite. No automatic view replacement.
- Expanded prefix length p must equal the first nonignored label position in the corresponding teacher batch. The patched position is p-1, after visual tokens and before the first answer token. Record raw length, expanded length, final prompt token, image range and label range separately for every view.
- Extraction receives only the prepared prefix and no labels/answer tokens. Frozen model parameters and both expert hooks are absent. A teacher-prefix comparison under causal attention must produce the same block21-input state as prefix-only extraction. The extraction artifact contains no post-answer state.
- Same source question at edits1 and9 has the same two derived question strings. Do not customize them by their differing targets. Directions can differ because images differ.

## 2. Injection contract

Let H be the FP32 residual input to `model.layers[21]`, shape [1,T,4096]. Let hk be its final-prefix vector in the unedited qk run. Compute detached dk=hk-h0 once. For the C augmented branch only:

```text
H_new = H.clone()
H_new[0,p0-1,:] = H[0,p0-1,:] + dk
```

This is a block forward-pre-hook before layer normalization and self-attention. It is not an addition to the 14336-dimensional down-projection input, nor to the down-projection output. Keep the existing TT/LoRA down-projection hook and ALL_TOKENS mask unchanged. At this site the unedited q0 context becomes hk, but other prompt states remain q0; do not claim equivalence to a full qk forward.

Only blocks0..20 are eligible for frozen teacher caching. Their returned tensor is cloned before patching; no in-place mutation of cached tensors. Block21 attention/MLP and all later layers are live. No edited-layer cache, patched-state cache, or teacher KV may be reused for free generation. Remove the intervention hook entirely at inference; preserve only the writer hook.

Zero or nonfinite directions, shape/position mismatch, or failed norm assertions stop the state hypothesis. Alpha is1; no rescaling to rescue admission. Record norms, h norms, cosine similarities and cancellation ratios without choosing views/layers from these values.

## 3. CPU mechanical checks required after future execution authorization

Use a small causal transformer fixture, clearly identified as a fixture, before a real-model run. Check:

1. Zero-direction hook is exact identity in logits, full-answer CE and expert gradients.
2. Exactly one specified prefix position changes at the intervention boundary, by exactly dk; all other coordinates are unchanged. No cached tensor changes after a request.
3. Direction is detached with no gradient or optimizer slot; frozen backbone remains untrained. Gradients reach the expert parameters through live block21+ computation.
4. Direct explicit input modification and hook modification agree in logits, loss and expert gradients. Check both teacher full sequence and prefix/stepwise decoding on the fixture.
5. Two separately computed identical native gradients, averaged in float64 before clipping, equal a single-native gradient and produce the same update. Loss scaling is not applied again inside the optimizer.
6. Branch order does not mutate expert/scalar/RNG state before the single update. Cache works separately for each q0/q1/q2 batch and never identifies them solely by owner or prefix length.
7. Random-vector generator is the locked NumPy PCG64 recipe; norm equality holds at rtol1e-5/atol1e-7 after float32 casting. Independent per owner/view, shared across structures. No orthogonalization or redraw.

## 4. Real-model admission required (both structures; all source/view bindings)

- With zero writer and no intervention, logits and source generation match the bound Base clean run exactly. With zero writer and nonzero intervention, compare against the SAME intervened Base computation, not clean Base: the intervention itself is allowed to change Base outputs.
- At initial writer state, all computed gradients are finite and the active zero-output factor (TT G1 / LoRA B) receives nonzero gradient. Zero gradient of an upstream factor behind a zero root is expected and not a failure.
- In a temporary declared nonzero-root state (fill output factor with0.001), compare direct and hooked intervention CE/gradient; restore exact initialization/scalars/RNG afterward. All expert gradients must be finite and aggregate gradient nonzero. Never leave this probe as the training initialization.
- Clean and injected losses and CPU-float64 gradient vectors must match uncached versus cached runs exactly, including a repeated cache hit. Perform at initialization and temporary nonzero-root state. The cache boundary remains before block21.
- For clean and intervened prefix state, compare teacher logits and real prefix+one-token KV logits at the first two prediction positions with the inherited fixed tolerance rtol1e-5/atol5e-4. This admits execution consistency, not scientific benefit. Do not relax tolerances on failure.
- Verify intervention removal yields exact clean logits/gradients with the same writer. Generation traces must show no intervention events. Full answers/EOS remain evaluated; do not replace them by first-token metrics.
- Save/load expert state must be exact. Record parameter count, module path, dtype, all gradient-pass/forward/generation counts and timing, including failed probes. Admission has a shared ceiling400 gradient passes and128 generations; exceeding it stops rather than borrowing from unrecorded retries.

## 5. Dependency order before scientific comparison

1. Explicit future launch authorization, remaining-window/storage admission and neutral process entries.
2. Static/future tokenizer and CPU mechanical gates.
3. Shared30 prefix extractions, freeze20 dk and20 random directions; no branch tuning from diagnostics.
4. Real-model mechanical gates and both full A160 replays. Compare each160-entry curve against V62 `SELECTION_FROZEN.json` and every219 fixed160 raw token sequence. A uses duplicate clean CE but one identical averaged update. If exact replay fails, stop B/C/R and preserve receipts.
5. All B/C branches, all frozen outputs, blind new labels and paired decision.
6. If conditional gate passes, both full R branches; otherwise record NOT_TRIGGERED.
7. All registered consumers/save-load/bindings complete, report/publish, then retire exact owned state list. No future training consumer is implicitly registered.

Checks above are implementation dependencies, not claims that a runner already exists. This preparation includes no executable training entry point.
