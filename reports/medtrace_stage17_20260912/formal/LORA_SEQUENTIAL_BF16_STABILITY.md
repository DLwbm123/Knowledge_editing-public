# Authorized LoRA sequential BF16 stability variant

On 2026-09-13 the user accepted the proposed separately labeled BF16 stability
experiment and requested a restart on my-gpu GPU2. This authorizes only
`LORA_SEQUENTIAL_BF16_STABILITY_V1`, starting from Base at edit 1, with the same
146-edit order, per-edit seeds, LoRA profile, 80 steps, learning rate, losses,
generation options and prefixes (1, 50, 100, 146).

The frozen official-native Base is loaded and validated in FP16, then converted
to BF16 before adapter creation. Language, projector and vision parameters and
prepared images use BF16; PEFT adapter optimization retains its existing FP32
parameter behavior. The base guard is captured after conversion. Dispatch and
every output binding carry the precision amendment; actual parameter dtypes are
recorded before editing. Frozen input bindings and historical FP16 Base answers
remain reference artifacts, not claims of BF16 Base-output equivalence.

This is an additional precision variant, not a successful rerun of the frozen
FP16 recipe. Its outputs must not silently replace the failed FP16 sequential
phase in the original campaign or be combined with FP16 single as a uniform
precision comparison. The original single 146/146 outputs and sequential 16/146
prefix, checkpoint and reproducible edit-17 NaN evidence remain preserved.

Only sequential is assigned to the new worker. It reuses the existing environment
and shared Base assets. Estimated new checkpoint weights are 15.4 GiB, with a
20 GiB run budget plus the existing 8 GiB disk reserve. Checkpoints' last consumers
are per-edit native generation, registered prefix panels and save/load validation.
The existing validated cleanup runs after the phase child exits; visibility
failures defer deletion and do not trigger retraining. Final scoring/public
delivery remains pending until actual generation completion and validation.

The existing hourly monitor follows this new GPU2 run and preserves the original
FP16 hard stop. BF16 is a stability hypothesis, not a guarantee of numerical or
scientific success; new failures retain evidence and are not bypassed.
