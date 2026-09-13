# User-approved LoRA acceptance amendment

On 2026-09-13 the user explicitly approved replacing the campaign's failed FP16
LoRA sequential requirement with the separately identified BF16 stability run.
The acceptance identifier is `LORA_SINGLE_FP16_SEQUENTIAL_BF16_V1`.

The accepted pair is the existing completed FP16 single phase and the continuing
BF16 sequential phase. Each retains its original runtime, code commit, method
lock, 146-edit order, inputs and output bindings. The import validates these
against separate immutable phase assignments; it does not rewrite either phase
to claim common provenance. The original FP16 sequential failure at edit 17,
step 4 remains a failed experiment with its valid prefix and diagnostic evidence.

The rented queue now waits for this accepted pair's actual completion,
provenance validation and checkpoint lifecycle receipts, instead of waiting for
the failed FP16 sequential run to finish. The original queue parent and active
GPU workers remain unchanged. Separate source directories are used for each
phase, and a failure in the unused FP16 sequential phase cannot mark its complete
FP16 single phase as failed. No incomplete BF16 run is marked complete.

Judge packets retain the actual phase runtime and generation dtype. Aggregate
JSON, CSV and the review report explicitly label LoRA single as FP16 and
sequential as BF16. Their comparison, and FP16 C_NO_H versus BF16 LoRA sequential,
is not precision matched and cannot isolate a method-only effect. Frozen Base
answers and masks and already accepted Astra verdicts are not changed or rejudged.

This amendment authorizes result acceptance, not additional training, tuning or
a claim that the FP16 recipe succeeded. Strict checkpoint cleanup checks remain;
any independent filesystem visibility blocker remains explicit until resolved.
