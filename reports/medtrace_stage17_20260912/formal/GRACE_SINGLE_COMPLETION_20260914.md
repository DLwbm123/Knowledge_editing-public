# GRACE single generation completed

On 2026-09-14, the assigned worker completed all 146 frozen GRACE single edits
and automatically continued to GRACE sequential on the rented RTX 4090 24GB.
This is generation completion, not completed judging or a performance result.

The completion and phase bindings passed the existing provenance validator.
A per-edit count check confirmed 146 native outputs and 2,378 panel outputs
(2,524 total), with all training and edit-completion receipts present. The original
FP16 recipe, 100 training steps, frozen cohort/order and 1,024-token cap remain
unchanged. Runtime code commit: `63b4785233bac4bb05c2d4a46c8cd8cf60dee8b4`.

The destination completion receipt records 36,019.47 seconds, approximately ten
hours, for the resumed destination invocation. It excludes the earlier source
invocation and migration pause and must not be presented as total single-run time.

Checkpoint lifecycle completed without retaining an archive:

| Location | Weight files deleted | Bytes |
| --- | ---: | ---: |
| Rented destination | 146 | 11,018,200 |
| Original migration source | 68 | 5,131,760 |
| Local transfer staging | 68 | 5,131,760 |

Source cleanup was gated on the destination's actual completion, phase-binding
match, per-edit output counts, stopped source worker and exact file/mount/open-file
checks. Local transport copies were checked for open handles before removal.
Original outputs, tokens, bindings, configuration and migration/cleanup receipts
remain private. No weight backup remains; reconstruction requires retraining.
No paused LoRA checkpoint or unrelated process was changed.

At the transition check, GRACE sequential edit 1 was trained and its first panel
had 26 of 46 outputs; its main and child processes were active with neutral entry
paths. Subsequent registered phases remain BELoRA single and sequential. The main
campaign and result relays continue to wait for their required phases; LoRA BF16
remains explicitly paused. Full judging, final reports and the overall experiment
are still incomplete.
