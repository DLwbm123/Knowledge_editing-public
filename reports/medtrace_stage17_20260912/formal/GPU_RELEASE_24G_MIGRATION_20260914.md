# Authorized GPU release and baseline migration

On 2026-09-14 the user approved releasing both active my-gpu devices: pause the
LoRA BF16 sequential run on GPU0, and migrate the GRACE worker from GPU1 back to
the rented RTX 4090 24GB server. The existing subsequent GRACE sequential and
BELoRA single/sequential queue remains assigned; no additional experiment was added.

Both source workers stopped at the existing cooperative STOP boundary. LoRA
retains 99 completed edits, the trained edit-100 checkpoint and 764 of 1,077
prefix-100 panel outputs. GRACE retains 67 completed single edits, the trained
edit-68 checkpoint and 13 panel outputs for that edit. The pause-related failure
receipts were preserved as operational evidence; they are not numerical failures.
The source STOP markers remain, and all three my-gpu GPU0/1/2 devices are excluded
from automatic restart until the user explicitly authorizes a new allocation.

The destination reuses the existing environment, verified official Base/vision
assets and clean baseline runtime commit
`63b4785233bac4bb05c2d4a46c8cd8cf60dee8b4`. Only operational dispatch fields change.
Original cohort, order, seeds, method/precision locks and existing phase bindings
remain intact. This is a cross-hardware continuation, not a claim of bitwise
hardware equivalence. Existing training and evaluation receipts are reused by the
normal resume path; unfinished queries continue under the same generation policy.

The 24GB GPU had 24,082 MiB available before migration. The data filesystem had
approximately 60 GiB available and passed a small write/read probe, covering the
existing 48 GiB baseline lifecycle budget plus the 8 GiB reserve. Each subsequent
phase retains the worker GPU-memory and disk-reserve checks. Checkpoint cleanup
remains dependency-bound; the paused LoRA resume state must not be deleted.

Baseline result relay configuration follows the destination worker and keeps the
existing provenance-validated import slot. The original campaign may wait for
paused LoRA while the independent baseline worker proceeds. LoRA transfer is
paused deliberately; no successful completion/import marker is fabricated.
The pre-existing FP16 single checkpoint-cleanup permission issue remains separate.

Status verified after deployment: both source experiment processes exited. The
destination worker resumed, completed edit 68, and saved the edit-69 training
receipt with new panel output. GPU memory use was approximately 15 GiB and no
phase failure was present. Main and child command lines used neutral entry paths.
The baseline relay is waiting on the new worker; hourly monitoring follows the
new assignment and respects the explicit LoRA pause.

Transfers completed using rsync compression and exit-status checks, followed by
one destination resume-state/readability check. No additional content hashes or
bitwise-equivalence claims were introduced. This report records deployment only;
full generation, judging and final experiment delivery remain incomplete.
