# DirectW exploratory pilot — 2026-10-01

## Authorization and scope

The user instructed: “你自己验证一下，可以了就立即实验”, followed by “去除这些限制，直接开始吧”. This supersedes the prior smoke-call budget and external-review stop. The isolated research branch now supports an explicitly labeled EXPLORATORY_PILOT exception; it does not turn incomplete data qualifications into verified scientific FIT roles.

Run `20261001E1` launches on pro5000 physical GPU 5, with a clean original LLaVA-Med Base and the sole editable native matrix `model.layers.21.mlp.down_proj.weight`. It uses no CP/LoRA adapter or optimizer state. PID at launch: 1201100; detached process command: `/data/bmw/envs/s0/bin/python -`.

## Frozen experiment

- Eight original training annotations, deterministically selected by source hashes and image-group round robin. Two permanent mechanical-smoke annotations are excluded.
- Independent clean-Base reset per case; W_FUNCTIONAL_QP only; at most three edit steps, 16 CG iterations, exact functional GGN, temporary FP32 functional model and BF16 deployed weights.
- Explicit slack, protection first-order term, normal BF16 score/KL checks, rollback and single-matrix audit use the shared editor.
- A distinct-image training reference has **unknown scope**. It is an exploratory KL reference, not a verified non-target medical case. No medical negative is generated.
- The repaired shared editor first undergoes native regression on the two reserved mechanical rows. Implementation errors or regression solver failures stop the process. Scientific-case rejections/nonconvergence remain results.
- Accepted matrices are checked by clean native reload and generated-token parity in a separate process, then the temporary exported matrix is deleted. Rejected cases restore Base.
- Four hours / 4096 new GGN calls are engineering safety ceilings chosen for this run, not a claim that the user specified those numbers. Previous usage remains 144.39321082888637 seconds and 46 GGN calls. Judge calls: 0.

Exact settings and source bindings: [PROTOCOL.json](PROTOCOL.json), [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json). The frozen source snapshot is separate from prior runs. No calibration, formal held-out evaluation, visual-evidence comparison or sequential editing is included.

## Verification and launch

The current source passed 36 CPU tests (20 Stage A numerical tests and 16 Phase B regression tests; 0 failures), including the explicit exploratory-authorization distinction, deterministic permanent-smoke exclusion and a real one-call GGN ceiling with rollback. CPU results do not establish native correctness or medical effectiveness.

Initial live checks confirmed the detached PID, neutral command line, GPU 5 UUID, and native model loading followed by `NATIVE_REGRESSION`. The run has started; this report does not claim completion or native regression success. A later live status/result receipt is authoritative.

## Data and interpretation limits

The source-verified pool contains 27 non-smoke training annotations across three image groups. Fact-family labels, legal non-target scopes, qualified visual counterexamples, historical teachers and independent calibration are incomplete. The original scientific data status remains **BLOCKED_DATA**. Any success measures training-support edit feasibility only; there is no clinical correctness, generalization or full W_EVIDENCE result here.

## Artifacts and follow-up

Remote output directory: `/data/bmw/Knowledge_editing/outputs/directw_evidence_v1/20261001E1/`. `STATUS.json` is atomically updated at phase/case boundaries. Detailed attempts, raw source rows, prompts, generated tokens and native tensors remain private. `stdout.private.log` and `FAILURE.private.json` (if produced) support diagnosis.

The process is independent of this chat/SSH session. No recurring monitoring or automatic next experiment has been created. When completion is next confirmed, publish sanitized case receipts and the final result report on this branch; do not label this launch report as completed experimental delivery.
