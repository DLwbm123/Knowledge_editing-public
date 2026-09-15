# BELoRA GPU3 scheduling split — 2026-09-15

The user assigned an available 40 GiB A100 GPU3 to the not-yet-started work in the 24 GiB server queue. BELoRA single and sequential are assigned to GPU3 in that order. GRACE sequential continues uninterrupted on the RTX 4090. No BELoRA phase had started on the original server when the split was installed. LoRA BF16 remains paused; GPU0, GPU1 and GPU2 remain unassigned to this campaign.

## Scientific provenance

Both BELoRA phases use the existing clean runtime commit `63b4785233bac4bb05c2d4a46c8cd8cf60dee8b4`. The operational controller is commit `58f499bdba0ad153d9f68b9a8d4b2194924b5fa0`; it does not change the frozen runtime checkout. Dispatch comparison confirmed matching cohort, order, seed, model/runtime lock and method configuration across both servers. N=146, E_U=146, E_H=0 and all generation settings remain frozen. BELoRA remains the disclosed independent paper-spec V2 effect-repaired implementation with 50 steps, not an author implementation or paper-exact five-step run.

This is a method-level scheduling split. The cumulative sequential experiment is not divided into independent edits. Existing model/data/environment copies are reused. No experimental weight archive is moved between hosts.

## Execution and return path

`scripts/medtrace/stage17_split.py` runs only BELoRA single followed by sequential using the pinned phase entry and the existing provenance validation, space checks and strict checkpoint cleanup. Active resume state is retained; completed generated checkpoints are deleted only after their existing required consumers and checks finish. A cleanup visibility failure remains explicit and retains state.

The original controller and active GRACE child are not restarted or hot-patched. Only future child entry invocations for BELoRA are redirected to a wait-and-validate gate. Both real phase results and cleanup receipts are transferred through a finite local relay after GPU3 completion. Destination validation checks the original frozen bindings before publication into the original baseline run. Existing directories are never overwritten. The existing four-phase relay and overall follower then continue their normal acceptance pipeline. Neither an import marker nor a completion receipt is fabricated.

The bridge needs the local computer online. Engineering failures are reported and handled by the already-authorized hourly monitor. GPU3 is now authorized for this BELoRA assignment only; the user's prior LoRA pause remains effective.

## Validation and limitations

- GPU3 UUID and available memory verified before launch; approximately 40 GiB free.
- Required NFS mount and capacity verified; a small write/read probe passed with the 48 GiB lifecycle budget plus 8 GiB reserve available.
- All 779 required image paths available; existing model configuration files readable.
- Runtime packages match the recorded stack: Torch 2.6.0, Transformers 4.51.3, PEFT 0.19.1, Accelerate 1.14.0 and safetensors 0.5.3.
- Gate regression test passes: rejects an unassigned method, stops on an explicit stop, requires completed cleanup, and propagates binding validation failure.
- Neutral main and child command lines verified; GPU process is confined to the assigned physical GPU3.
- Startup produced a complete first edit (1/146) and advanced to the second edit without a phase failure. GRACE remained active on the original server.
- In the filtered public repository, controller and test files live under `experiments/medtrace_stage17_20260912/`, alongside the existing runtime snapshot.

This report records a scheduling/startup change, not completed generation, judging or scientific performance. Private dispatches, raw questions/answers/tokens, image paths, credentials and weights are excluded from public delivery.
