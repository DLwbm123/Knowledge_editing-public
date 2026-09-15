# Stage17 concurrent experiment resume and completed-phase scoring

Snapshot: 2026-09-15 21:36 China Standard Time. This is a startup record, not a completed-results report.

The user authorized resuming unfinished experiments while scoring completed methods. The main T0 cohort remains N=146 (E_U=146, E_H=0); this does not add the unsupported H-dependent full C_FACT branch.

## Experiment resume

The remaining main-cohort generation phase, LoRA sequential with the previously approved BF16 amendment, has resumed on GPU3. Before resume, 99 edits had complete evaluation receipts; edit 100 had saved training state and 764/1077 panel records. The existing loop loads earlier saved states and retains prior outputs before completing edit 100 and continuing edits 101–146. Its temporarily low startup counter reflects replay of completed states, not a restart of training.

The scientific runtime stays at commit `7ba912c9b29d03a3113d5b76a3f7066391e69aac`. The resume changes device scheduling only: the original cohort, order, method configuration and BF16 precision are preserved. The previous user-pause marker was archived with the new authorization. The detached parent and GPU child are live, use neutral command lines, and the GPU child is assigned to the selected GPU UUID. Completed checkpoints remain available to their remaining consumers under the existing lifecycle policy.

## Concurrent scoring

A separate frozen queue has started: **17,198 new records in 344 batches**, using **gpt-6-astra / high** under the existing source-agreement protocol. The six newly scored phases are:

- BalancEdit sequential
- LoRA single
- GRACE single and sequential
- BELoRA single and sequential

The queue reuses 5,732 accepted C_NO_H single/sequential judgments; previously accepted BalancEdit single and Base judgments remain separate reusable inputs. No accepted judgment is scheduled for a fresh judgment. The initial execution record is RUNNING, and the first batch has started with all five configured isolation checks passing. No new score is claimed in this startup record.

The final follower waits for this queue and the remaining LoRA generation. Its final queue excludes previously accepted records, including the nested C_NO_H reuse, and then scores the remaining LoRA sequential outputs. The LoRA import relay is also running.

## Controller change and validation

Controller commit `0d19525` adds explicit completed-phase selection and validated nested accepted-judgment reuse to the existing closeout workflow. Partial reports list pending phases and omit unavailable phase metrics rather than inventing zero scores. Seven focused tests pass, covering existing campaign behavior, phase selection, absent-phase reporting, and nested reuse.

The existing 30-minute heartbeat remains active for fresh process, output, relay and Judge checks, including authorized recoverable fault handling. It must preserve accepted verdicts and scientific bindings, and notify on meaningful progress, completion, failure, or required user action.

Raw prompts, answers, credentials, private artifacts and model weights are excluded from this public record. The experiment and scoring remain in progress; no completion time or final scientific conclusion is asserted here.
