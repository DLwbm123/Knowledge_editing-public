# M3Bench reproduction status

## Boundary

This project stops after baseline, task construction, one-edit LoRA, and ten-edit sequential LoRA smoke gates. It must not start a 200-edit experiment without a separate explicit launch.

## Status at project creation

| Item | Status |
|---|---|
| Source audit and source freeze | COMPLETE |
| Release/paper protocol split | COMPLETE |
| Data audit | COMPLETE - 0 missing metadata images; 44 SLAKE metadata/raw QA text discrepancies recorded |
| Python 3.10 isolated environment | COMPLETE - existing frozen `.venv` verified; no package changes made |
| LLaVA-Med load and deterministic baseline | COMPLETE - Gate A passed after a model-specific continuation-only decode-contract repair; historical empty-output record retained |
| Full SLAKE/VQA-RAD baseline | NOT RUN |
| Release task construction | NOT RUN |
| LoRA one-edit smoke | NOT RUN |
| LoRA ten-edit sequential smoke | NOT RUN |
| GRACE/BalancEdit/MEND interfaces | PENDING (MEND exact reproduction blocked) |
| 200-edit experiment | NOT AUTHORIZED / NOT RUN |

## Gate A remediation record (2026-08-11)

The original `outputs/logs/llava_gate_a.json` is retained unchanged and records the historical two empty strings. Token forensics demonstrated that LLaVA-Med's multimodal `generate` call returns a continuation-only token sequence in this frozen Transformers 4.36.2 path. The previous adapter treated it as prompt-prefixed and sliced at the 21-token prompt length, leaving only whitespace and EOS.

`outputs/logs/gate_a_forensics/upstream_answer.jsonl` records a successful run of the exact frozen upstream `llava/eval/model_vqa.py` on the same image/question. `token_forensics.json` records finite logits, one image token, raw IDs, and the full-versus-legacy decode comparison. `llava_gate_a_three_case_validation.json` records three VQA-RAD cases repeated three times each: every answer is nonempty and both text and raw IDs are exactly deterministic.

Final Gate A label: `LLAVA_GATE_A_PASS__CONTINUATION_DECODE_FIXED`. No baseline construction, task building, edit method, LoRA, GRACE, BalancEdit, MEND, or 200-edit experiment was run.

## Next-phase base manifest gate (2026-08-11)

`LLAVA_NEXT_STOP__BASE_MANIFEST_INVALID`: canonical expansion correctly found 14,028 SLAKE and 2,248 VQA-RAD metadata QA rows, no duplicate canonical key, and retained all 44 known SLAKE text mismatch flags. However, released authoritative M3Bench metadata row `xmlab281_3` (global index 4467) has an empty `gold_answer`. The raw baseline contract requires every gold answer to be non-empty, and this project must not substitute a downloaded-source answer or fabricate one. No final base manifest was written; preprocessing audit, Gate B, GPU worker planning, raw baseline inference, judging, task construction, and all editing methods remain NOT RUN.

## Release-erratum continuation (2026-08-11)

`M3BENCH_RELEASE_ERRATUM__ONE_MISSING_GOLD_EXCLUDED_FROM_EVALUATION` now retains all 16,276 released records for inference while excluding only `xmlab281_3` from evaluation. Source/inference/evaluation manifests reconcile at 16,276/16,276/16,275; no gold was imputed. Gate B passed (100 records plus 20 three-way repeats). The raw baseline was not launched: no GPU was genuinely free after Gate B, and GPU 2 was running an unrelated LoRA process. Final label: `LLAVA_ERRATUM_STOP__BASELINE_INCOMPLETE`.
