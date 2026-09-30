# Phase B0 — source-only data qualification

`audit_execution = COMPLETE`. `scientific_data_status = BLOCKED_DATA`. This is a qualification result, not a failure of the editing methods.

The reviewed baseline is `0be4446be7c472cb942b5c616dc04117a52096f3` on `research/directw-evidence-v1`, repository `DLwbm123/Knowledge_editing-public`. Stage-A evidence and CP/LoRA artifacts are preserved. No old adapter, optimizer, Judge verdict or model score was loaded.

The audit inspected 29 previously source-isolated training/adaptation annotations from three SLAKE image groups. The recorded original training release, CC-BY-4.0 provenance, original annotation-file content digest and image-content digests were checked. Existing train metadata is evidence of original provenance; it does not by itself establish corrected edit targets, per-role DirectW permission, fact-family scope, future event timing, bound Base correctness, accepted history or independent calibration.

Two rows were reserved deterministically by source identity hash as `SMOKE_MECHANICAL`, without model-result selection. They are permanently excluded from all scientific FIT, development, confirmation and Pilot denominators. Mechanical permission is the user's explicit Phase-B prompt; it does not grant scientific FIT permission. Operational paths use `/data/bmw/` on pro5000, preserving the historical source paths in the private manifest. The two mechanical image content digests were also checked on pro5000 before model loading. No formal test QA, old evaluation QA, scores or counterexample generation were used.

| Role | Legal scientific count | Qualification blocker |
|---|---:|---|
| EDIT_FIT | 0 | No verified corrected-edit target, role permission or event/fact-family ledger |
| GEN_FIT | 0 | No independently verified same-fact paraphrase linkage |
| PROTECT_BG_FIT | 0 | Scope ledger, frozen Base masks and bound legal teacher not established |
| PROTECT_NEAR_FIT | 0 | No `VERIFIED_OUT_OF_SCOPE` evidence |
| VIS_PAIR_FIT | 0 | Required equivalence, independent labels, incompatibility, negative scope and bound Base correctness not established |
| PAST_EDIT_MEMORY | 0 | No accepted DirectW edit or accepted historical teacher |
| SMOKE_MECHANICAL | 2 mechanical only | Original train/adaptation provenance and mechanical reading permission checked |
| DEV_CAL | 0 | `NOT_ESTABLISHED` |
| TEST_CONFIRM | 0 | `NOT_ESTABLISHED` |

Each scientific candidate records original split/role, original provenance and permission basis, source/patient information, image/question/answer hashes, annotation provenance, role evidence and unresolved fact-family/time/scope fields. Unknown patient identity and fact-family linkage are retained as unknown; no medical fact family is fabricated. Mechanical availability is index zero of this static smoke pool only; it is not a scientific event timeline. The 27 unreserved candidates remain `CANDIDATE_ONLY`, with explicit missing evidence for each scientific role.

All five branches are `BLOCKED_DATA`. The four non-Evidence branches are blocked by their own missing common scientific roles; they are not blocked solely because visual pairs are absent. W_EVIDENCE_QP has the additional visual-pair blocker. No native medical evidence edit is allowed from this audit.

The complete `DIRECTW_ROLE_MANIFEST_V1.json` is private under the independent local RUN_ROOT `outputs/directw_evidence_v1/20260930B1/`. It contains raw question/answer and source paths and is excluded from GitHub. Public companions are `DIRECTW_DATA_AUDIT_V2.json` and `DIRECTW_METHOD_ELIGIBILITY_V2.json`.
