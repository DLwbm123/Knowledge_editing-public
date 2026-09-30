"""Phase B0 source-only qualification and mechanical result isolation."""
from __future__ import annotations
import json
from pathlib import Path
from .contracts import FIT, SMOKE, audit_data, digest, method_eligibility

BRANCHES = ('W_FT', 'W_EUCLIDEAN_QP', 'W_KEY_QP', 'W_FUNCTIONAL_QP', 'W_EVIDENCE_QP')
RESULT_KIND = 'MECHANICAL_NATIVE_VALIDATION'


def qualify(source: dict, authorization_reference: str) -> tuple[dict, dict, dict]:
    """Two deterministic smoke rows; unresolved scientific permissions stay blocked.

    Source annotations establish original VQA provenance, not corrected edit targets,
    scope exclusions, equivalent paraphrases, or accepted historical teachers.
    """
    raw = source['rows']
    if not source['support_isolated'] or any(x['source_role'] != 'train' for x in raw):
        raise ValueError('non-isolated or non-training source')
    rows = []
    for x in raw:
        image_hash = source['image_content_digests'][x['image_path']]
        if image_hash != x['image_sha256']:
            raise ValueError('source image content changed')
        if source['source_file_digests'][x['source_file']] != source['provenance']['train_sha256']:
            raise ValueError('training annotation content changed')
        row = dict(id=digest([x['dataset'], x['source_release'], x['source_qid']]),
            source_group=x['source_group'], patient_group=x['patient_id'], fact_family=None,
            image_hash=image_hash, question_hash=digest(x['question']), answer_hash=digest(x['reference']),
            role='CANDIDATE_ONLY', original_role='TRAIN_ADAPTATION', source_split='train',
            permission_basis=dict(original=source['provenance'], phase_b=authorization_reference),
            fit_permission_verified=False, mechanical_permission_verified=True,
            available_at_edit_index=None, temporal_availability_evidence='No DirectW event timeline or accepted edit exists',
            scope_evidence='SCOPE_UNKNOWN', annotation_source=x['source_annotation'],
            ever_developed=None, ever_scored=None, teacher_reference=None,
            image_path=x['image_path'], question=x['question'], answer=x['reference'],
            original_source_qid=x['source_qid'], original_source_file=x['source_file'],
            proposed_roles={role:'BLOCKED_MISSING_ROLE_EVIDENCE' for role in sorted(FIT)})
        rows.append(row)
    rows.sort(key=lambda x:x['id'])
    for row in rows[:2]:
        row.update(role=SMOKE, available_at_edit_index=0,
                   temporal_availability_evidence='Original training annotation already present before this mechanical run; smoke only')
        row['proposed_roles'] = {role:'EXCLUDED_MECHANICAL_ONLY' for role in sorted(FIT)}
    smoke = [x for x in rows if x['role'] == SMOKE]
    mechanical_audit = audit_data(smoke, edit_index=0, test_ids=set(), expected_model=None)
    scientific = dict(eligible_counts={}, status='BLOCKED_DATA')
    eligible = method_eligibility(scientific)
    audit = dict(status='BLOCKED_DATA', audit_execution='COMPLETE', candidate_rows=len(rows),
        candidate_source_groups=len({x['source_group'] for x in rows}),
        smoke_data_audit_status=mechanical_audit['status'], mechanical_smoke_inputs=len(smoke),
        smoke_findings=mechanical_audit['findings'], scientific_role_counts={role:0 for role in sorted(FIT)},
        original_annotation_digest_verified=True, original_image_content_verified=True,
        independent_dev_cal='NOT_ESTABLISHED', independent_test_confirm='NOT_ESTABLISHED',
        test_qa_read=0, old_evaluation_qa_read=0, model_scores_read=0, synthesized_medical_negatives=0,
        selection_basis=source['selection_basis'],
        blockers=dict(EDIT_FIT='No verified corrected-edit target/role permission and fact-family timeline',
            GEN_FIT='No independently verified same-fact paraphrase linkage',
            PROTECT_BG_FIT='No reviewed scope ledger, frozen Base masks, or legal bound teacher',
            PROTECT_NEAR_FIT='No VERIFIED_OUT_OF_SCOPE evidence',
            VIS_PAIR_FIT='No independently annotated equivalent question pair, incompatible answer, negative scope or bound Base correctness',
            PAST_EDIT_MEMORY='No accepted DirectW edit or historical teacher'))
    methods = {branch:dict(scientific_data_status='BLOCKED_DATA',
        native_dependent_status=eligible[branch], legal_counts=audit['scientific_role_counts'],
        independent_source_count=0, available_DEV_CAL=0, available_TEST_CONFIRM=0,
        pilot_allowed=False) for branch in BRANCHES}
    return dict(version=1, private=True, rows=rows, authorization_reference=authorization_reference), audit, methods


def write_b0(source_path: Path, private_root: Path, report_root: Path, authorization_reference: str) -> None:
    manifest, audit, methods = qualify(json.loads(source_path.read_text()), authorization_reference)
    private_root.mkdir(parents=True, exist_ok=True); report_root.mkdir(parents=True, exist_ok=True)
    for path, value in [(private_root/'DIRECTW_ROLE_MANIFEST_V1.json',manifest),
                        (report_root/'DIRECTW_DATA_AUDIT_V2.json',audit),
                        (report_root/'DIRECTW_METHOD_ELIGIBILITY_V2.json',methods)]:
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
