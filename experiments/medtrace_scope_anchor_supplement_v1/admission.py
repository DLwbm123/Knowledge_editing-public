"""Freeze reviewed source anchors and a separately versioned edit, without models."""
import collections
import copy
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def admitted(rows):
    return (collections.Counter(r['owner'] for r in rows) == {i: 2 for i in range(1, 9)}
            and all(len({r['image_file'] for r in rows if r['owner'] == i}) == 2 for i in range(1, 9))
            and all(r['admitted'] and r['relation_status'] == 'PASS_SOURCE_EVIDENCE'
                    and r['basis_conflict_review'] == 'PASS_NO_CONTRADICTORY_TARGET'
                    and r['image_review'] == 'PASS_READABLE_SOURCE_FIGURE'
                    and r['license'] for r in rows))


def selfcheck():
    row = dict(admitted=True, relation_status='PASS_SOURCE_EVIDENCE',
               basis_conflict_review='PASS_NO_CONTRADICTORY_TARGET',
               image_review='PASS_READABLE_SOURCE_FIGURE', license=['source terms'])
    good = [dict(row, owner=i, image_file=f'{i}_{j}') for i in range(1, 9) for j in range(2)]
    assert admitted(good) and not admitted(good[:-1])
    bad = copy.deepcopy(good); bad[-1]['basis_conflict_review'] = 'UNKNOWN'
    assert not admitted(bad)
    bad = copy.deepcopy(good); bad[-1]['image_file'] = bad[-2]['image_file']
    assert not admitted(bad)
    return dict(status='PASS', missing_owner_rejected=True, unknown_review_rejected=True,
                duplicate_image_rejected=True)


def main():
    checks = selfcheck()
    out = Path(os.environ['SUPPLEMENT_ROOT'])
    parent = Path(os.environ['SUPPLEMENT_PARENT'])
    source = out/'private'
    command = subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'args='], text=True).strip()
    assert not any(x in command.lower() for x in ('wangbomin', 'knowledge_editing', 'medtrace', 'supplement'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    review = read(source/'SOURCE_REVIEW.json'); anchors = review['rows']
    assert admitted(anchors), 'Source evidence gate not met'
    pool = read(parent/'private/ELIGIBLE_SOURCE_POOL.json')
    env = read(parent.parent.parent/'medtrace-fp32-protection160-v1-20261009/run/private/LAUNCH_ENV.json')
    original = read(parent.parent.parent/'medtrace-fp32-protection160-v1-20261009/run/private/BENCHMARK146_QUEUE.json')
    tasks = copy.deepcopy(original['tasks'][:8])
    qualification = read(Path(env['FP32_BASELINE_PARENT'])/'private/FP32_SEMANTIC_QUALIFICATION.json')['rows']
    basis = [r['row'] for r in qualification if r['role'] == 'BASIS' and r['correct'] == 1]
    assert len(basis) == 61
    excluded = {r['row']['source_group'] for r in qualification}
    excluded.update(t['native']['source_group'] for t in tasks)
    rows = []
    for index, a in enumerate(anchors):
        image = (source/a['image_file']).resolve()
        assert image.parent == source.resolve() and image.is_file() and image.stat().st_size > 1000
        assert a['case'] not in excluded
        evidence = []
        if a.get('source_qids'):
            dataset, image_id = a['case'].split(':', 1)
            evidence = [r for r in pool if r['dataset'] == dataset and r['image_id'] == image_id
                        and int(r['qid']) in a['source_qids']]
            assert len(evidence) == len(a['source_qids'])
            if a['annotation_kind'] == 'ORIGINAL_SOURCE_QA':
                assert any(r['question'] == a['question'] and r['answer'] == a['reference'] for r in evidence)
        else:
            assert a['source_url'].startswith('https://pmc.ncbi.nlm.nih.gov/articles/')
            assert (source/a['source_xml']).is_file() and a['source_caption'] and a['image_member']
        rows.append(dict(query_id=f'SCOPE_FIT_{index+1:02d}', owner=a['owner'],
            dataset='SOURCE_ANCHOR', image_path=str(image), original_image_path=str(image),
            source_group=a['case'], question=a['question'], reference=a['reference'],
            role='GFIT', review_index=index, source_evidence=evidence))
    revision = read(source/'OWNER8_REVISION.json')
    assert len(revision['fit_questions']) == len(revision['t1g_questions']) == len(revision['t2g_questions']) == 4
    t = tasks[7]; old_id = t['edit_id']; new_id = 'OWNER8_RISK_V1_NATIVE'
    t['edit_id'] = new_id
    t['native'].update(query_id=new_id, question=revision['native_question'], reference=revision['reference'],
                       base_correct=None, revision='OWNER8_RISK_V1')
    t['native'].pop('opaque_Base_id', None)
    t['fit_questions'] = revision['fit_questions']
    replacements = []
    for role, questions in [('T1G', revision['t1g_questions']), ('T2G', revision['t2g_questions'])]:
        for i, question in enumerate(questions):
            row = dict(t['native'], query_id=f'OWNER8_RISK_V1_{role}_{i+1}', question=question,
                       audit_role=role, original_primary=False, same_reference=True)
            if role == 'T2G':
                image = source/revision['t2g_images'][i]
                assert image.is_file() and image.name not in {a['image_file'] for a in anchors}
                row.update(dataset='SOURCE_REVISION_EVAL', image_path=str(image), original_image_path=str(image),
                           source_group=f'REVISED_EVAL_CASE_{1+i//2}')
                row.pop('image_sha256', None); row.pop('opaque_Base_id', None)
            replacements.append(row)
    assert len(t['events']) == len(original['tasks'][7]['events'])
    for event in t['events']:
        event['edit_query_id'] = new_id
        event['event_id'] = event['event_id'].replace(old_id, new_id)
        if event['task'] == 'T0': event['all_probe_query_ids'] = [new_id]
        if event['task'] in ('T1G', 'T2G'):
            event['all_probe_query_ids'] = [r['query_id'] for r in replacements if r['audit_role'] == event['task']]
    assert tasks[:7] == original['tasks'][:7]
    write(source/'FROZEN_ANCHORS.json', rows)
    write(source/'FROZEN_TASKS.json', tasks)
    write(source/'REVISED_EVENT_ROWS.json', replacements)
    write(source/'BASIS_RELATION_REVIEW.json', dict(status='PASS_SOURCE_EVIDENCE', count=61,
        rows=[dict(query_id=r['query_id'], question=r['question'], reference=r['reference'],
                   relation='PRESERVE_ORIGINAL_ANSWER_NO_CONTRADICTORY_TARGET') for r in basis],
        note='Mostly organ, modality, position, plane and count questions; the abnormality-count item is not relabeled. No HELDOUT answer used for anchor construction.'))
    result = dict(status='SOURCE_EVIDENCE_GATE_PASS', owners=8, anchors=16, unique_images=14,
        per_owner=[dict(owner=i, anchors=2) for i in range(1,9)], original_source_QA=2, derived_QA=14,
        rejected_original_annotation_candidates=len(review['rejected']), basis_review=61,
        owner8_revision=True, unchanged_owners=7, new_base_qualification_pending=True,
        original_results_immutable=True, training_started=False, new_GPU_hours=0, new_Judge=0,
        cumulative_GPU_hours=54.96259710470835, cumulative_Judge=13149,
        independent_confirmation=False, clinical_adjudication=False,
        external_prior_case_aliases='NO_KNOWN_ALIAS_NOT_EXHAUSTIVE_PATIENT_MATCHING',
        limitations=review['limitations'], selfcheck=checks, epoch=time.time())
    write(out/'public/DATA_ADMISSION.json', result)
    write(source/'EXECUTION.json', dict(status='COMPLETE_CPU_FREEZE', command=command, pid=os.getpid(), epoch=time.time()))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
