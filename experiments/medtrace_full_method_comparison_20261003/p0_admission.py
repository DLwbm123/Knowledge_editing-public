"""CPU-only admission; source candidates are never automatic H truth labels."""
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def source_identity(dataset, path):
    found = re.findall(r'xmlab\d+|synpic\d+', str(path), re.I)
    if len(set(x.casefold() for x in found)) != 1:
        raise ValueError('Ambiguous or missing original source image identity')
    return dataset, found[0].casefold()


def exact_candidate(native, row, roles, allowed):
    return (row['dataset'] == native['dataset']
            and source_identity(native['dataset'], native['original_image_path'])[1] != row['image_id']
            and normalized(native['question']) == normalized(row['question'])
            and normalized(native['reference']) != normalized(row['answer'])
            and row['image_exists'] and Path(row['image_path']).is_file()
            and allowed(roles)
            and (row['dataset'] != 'VQA-RAD' or row['source_QA_evaluation'] == 'evaluated'))


def selfcheck():
    allowed = lambda roles: roles == ['BASE_QUERY_INVENTORY']
    native = dict(dataset='SLAKE', original_image_path='xmlab1/source.jpg', question=' Is it CT? ', reference='yes')
    row = dict(dataset='SLAKE', image_id='xmlab2', question='is it ct?', answer='no', image_exists=True, image_path=__file__)
    assert exact_candidate(native, row, ['BASE_QUERY_INVENTORY'], allowed)
    for key, value in [('image_id', 'xmlab1'), ('answer', 'yes'), ('question', 'is it ct')]:
        assert not exact_candidate(native, dict(row, **{key: value}), ['BASE_QUERY_INVENTORY'], allowed)
    assert not exact_candidate(native, row, ['PROTECTED_SPLIT'], allowed)
    assert normalized(' ＣＴ?\n') == 'ct?'
    assert source_identity('VQA-RAD', '/images/synpic99.jpg') == ('VQA-RAD', 'synpic99')


def write_new(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    started = time.time()
    cpu_started = time.process_time()
    root = Path(os.environ['RUN_ROOT'])
    policy = json.loads((root / 'P0_CONFIG.json').read_text())
    manifest = json.loads((root / 'RUN_MANIFEST.json').read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    argv = Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert not any(x in argv.casefold() for x in ('wangbomin', 'knowledge_editing', 'medtrace'))
    selfcheck()
    legacy = root / 'private/legacy_stage17'
    audit_root = Path(os.environ['SOURCE_AUDIT_ROOT'])
    role_root = Path(os.environ['SOURCE_ROLE_ROOT'])
    helper = Path(os.environ['REVIEW_RULES_FILE'])
    spec = importlib.util.spec_from_file_location('review_rules', helper)
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    rules.selfcheck()
    inputs = [legacy / 'COHORT_AND_SUPPORT_LEDGER.json', legacy / 'ROLE_BOUNDARY_JOIN.json',
              audit_root / 'private/SOURCE_QA_ROLE_AUDIT.json', role_root / 'private/IDENTITY_ROLE_LEDGER.json']
    input_bytes = sum(p.stat().st_size for p in inputs)
    assert input_bytes <= policy['input_bytes_cap']
    ledger, join, rows, roles = [json.loads(p.read_text()) for p in inputs]
    # Required frozen scientific identity; not a transfer-integrity hash scan.
    value = {k: v for k, v in ledger.items() if k != 'freeze_id'}
    identity = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert identity == ledger['freeze_id'] == policy['original_cohort_freeze_id']
    main_ids = ledger['main_T0']
    task_map = {t['edit_id']: t for t in ledger['tasks']}
    assert len(main_ids) == len(set(main_ids)) == policy['N']
    tasks = [task_map[edit] for edit in main_ids]
    assert all(t['native']['base_correct'] is False for t in tasks)
    assert len(rows) == len(roles) == 7167
    role_map = {(r['dataset'], r['image_id'], r['qid']): r['roles'] for r in roles}
    assert len(role_map) == len(roles)
    blocked = {source_identity(q['dataset'], q['original_image_path']) for q in ledger['queries'].values()}
    blocked |= {source_identity(dataset, image) for dataset, image in join['reserved_source_groups']}
    pool = [r for r in rows if rules.allowed(role_map[(r['dataset'], r['image_id'], r['qid'])])
            and (r['dataset'], r['image_id']) not in blocked
            and r['image_exists'] and Path(r['image_path']).is_file()
            and (r['dataset'] != 'VQA-RAD' or r['source_QA_evaluation'] == 'evaluated')]
    by_question = collections.defaultdict(list)
    for row in pool:
        by_question[(row['dataset'], normalized(row['question']))].append(row)
    proposals, gaps = [], []
    for position, task in enumerate(tasks, 1):
        assert time.time() < manifest['deadline_epoch'] and not (root / 'STOP').exists()
        native = task['native']
        candidates = [r for r in by_question[(native['dataset'], normalized(native['question']))]
                      if exact_candidate(native, r, role_map[(r['dataset'], r['image_id'], r['qid'])], rules.allowed)]
        candidates.sort(key=lambda r: (r['dataset'], r['image_id'], str(r['qid'])))
        selected = candidates[:policy['maximum_exact_candidates_per_edit']]
        for index, row in enumerate(selected, 1):
            proposals.append(dict(review_id=f'H{position:03d}-{index}', edit_id=task['edit_id'],
                                  position=position, native=native, candidate=row,
                                  source_roles=role_map[(row['dataset'], row['image_id'], row['qid'])],
                                  relation_label='UNKNOWN', admitted_H_fit=False,
                                  reviewer=None, same_proposition_evidence=None, out_of_scope_evidence=None,
                                  patient_independence='UNKNOWN', data_role='REVIEW_ONLY'))
        gaps.append(dict(position=position, edit_id=task['edit_id'],
                         existing_H_support=len(task['H_fit']), original_U_supported=task['roles']['U_fit'],
                         exact_source_candidates=len(candidates), selected_for_review=len(selected),
                         admission='UNSUPPORTED_PENDING_RELATION_EVIDENCE',
                         missing_reason=task['missing_reason']))
    mapped = lambda path: Path(path.replace('/remote-home/wangbomin/', '/data/bmw/', 1))
    main_queries = {q for t in tasks for event in t['events'] for q in [event['edit_query_id'], *event['all_probe_query_ids']]}
    image_paths = {str(mapped(ledger['queries'][q]['original_image_path'])) for q in main_queries}
    missing_images = [p for p in sorted(image_paths) if not Path(p).is_file()]
    write_new(root, 'private/H_RELATION_REVIEW_PACKET.json', proposals)
    write_new(root, 'private/H_SUPPORT_GAPS.json', gaps)
    write_new(root, 'private/INPUT_RECEIPTS.json', dict(inputs=[dict(path=str(p), bytes=p.stat().st_size) for p in inputs],
                                                     copied_legacy_files=[dict(name=p.name, bytes=p.stat().st_size) for p in sorted(legacy.iterdir())],
                                                     missing_required_images=missing_images))
    assert not any(p['admitted_H_fit'] for p in proposals)
    assert all(g['existing_H_support'] == 0 for g in gaps), 'New H evidence requires a separate reviewed admission join'
    public = dict(status='P0_COMPLETE_BLOCKED_REQUIRED_H', N=len(tasks), original_freeze_binding='PASS',
                  original_U_supported=sum(g['original_U_supported'] for g in gaps), existing_verified_H=0,
                  source_QA_rows=len(rows), review_role_qualified_source_QA=len(pool),
                  exact_question_different_answer_candidate_edits=sum(g['exact_source_candidates'] > 0 for g in gaps),
                  review_proposals=len(proposals), verified_new_H=0,
                  edits_without_exact_source_candidate=sum(g['exact_source_candidates'] == 0 for g in gaps),
                  candidate_status='UNKNOWN; source-answer differences do not automatically establish edit scope',
                  required_main_queries=len(main_queries), required_main_images=len(image_paths), missing_required_main_images=len(missing_images),
                  native_dataset_counts=dict(collections.Counter(t['native']['dataset'] for t in tasks)),
                  scientific_GPU_admission=False, GPU_hours=0, new_Judge=0, new_generation=0,
                  no_smaller_cohort_substitution=True, independence='EXPOSED_ORIGINAL_PANEL; patient independence UNKNOWN')
    write_new(root, 'public/P0_ADMISSION.json', public)
    write_new(root, 'public/P0_EXECUTION_AUDIT.json', dict(status='COMPLETE', CPU_seconds=time.process_time()-cpu_started,
                                                       task_wall_seconds=time.time()-started, input_bytes=input_bytes,
                                                       cohort_identity_check='PASS', role_guard_selfcheck='PASS',
                                                       exact_match_and_no_label_promotion_selfcheck='PASS',
                                                       GPU_hours=0, new_Judge=0, first_clock_preserved=True))
    write_new(root, 'P0_PROCESS_RECEIPT.json', dict(pid=os.getpid(), argv=argv, starting_epoch=started, CUDA_VISIBLE_DEVICES=''))
    (root / 'RUN_STATUS.json').write_text(json.dumps(dict(status='BLOCKED_REQUIRED_H_SUPPORT',
                                                        completed_phase='P0', next_phase='P1_NOT_ADMITTED',
                                                        epoch=time.time(), GPU_admitted=False), indent=2)+'\n')
    print(json.dumps(public))


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        selfcheck()
        print('PASS: exact question/source/answer and role rejection checks')
    else:
        try:
            main()
        except Exception as error:
            import traceback
            root = Path(os.environ['RUN_ROOT'])
            write_new(root, 'P0_FAILURE.json', dict(error=str(error), traceback=traceback.format_exc(), epoch=time.time()))
            (root / 'RUN_STATUS.json').write_text(json.dumps(dict(status='P0_FAILED_PRESERVED', GPU_admitted=False))+'\n')
            raise
