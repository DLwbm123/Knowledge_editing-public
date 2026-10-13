"""One source/protection audit; no model, Judge, old-file writes or GPU launch."""
import hashlib
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def binding(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def normalized(text):
    return ' '.join(text.casefold().split())


def collisions(basis, evaluation):
    inputs = {(r['image_sha256'], normalized(r['question'])) for r in evaluation}
    groups = {r['source_group'] for r in evaluation}
    texts = {normalized(r['question']) for r in evaluation}
    return dict(full_input=sum((r['image_sha256'], normalized(r['question'])) in inputs for r in basis),
                source_group=len({r['source_group'] for r in basis} & groups),
                text_only=sum(normalized(r['question']) in texts for r in basis))


def audit(original, historical):
    original, historical = Path(original), Path(historical)
    pause = read(original/'public/PAUSE_PREFIX100_RECEIPT.json')
    assert pause['status'] == 'PAUSED_USER_REQUEST' and not pause['full_run_complete']
    ledger_cost = read(original/'RESOURCE_LEDGER.json')
    assert all(x.get('ended_epoch') for x in ledger_cost['gpu_sessions'])
    env = read(historical/'private/LAUNCH_ENV.json')
    parent = Path(env['ANSWER_PARENT'])
    qualified = read(parent/'private/SEMANTIC_QUALIFICATION.json')['rows']
    assert len(qualified) == 192 and all(x['status'] == 'FORMAT_VALID' for x in qualified)
    order = {r['query_id']: i for i, r in enumerate(read(parent/'private/ANSWER_ROWS.json'))}
    basis = [dict(x['row'], semantic_correct=True, response_path=x['path']) for x in qualified
             if x['row']['audit_role'] == 'BASIS' and x['correct'] == 1]
    basis.sort(key=lambda r: order[r['query_id']])
    assert len(basis) == len({r['query_id'] for r in basis}) == 61
    assert len({r['source_group'] for r in basis}) == 31
    sources, response_lengths, runtimes = {}, [], []
    for row in basis:
        assert row['role'] == 'REPLAY_FIT' and row['source_role'] == 'train'
        source = Path(row['source_file'])
        assert row['dataset'] == 'SLAKE' and source.name == 'train.json'
        if source not in sources:
            sources[source] = {str(r['qid']): r for r in read(source)}
        raw = sources[source][str(row['source_qid'])]
        assert raw['q_lang'] == 'en'
        assert raw['question'] == row['question'] and raw['answer'] == row['reference']
        assert 'SLAKE:' + raw['img_name'].split('/')[0] == row['source_group']
        assert Path(row['image_path']).is_file()
        output = read(row['response_path'])
        assert output['unedited_Base'] and not output['audit']['at_generation_cap']
        assert output['audit']['generated_roundtrip_tokens_exact']
        assert output['binding']['input'] == {k: v for k, v in row.items()
                                              if k not in ('semantic_correct', 'response_path')}
        jb = output['binding']['judge_input']
        assert all(jb[k] == row[k] for k in ('question', 'reference', 'image_sha256', 'image_path'))
        tokens = output['R0']['raw_token_ids']
        assert 0 < len(tokens) <= 128 and all(type(x) is int for x in tokens)
        response_lengths.append(len(tokens)); runtimes.append(jb['runtime'])
    tasks = read(original/'private/BENCHMARK146_QUEUE.json')['tasks']
    ledger = read(original/'private/EVAL_LEDGER.json')
    evaluation = ledger['queries']
    by_role = {}
    for role in ('T0', 'T1G', 'T2G', 'T1L', 'T2L'):
        ids = {q for t in tasks for e in t['events'] if e['task'] == role
               for q in e['all_probe_query_ids']}
        by_role[role] = collisions(basis, [evaluation[q] for q in ids])
    combined = collisions(basis, list(evaluation.values()))
    failures = []
    if combined['full_input']: failures.append('EVALUATION_FULL_INPUT_OVERLAP')
    if combined['source_group']: failures.append('EVALUATION_SOURCE_GROUP_OVERLAP')
    source_rows = [dict(t['native'], question=q) for t in tasks
                   for q in [t['native']['question']] + t['fit_questions']]
    source_overlap = collisions(basis, source_rows)
    if source_overlap['source_group']: failures.append('EDIT_SOURCE_GROUP_OVERLAP')
    norms, native_norms, total_zero = [], [], 0
    for owner in range(1, 101):
        for step in range(1, 161):
            event = read(original/'private/steps/bank_R0'/str(owner)/(str(step)+'.json'))
            assert event['owner'] == owner and event['step'] == step and event['mode'] == 'bank_R0'
            proposal = event['proposal_norm']
            assert math.isfinite(proposal) and 0 < proposal <= .050001
            norms.append(proposal); native_norms.append(event['actual_native_norm'])
            total_zero += event['changed_elements'] == 0
    assert len(norms) == 16000 and total_zero == 0
    targets = dict(shape=[100, 160], proposal_norms=norms, native_step_norms=native_norms)
    panel_counts = {str(n): len({q for t in tasks[:n] for e in t['events']
                                for q in e['all_probe_query_ids']}) for n in (1, 50, 100)}
    insertion_T2G = sum(len({q for e in t['events'] if e['task'] == 'T2G'
                            for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]})
                        for t in tasks[:100])
    insertion_outputs = 100 + insertion_T2G
    new_evaluation_outputs = sum(panel_counts.values()) + insertion_outputs
    held = [x['row'] for x in qualified if x['row']['audit_role'] == 'HELDOUT']
    assert len(held) == 96 and not {r['source_group'] for r in held} & {r['source_group'] for r in basis}
    result = dict(status='PASS' if not failures else 'BLOCKED', failures=failures,
                  basis_queries=61, basis_source_groups=31, source_QA_verified=61,
                  historical_Base_responses_verified=61, source='SLAKE train annotations',
                  current146_collision=combined, collision_by_role=by_role,
                  source_collision=source_overlap, historical_HELD_group_overlap=0,
                  basis_binding=binding(basis), norm_binding=binding(targets),
                  reference_steps=16000, reference_zero_writes=total_zero,
                  panel_output_counts=panel_counts, insertion_T2G_outputs=insertion_T2G,
                  insertion_outputs=insertion_outputs, evaluation_outputs=new_evaluation_outputs,
                  generation_limit=new_evaluation_outputs + 64 + 2,
                  proposal_min=min(norms), proposal_max=max(norms),
                  proposal_path_length=sum(norms), native_path_length=sum(native_norms),
                  response_token_min=min(response_lengths), response_token_max=max(response_lengths),
                  native_FP16_teacher_admission='NOT_RUN', model_calls=0, Judge_calls=0,
                  new_GPU_updates=0, evaluation_answers_used=False,
                  no_collision_based_filter_or_repair=True,
                  patient_independence='UNKNOWN', independent_confirmation=False,
                  epoch=time.time())
    private = dict(basis=basis, targets=targets, historical_Base_runtimes=runtimes,
                   original_run=str(original), historical_run=str(historical))
    return result, private


def selfcheck():
    a = dict(image_sha256='a', question=' Which organ? ', source_group='g1')
    b = dict(image_sha256='b', question='which ORGAN?', source_group='g2')
    assert collisions([a], [b]) == dict(full_input=0, source_group=0, text_only=1)
    assert collisions([a], [dict(b, image_sha256='a')])['full_input'] == 1
    assert collisions([a], [dict(b, source_group='g1')])['source_group'] == 1
    assert binding({'a': 1, 'b': 2}) == binding({'b': 2, 'a': 1})
    return dict(status='PASS', normalized_full_input=True, group_isolation=True,
                text_only_reported_separately=True, canonical_binding=True,
                model_execution_checked=False)
