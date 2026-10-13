"""CPU-only coverage/queue/report integration; synthetic answers never become results."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import torch
import full146_contract as contract
import full146_run as run
import full146_report as report

ROOT = Path(os.environ['RUN_ROOT'])
UP = Path(os.environ['UPSTREAM_ROOT'])


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    q = report.q
    tasks = q.read(ROOT/'private/ORIGINAL146_TASKS.json')
    dummy = [dict(query_id=f'UNQUALIFIED_GFIT146_{i}_{j}', owner=i, source_group=f'UNQUALIFIED_{i}_{j}')
             for i in range(1, 147) for j in range(2)]
    mask = q.read(ROOT/'private/ORIGINAL_BASE_MASK146.json')
    plan = contract.evaluation_plan(tasks, q.read(ROOT/'private/EVAL_QUERIES146.json'), dummy, mask)
    q.write(ROOT/'private/EVAL_PLAN_TEMPLATE146.json', plan)
    for task in tasks[:7]:
        cold = run.r.q.d.start_state(task)
        prior = run.r.p.load_state(UP/'private/candidates/C'/f"{task['order']}.pt")['states']['BASE']
        assert all(torch.equal(cold[k], prior[k]) for k in cold)
    for task in tasks:
        router = run.r.p.load_state(Path(q.read(ROOT/'private/ROUTER_BINDINGS146.json')[str(task['order'])]))
        assert len(router['entries']) == 1 and router['entries'][0]['logical_edit_id'] == task['edit_id']
    reviews, anchors = {}, []
    for task in tasks:
        owner = task['order']; reviews[str(owner)] = []
        for j in range(2):
            key = f'CPU_FIXTURE_{owner}_{j}'; path = f'/CPU_FIXTURE_ONLY/{key}'
            anchors.append(dict(owner=owner, candidate_id=key, source_group=key, image_path=path, role='GFIT'))
            reviews[str(owner)].append(dict(candidate_id=key, admitted=True, license=['CPU_FIXTURE_ONLY'],
                source_evidence=[dict(path='/CPU_FIXTURE_ONLY')], image_binding=dict(path=path),
                task_binding=contract.task_binding(task), relation_status='PASS_SOURCE_EVIDENCE',
                image_review='PASS_READABLE_SOURCE_FIGURE', basis_conflict_review='PASS_NO_CONTRADICTORY_TARGET',
                role_check='PASS_KNOWN_IMAGE_AND_CASE_EXCLUSION'))
    assert contract.source_review(tasks, tasks, anchors, reviews, set())['status'] == 'PASS'
    for key, value in [('relation_status', 'UNKNOWN'), ('basis_conflict_review', 'UNKNOWN'), ('license', [])]:
        bad = copy.deepcopy(reviews); bad['146'][1][key] = value
        assert contract.source_review(tasks, tasks, anchors, bad, set())['status'] != 'PASS'
    bad = copy.deepcopy(anchors); bad[-1]['image_path'] = bad[-2]['image_path']
    assert contract.source_review(tasks, tasks, bad, reviews, set())['status'] != 'PASS'
    assert contract.source_review(tasks, tasks, anchors[:-1], reviews, set())['status'] != 'PASS'
    assert contract.source_review(tasks, tasks, anchors, reviews, {anchors[-1]['source_group']})['status'] != 'PASS'
    shards = [{contract.consumer_id(v) for i, v in enumerate(plan) if i % 3 == part} for part in range(3)]
    assert sum(map(len, shards)) == len(set.union(*shards)) == 17373
    fixture = Path(tempfile.mkdtemp(prefix='CPU_FIXTURE_ONLY_', dir=ROOT/'private'))
    try:
        q.write(fixture/'private/SCOPE_LOCK.json', dict(CPU_FIXTURE_ONLY=True))
        for name, value in [('EVAL_PLAN146.json', plan), ('ORIGINAL146_TASKS.json', tasks),
                            ('ORIGINAL_BASE_MASK146.json', mask), ('LAUNCH_ENV.json', q.read(ROOT/'private/LAUNCH_ENV.json')),
                            ('INHERITED_COST.json', dict(gpu_seconds_used=0., Judge_attempts=0)),
                            ('MODULE_GENERATION_COMPLETE.json', dict(CPU_FIXTURE_ONLY=True))]:
            q.write(fixture/'private'/name, value)
        q.write(fixture/'RESOURCE_LEDGER.json', dict(gpu_seconds_used=0., Judge_attempts=0, gpu_sessions=[]))
        scientific = q.digest(dict(CPU_FIXTURE_ONLY=True))
        for row in plan:
            query = dict(row['query'], question=row['query'].get('question', 'CPU_FIXTURE_Q'),
                reference=row['query'].get('reference', 'CPU_FIXTURE_GOLD'), image_sha256='CPU_FIXTURE_IMAGE')
            binding = {k: query[k] for k in ('question', 'reference', 'image_sha256')}
            binding.update(image_path='/CPU_FIXTURE_ONLY', prompt_ids=[1], attention_mask=[1],
                           runtime=dict(CPU_FIXTURE_ONLY=True), generation=dict(CPU_FIXTURE_ONLY=True))
            path = fixture/'private/outputs'/f"{row['method']}_{row['mode']}_p{row['prefix']}"/str(row['owner'])/(q.digest(query['query_id'])+'.json')
            q.write(path, dict(method=row['method'], mode=row['mode'], prefix=row['prefix'], eval_owner=row['owner'],
                role='CPU_FIXTURE_ONLY', binding=dict(input=query, judge_input=binding),
                R0=dict(raw_answer='CPU_FIXTURE_'+row['method'], raw_token_ids=[1]), lock=scientific,
                consumer_id=contract.consumer_id(row), active_target=False))
        report.queue.RUN = report.RUN = fixture
        q.ROOT = fixture/'private/judge_full146_astra_medium'
        db = q.connect(); report.queue.ingest(db)
        assert db.execute('SELECT count(*) FROM consumer').fetchone()[0] == 17373
        for key, binding in db.execute('SELECT key,binding FROM payload').fetchall():
            binding = json.loads(binding); answer = binding['output']['raw_answer']
            score = int(answer.endswith('REFINE')) if not answer.endswith('BASE') else int(mask.get(binding['query_id'], False))
            db.execute("UPDATE payload SET status='FORMAT_VALID',correct=? WHERE key=?", (score, key))
        db.commit()
        q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json', dict(CPU_FIXTURE_ONLY=True))
        report.main()
        assert contract.success(q.read(fixture/'public/FULL146_RESULTS.json')['gates'])
        fit = {v['query_id'] for v in contract.fit_rows(tasks[-1])}
        for key, binding in db.execute('SELECT key,binding FROM payload').fetchall():
            b = json.loads(binding)
            if b['query_id'] in fit and not b['output']['raw_answer'].endswith('BASE'):
                db.execute('UPDATE payload SET correct=? WHERE key=?', (int(b['output']['raw_answer'].endswith('M1')), key))
        db.commit(); report.main()
        result = q.read(fixture/'public/FULL146_RESULTS.json')
        assert not result['gates']['FIT_per_owner'] and not contract.success(result['gates'])
        key = db.execute('SELECT key FROM payload LIMIT 1').fetchone()[0]
        db.execute("UPDATE payload SET status='MISSING',correct=NULL WHERE key=?", (key,));db.commit()
        report.main(); assert not q.read(fixture/'public/FULL146_RESULTS.json')['gates']['complete']
        db.close()
    finally:
        assert fixture.parent == ROOT/'private' and fixture.name.startswith('CPU_FIXTURE_ONLY_')
        shutil.rmtree(fixture)
    stat = (ROOT/'private/inherited/OPERATOR_FISHER.pt').stat()
    basis = run.r.q.split()[0]
    assert len(basis) == 61
    runtime = next(iter(q.read(ROOT/'private/EVAL_BINDINGS.json').values()))['runtime']
    q.write(ROOT/'private/FULL146_INHERITED_FISHER_BINDING.json', dict(status='PASS', questions=61,
        BASIS_binding=q.digest(basis), runtime_binding=q.digest(runtime), device=stat.st_dev, inode=stat.st_ino))
    checks = dict(status='PASS', scope='CPU_PREPARATION_NOT_SOURCE_OR_GPU_ADMISSION',
        original146_consumers=17373, original_R0_records=146, frozen_T2L_probes=54, frozen_T2L_edit_units=31,
        full_queue_and_report_integration=True, synthetic_fixture_removed=True,
        missing_and_single_owner_regression_rejected=True, source_unknown_missing_duplicate_forbidden_rejected=True,
        old_C_seven_seeded_cold_tensor_parity=True, three_disjoint_exhaustive_shards=True,
        contract=contract.selfcheck(), report=report.selfcheck(), payload=report.queue.selfcheck(),
        new_model_calls=0, new_Judge=0, epoch=time.time())
    q.write(ROOT/'public/FULL146_CPU_CHECKS.json', checks)
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
