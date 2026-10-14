"""One gated full consumer; direction failures never fall back to an unaligned step."""
import copy
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import full_stage as baseline
import aligned as a

p,j,g,r,c,RUN=a.p,a.j,a.p.g,a.r,a.c,a.RUN
NEW='ALIGNED_MIN'
p.ARMS=(NEW,);r.mm.ARMS=('C',NEW)
baseline.trial=a.trial


def count(*_):
    c.budget();r.COUNTS['forwards']+=1
    if not r.GENERATING:
        r.COUNTS['non_generation_forwards']+=1
        assert r.COUNTS['non_generation_forwards']<=dict(paired_train=600000,paired_evaluate=200,paired_route=512)[os.environ['ACTION']]
r.count=count


def solve_count():
    r.COUNTS['local_solves']+=1
    assert r.COUNTS['local_solves']<=25000
p.solve_count=solve_count


def correction(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,arm):
    assert arm==NEW
    try:previous,old=baseline.correction(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,'FISHER_MIN')
    finally:p.correction=correction
    if old['baseline']['original']['pure_source_gain']<=0:
        return previous,dict(kind='FISHER_MIN_NONPOSITIVE_FALLBACK',baseline=old)
    radius=r.fm.map_norm(before,pure)
    aligned=a.alignment(before,previous,pure,radius)>=a.FLOOR-1e-7
    prior_penalty=float(r.mm.operator_penalty(previous,cache)) if aligned else float('inf')
    records=[]
    with c.diagnostic_scope(hook):
        for fraction in j.FRACTIONS:
            state,rec=a.trial(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,NEW,fraction)
            records.append(rec)
            if state is not None and float(r.mm.operator_penalty(state,cache))<prior_penalty-1e-4*budget:
                return state,dict(kind='ALIGNED_MIN_ACCEPTED',trials=records,baseline=old,
                    operator_alignment=a.alignment(before,state,pure,radius))
    if aligned:return previous,dict(kind='ALIGNED_FISHER_MIN_FALLBACK',trials=records,baseline=old,
        operator_alignment=a.alignment(before,previous,pure,radius))
    return before,dict(kind='SKIPPED_ALIGNMENT_INFEASIBLE',trials=records,baseline=old,operator_alignment=0.)
p.correction=correction


def admission():
    assert c.read(RUN/'private/MECHANISM_COMPLETE.json')['mechanism_gate']
    assert c.read(RUN/'private/BASELINE_REPAIR_EXIT1.json')['exit_code']==0
    assert c.read(RUN/'private/BASELINE_REPAIR1.json')['status']=='PASS'
    assert not list((RUN/'private').glob('FAILURE_full_*'))
    cache=r.p.load_state(RUN/'private/OPERATOR_FISHER.pt');rows=[]
    for task in r.s.tasks():
        owner=task['order'];probe=r.p.load_state(RUN/'private/probes'/f'{owner}.pt')
        active=r.p.load_state(RUN/'private/active'/f'{owner}.pt')
        assert active['step']==probe['step']==active['scalar']['step']
        assert active['scalar']==probe['scalar_after'] and probe['scalar_before']['step']==probe['step']-1
        assert active['seed']==probe['seed']==task['seed']
        assert probe['source_order']==[v['query_id'] for v in r.s.source(task)]
        assert probe['scope_lock']==c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))
        assert probe['target_lock']==c.digest(c.read(RUN/'private/TARGETS.json'))
        cal=c.read(Path(os.environ['MODULE_PARENT'])/'private/calibration'/f'{owner}.json')
        assert probe['budget']==.1*cal['C_operator_penalty']
        assert all(torch.equal(active['state'][k],probe['before'][k]) for k in r.mm.KEYS)
        assert all(torch.equal(active['rng'][k],probe['rng'][k]) for k in ('torch_rng','cuda_rng'))
        assert active['rng']['python_rng']==probe['rng']['python_rng']
        pair=c.read(RUN/'private/paired'/f'{owner}.json');chosen=copy.deepcopy(pair['arms'][NEW]['selected']);assert chosen
        def load(selected):return r.p.load_state(Path(selected['path']))['state']
        changed=load(chosen);previous=pair['arms']['FISHER_MIN']['selected']
        if previous is not None:
            old=load(previous);radius=r.fm.map_norm(probe['before'],probe['pure'])
            if a.alignment(probe['before'],old,probe['pure'],radius)>=a.FLOOR-1e-7:
                if float(r.mm.operator_penalty(changed,cache))>=float(r.mm.operator_penalty(old,cache))-1e-4*probe['budget']:
                    chosen=copy.deepcopy(previous)
        state=r.p.load_state(Path(chosen['path']))
        assert state['step']==probe['step'] and state['seed']==probe['seed'] and state['fraction']==chosen['fraction']
        ratio=a.alignment(probe['before'],state['state'],probe['pure'],r.fm.map_norm(probe['before'],probe['pure']))
        assert ratio>=a.FLOOR-1e-7
        c.write(RUN/'private'/f'START_SELECTION_{owner}.json',dict(chosen=chosen,prior=pair['arms'][NEW]['selected'],operator_alignment=ratio))
        pair['arms'][NEW]['selected']=chosen;c.write(RUN/'private/paired'/f'{owner}.json',pair)
        c.restore_rng(dict(active['rng'],cuda_rng=None));assert torch.equal(torch.get_rng_state(),active['rng']['torch_rng'])
        p.f.numpy_restore(probe['numpy_rng']);n=np.random.get_state()
        assert np.array_equal(n[1],probe['numpy_rng']['keys']) and n[2:]==(probe['numpy_rng']['position'],probe['numpy_rng']['has_gauss'],probe['numpy_rng']['cached_gaussian'])
        rows.append(dict(step=probe['step'],source_RMS=active['scalar']['step'],full_state_RNG_seed_source_budget_locks=True,operator_alignment=ratio))
    assert sum(v['step'] for v in rows)==341
    c.write(RUN/'private/FULL_RESUME_ADMISSION.json',dict(status='PASS',rows=rows,new_model_calls=0,new_proposals=939))


def report_source():
    source=baseline.report_source().replace('FISHER_MIN',NEW).replace('judge_fisher_min_astra_medium','judge_aligned_astra_medium')
    source=source.replace("('MULTICORE_PARENT', 'MODULE_SEMANTIC.json')", "('MULTICORE_PARENT', 'MODULE_SEMANTIC.json'), ('BASE_PARENT', 'MODULE_SEMANTIC.json')")
    source=source.replace("('MULTICORE_PARENT', ('MULTI', 'NATURAL_MULTI'))", "('MULTICORE_PARENT', ('MULTI', 'NATURAL_MULTI')), ('BASE_PARENT', ('FISHER_MIN', 'NATURAL_FISHER_MIN'))")
    source=source.replace("forced = ('C', 'M1', 'M3', 'M13', 'E', 'MULTI', 'ALIGNED_MIN')", "forced = ('C', 'M1', 'M3', 'M13', 'E', 'MULTI', 'FISHER_MIN', 'ALIGNED_MIN')")
    source=source.replace("('MULTI', 'ALIGNED_MIN'))", "('MULTI', 'ALIGNED_MIN'), ('FISHER_MIN', 'ALIGNED_MIN'))")
    source=source.replace("len(ledger['gpu_sessions']) == 11 + failed_leases", "len(ledger['gpu_sessions']) == 13 + (2 if (RUN/'private/USER_RESUME_ADMISSION1.json').exists() else 0)")
    source=source.replace("resource['zero_work_failed_GPU_leases'] = failed_leases", "resource['zero_work_failed_GPU_leases'] = 1\n    resource['baseline_repair_GPU_leases'] = 2")
    source=source.replace("resource['baseline_repair_GPU_leases'] = 2", "resource['baseline_repair_GPU_leases'] = 2\n    if (RUN/'private/USER_RESUME_ADMISSION1.json').exists():\n        resource['user_interruption'] = q.read(RUN/'private/USER_RESUME_ADMISSION1.json')['public_accounting']")
    assert "('BASE_PARENT', ('FISHER_MIN', 'NATURAL_FISHER_MIN'))" in source
    assert "('MULTI', 'FISHER_MIN', 'ALIGNED_MIN')" not in source
    compile(source,'aligned_full_report','exec');return source


def cpu():
    from contextlib import nullcontext
    from unittest.mock import patch
    import full_queue as queue
    assert queue.CONSUMERS==queue.MAX_JUDGE==1158 and queue.base.EPOCH=='MEDTRACE_ALIGNED_FISHER_20261014_V1'
    assert 'BASE_PARENT' in queue.base.PARENTS
    source=report_source();assert 'no_owner_worse' in source and 'natural_M3_damage' in source
    before={'value':0.};old={'value':.3};new={'value':.2}
    with patch.object(c,'diagnostic_scope',side_effect=lambda _:nullcontext()), \
        patch.object(r.mm,'operator_penalty',side_effect=lambda s,_:s['value']), \
        patch.object(r.fm,'map_norm',return_value=1.), \
        patch.object(baseline,'correction',return_value=(old,{'baseline':{'original':{'pure_source_gain':1.}}})), \
        patch.object(a,'alignment',side_effect=lambda b,s,p,r:.95 if s is new else .2):
        args=(None,None,None,before,None,1,None,None,None,None,1.,NEW)
        with patch.object(a,'trial',return_value=(new,{'original_rule_pass':True})):
            chosen,record=correction(*args);assert chosen is new and record['kind']=='ALIGNED_MIN_ACCEPTED'
        with patch.object(a,'trial',return_value=(None,{})):
            chosen,record=correction(*args);assert chosen is before and record['kind']=='SKIPPED_ALIGNMENT_INFEASIBLE'
        with patch.object(a,'alignment',return_value=.95),patch.object(a,'trial',return_value=(None,{})):
            chosen,record=correction(*args);assert chosen is old and record['kind']=='ALIGNED_FISHER_MIN_FALLBACK'
    c.write(RUN/'public/FULL_CPU_CHECKS.json',dict(status='PASS',consumers=1158,original_finite_rules=True,
        unaligned_fallback_rejected=True,aligned_baseline_preserved=True,report_gates_preserved=True,new_model_calls=0))


def retire():
    folders=('probes','active','diagnostic_states','full_resume','candidates/ALIGNED_MIN')
    paths=[RUN/'private/OPERATOR_FISHER.pt']+[x for folder in folders for x in (RUN/'private'/folder).glob('*.pt')]
    assert all(x.is_file() and not x.is_symlink() and RUN.resolve() in x.resolve().parents for x in paths)
    c.write(RUN/'private/CONSUMED_FULL.json',dict(paths=[str(x) for x in paths],files=len(paths),bytes=sum(x.stat().st_size for x in paths)))
    for x in paths:x.unlink()
    assert not [x for x in (RUN/'private').rglob('*.pt') if not x.is_symlink()]
    c.write(RUN/'public/EXECUTION_AUDIT.json',dict(status='PASS',owned_pt_remaining=0,lifecycle_files=len(paths)))


def controller():
    with (RUN/'private/FULL_ONCE').open('x'):pass
    assert c.read(RUN/'private/BASELINE_REPAIR_EXIT1.json')['exit_code']==0
    assert c.read(RUN/'private/MECHANISM_COMPLETE.json')['mechanism_gate']
    assert c.read(RUN/'public/FULL_CPU_CHECKS.json')['status']=='PASS'
    assert c.read(RUN/'public/FULL_INTEGRATION_CHECKS.json')['status']=='PASS'
    with (RUN/'private/FULL_ADMISSION_ONCE').open('x'):pass
    admission();assert all(s.get('ended_epoch') for s in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    manifest=c.read(RUN/'RUN_MANIFEST.json');manifest['Judge_limit']=17341+1158;c.write(RUN/'RUN_MANIFEST.json',manifest)
    plan=c.read(RUN/'PLAN_CONFIG.json');plan.update(total_NN_backward_limit=200000,maximum_NN_backwards=200000,
        operator_backward_limit=46000,local_solve_limit=65000,maximum_non_generation_forwards=1500000,
        maximum_generations=1158,maximum_new_Judge=1158,full_consumer_implemented=True)
    c.write(RUN/'PLAN_CONFIG.json',plan)
    c.write(RUN/'private/FULL_READY.json',dict(status='PASS',new_proposals=939,alignment_floor=.9))
    c.write(RUN/'public/PROGRESS.json',dict(status='ALIGNED_FULL160_TRAINING_RUNNING'))
    j.stage('paired_train',range(3),file='full_aligned.py')
    finish()


def finish():
    for owner in range(1,9):
        result=c.read(RUN/'private/results'/NEW/f'{owner}.json')
        assert len(result['curve'])==result['scalar_optimizer_state']['step']==160 and result['save_load_exact']
    c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(candidates=8,selection=False))
    j.stage('paired_evaluate',range(3),file='full_aligned.py');j.stage('paired_route',[0],file='full_aligned.py')
    assert len(list((RUN/'private/outputs').glob('*/*/*.json')))==1158
    counts=[c.read(x) for x in (RUN/'private').glob('COUNTS_*.json')];totals={k:sum(x[k] for x in counts) for k in counts[0]}
    assert totals['updates']==1280 and totals['backwards']<=200000 and totals['operator_backwards']<=46000
    assert totals['local_solves']<=65000 and totals['non_generation_forwards']<=1500000 and totals['generation_attempts']<=1158
    recovery=RUN/'private/USER_RESUME_ADMISSION1.json'
    if recovery.exists():
        upper=c.read(recovery)['discarded_compute_upper_bound']
        for key,limit in dict(backwards=200000,operator_backwards=46000,local_solves=65000,non_generation_forwards=1500000).items():
            assert totals[key]+upper[key]<=limit
    c.write(RUN/'private/MODULE_GENERATION_COMPLETE.json',dict(outputs=1158,counts=counts))
    import pipeline
    pipeline.wait([pipeline.launch('full_queue.py','ingest')])
    c.write(RUN/'public/PROGRESS.json',dict(status='ALIGNED_GENERATED_SCORING_PENDING'))
    root=RUN/'private/judge_aligned_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(20)
    import tempfile,subprocess
    fd,path=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd);Path(path).write_text(report_source())
    subprocess.run([os.environ['TRAIN_PYTHON'],path],check=True)
    full=c.read(RUN/'public/FULL_RESULTS.json')
    lines=['# 方向保持完整验证','',f"完整无缺失：{full['gates']['complete']}；兼顾判定：{full['decision']}。",'',
        '| 方法 | FIT /32 | T2G /32 | 损伤 /496 |','|---|---:|---:|---:|']
    for method in ('M1','M3','M13','E','MULTI','FISHER_MIN',NEW):
        x=full['panels'][method];lines.append(f"| {method} | {x['FIT']['candidate_correct']} | {x['T2G']['candidate_correct']} | {x['FP32_QUALIFIED']['new_damage']} |")
    lines+=['','必须同时保留逐编辑M1源/泛化及M3保护，并满足自然路由门槛。全部DEV；T2L两探针/Base正确0，保持率NA。']
    (RUN/'public/FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    c.write(RUN/'public/PAIRED_FULL_RESULTS.json',dict(status='COMPLETE',complete_no_missing=full['missing_consumers']==0,
        compatibility=full['compatibility'],scientific_success=full['gates']['complete'] and all(full['compatibility'].values()),resource=full['resource']))
    retire();c.write(RUN/'private/COMPLETE.json',dict(status='COMPLETE',epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status='ALIGNED_COMPLETE_PUBLICATION_PENDING'))


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action in ('paired_train','paired_evaluate','paired_route'):
            c.write(RUN/'private'/f"FULL_LOADED_SOURCE_{action}_{os.environ.get('PARTITION')}.json",
                dict(c.read(RUN/'private/FULL_GPU_SOURCE_VERSION1.json'),pid=os.getpid(),epoch=time.time()))
        if action=='aligned_full_cpu':cpu()
        elif action=='aligned_full_controller':
            controller();c.write(RUN/'private/FULL_EXIT1.json',dict(exit_code=0,pid=os.getpid(),epoch=time.time()))
        elif action=='paired_train':p.train()
        elif action=='paired_evaluate':r.evaluate(expected_candidates=8)
        elif action=='paired_route':
            import module_route
            module_route.main(expected_candidates=8)
        else:raise ValueError(action)
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f"FAILURE_full_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        if os.environ['ACTION']=='aligned_full_controller':
            c.write(RUN/'private/FULL_EXIT1.json',dict(exit_code=1,pid=os.getpid(),epoch=time.time()))
        raise
