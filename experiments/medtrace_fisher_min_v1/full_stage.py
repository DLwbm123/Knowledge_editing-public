"""One independently admitted consumer after the finite mechanism finishes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import torch
import paired_run as p
from mechanism import trial

j,g,r,c,RUN=p.j,p.g,p.r,p.c,p.RUN
NEW='FISHER_MIN'
original_correction=p.correction
p.ARMS=(NEW,);r.mm.ARMS=('C',NEW)


def count(*_):
    c.budget();r.COUNTS['forwards']+=1
    if not r.GENERATING:
        r.COUNTS['non_generation_forwards']+=1
        assert r.COUNTS['non_generation_forwards'] <= dict(paired_train=320000,paired_evaluate=200,paired_route=512)[os.environ['ACTION']]
r.count=count


def correction(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,arm):
    assert arm==NEW
    baseline,old=original_correction(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,'MULTI')
    if old['original']['pure_source_gain']<=0:return baseline,dict(kind='MULTI_NONPOSITIVE_FALLBACK',baseline=old)
    baseline_penalty=float(r.mm.operator_penalty(baseline,cache));records=[]
    with c.diagnostic_scope(hook):
        for fraction in j.FRACTIONS:
            candidate,record=trial(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,NEW,fraction)
            records.append(record)
            if candidate is not None:
                penalty=float(r.mm.operator_penalty(candidate,cache))
                if penalty < baseline_penalty-1e-4*budget:
                    return candidate,dict(kind='FISHER_MIN_ACCEPTED',trials=records,baseline=old,
                        baseline_Fisher_ratio=baseline_penalty/budget,new_Fisher_ratio=penalty/budget)
    return baseline,dict(kind='MULTI_FALLBACK',trials=records,baseline=old)
p.correction=correction


def admission():
    assert c.read(RUN/'private/MECHANISM_COMPLETE.json')['mechanism_gate']
    assert c.read(RUN/'private/EXIT1.json')['exit_code']==0
    assert not list((RUN/'private').glob('FAILURE_*'))
    rows=[]
    for task in r.s.tasks():
        owner=task['order'];probe=r.p.load_state(RUN/'private/probes'/f'{owner}.pt')
        active=r.p.load_state(RUN/'private/active'/f'{owner}.pt')
        assert active['step']==probe['step']==active['scalar']['step']
        assert active['scalar']==probe['scalar_after'] and probe['scalar_before']['step']==probe['step']-1
        assert probe['seed']==active['seed']==task['seed']
        assert probe['source_order']==[v['query_id'] for v in r.s.source(task)]
        assert probe['scope_lock']==c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))
        assert probe['target_lock']==c.digest(c.read(RUN/'private/TARGETS.json'))
        calibration=c.read(Path(os.environ['MODULE_PARENT'])/'private/calibration'/f'{owner}.json')
        assert probe['budget']==.1*calibration['C_operator_penalty']
        assert all(torch.equal(active['state'][k],probe['before'][k]) for k in r.mm.KEYS)
        assert all(torch.equal(active['rng'][k],probe['rng'][k]) for k in ('torch_rng','cuda_rng'))
        assert active['rng']['python_rng']==probe['rng']['python_rng']
        pair=c.read(RUN/'private/paired'/f'{owner}.json')
        baseline=pair['arms']['MULTI']['selected'];changed=pair['arms'][NEW]['selected']
        assert baseline is not None
        def fisher_ratio(arm,selected):
            record=next(v for v in pair['arms'][arm]['trials'] if v['fraction']==selected['fraction'])
            assert record['original_rule_pass']
            return record['attempts'][-1]['Fisher_ratio']
        chosen=baseline if changed is None or fisher_ratio(NEW,changed)>=fisher_ratio('MULTI',baseline)-1e-4 else changed
        state=r.p.load_state(Path(chosen['path']))
        assert state['step']==probe['step'] and state['seed']==probe['seed']
        assert state['fraction']==chosen['fraction'] and set(state['state'])==set(r.mm.KEYS)
        c.write(RUN/'private'/f'START_SELECTION_{owner}.json',dict(chosen=chosen,prior=pair['arms'][NEW]['selected']))
        pair['arms'][NEW]['selected']=chosen
        c.write(RUN/'private/paired'/f'{owner}.json',pair)
        c.restore_rng(dict(active['rng'],cuda_rng=None));assert torch.equal(torch.get_rng_state(),active['rng']['torch_rng'])
        p.f.numpy_restore(probe['numpy_rng'])
        npstate=p.np.random.get_state()
        assert p.np.array_equal(npstate[1],probe['numpy_rng']['keys'])
        assert npstate[2:]==(probe['numpy_rng']['position'],probe['numpy_rng']['has_gauss'],probe['numpy_rng']['cached_gaussian'])
        rows.append(dict(step=probe['step'],full_state=True,seed_source_budget_bound=True,source_RMS=active['scalar']['step']))
    assert sum(v['step'] for v in rows)==341
    c.write(RUN/'private/FULL_RESUME_ADMISSION.json',dict(status='PASS',rows=rows,new_model_calls=0,remaining_proposals=939))


def report_source():
    source=(RUN/'private/tools/frozen_full_report.py').read_text().replace('REFINE',NEW)
    source=source.replace("q.ROOT = RUN / 'private/judge_full_astra_medium'", "q.ROOT = RUN / 'private/judge_fisher_min_astra_medium'")
    source=source.replace("('M13_PARENT', 'MODULE_SEMANTIC.json')", "('M13_PARENT', 'MODULE_SEMANTIC.json'), ('MULTICORE_PARENT', 'MODULE_SEMANTIC.json')")
    source=source.replace("('M13_PARENT', ('M13', 'NATURAL_M13'))", "('M13_PARENT', ('M13', 'NATURAL_M13')), ('MULTICORE_PARENT', ('MULTI', 'NATURAL_MULTI'))")
    source=source.replace("forced = ('C', 'M1', 'M3', 'M13', 'E', 'FISHER_MIN')", "forced = ('C', 'M1', 'M3', 'M13', 'E', 'MULTI', 'FISHER_MIN')")
    source=source.replace("('M13', 'FISHER_MIN'))", "('M13', 'FISHER_MIN'), ('MULTI', 'FISHER_MIN'))")
    compile(source,'frozen_report_adapter','exec')
    assert "('MULTI', 'FISHER_MIN')" in source and 'MULTICORE_PARENT' in source
    return source


def cpu():
    from contextlib import nullcontext
    from unittest.mock import patch
    import full_queue as queue
    assert queue.CONSUMERS==queue.MAX_JUDGE==1158
    assert queue.base.EPOCH=='MEDTRACE_FISHER_MIN_20261014_V1'
    assert 'MULTICORE_PARENT' in queue.base.PARENTS
    source=report_source();assert 'no_owner_worse' in source and 'natural_M3_damage' in source
    for name in ('module_route.py','module_run.py','paired_run.py','full_queue.py'):
        compile((RUN/'private/tools'/name).read_text(),name,'exec')
    assert c.read(RUN/'public/CPU_CHECKS.json')['status']=='PASS'
    baseline={'value':.5};low={'value':.2};high={'value':.6}
    with patch.object(c,'diagnostic_scope',side_effect=lambda _:nullcontext()), \
         patch.object(r.mm,'operator_penalty',side_effect=lambda state,_:state['value']), \
         patch(__name__+'.original_correction',return_value=(baseline,{'original':{'pure_source_gain':1.}})):
        args=(None,None,None,None,None,1,None,None,None,None,1.,NEW)
        with patch(__name__+'.trial',return_value=(low,{'original_rule_pass':True})):
            selected,audit=correction(*args);assert selected is low and audit['kind']=='FISHER_MIN_ACCEPTED'
        with patch(__name__+'.trial',return_value=(high,{'original_rule_pass':True})) as attempts:
            selected,audit=correction(*args);assert selected is baseline and audit['kind']=='MULTI_FALLBACK' and attempts.call_count==4
    c.write(RUN/'public/FULL_CPU_CHECKS.json',dict(status='PASS',single_arm_consumers=1158,
        historical_MULTI_and_M1_M3_M13_E_bound=True,report_failure_gates_preserved=True,
        lower_Fisher_selection_and_baseline_fallback=True,real_resume_admission_pending=True,new_model_calls=0))


def retire():
    folders=('probes','active','diagnostic_states','full_resume','candidates/FISHER_MIN')
    paths=[RUN/'private/OPERATOR_FISHER.pt']+[v for folder in folders for v in (RUN/'private'/folder).glob('*.pt')]
    assert all(v.is_file() and not v.is_symlink() and RUN.resolve() in v.resolve().parents for v in paths)
    receipt=dict(files=len(paths),bytes=sum(v.stat().st_size for v in paths),reason='Single new full arm and all declared consumers complete')
    c.write(RUN/'private/CONSUMED_FULL.json',dict(receipt,paths=[str(v) for v in paths]))
    for v in paths:v.unlink()
    c.write(RUN/'public/EXECUTION_AUDIT.json',dict(status='PASS',lifecycle=receipt,owned_pt_remaining=0))


def controller():
    with (RUN/'private/FULL_ONCE').open('x'):pass
    while not (RUN/'private/EXIT1.json').exists():
        c.budget();assert not list((RUN/'private').glob('FAILURE_*'));time.sleep(20)
    assert c.read(RUN/'private/EXIT1.json')['exit_code']==0
    if not c.read(RUN/'private/MECHANISM_COMPLETE.json')['mechanism_gate']:
        c.write(RUN/'private/FULL_NOT_STARTED.json',dict(reason='NO_REAL_FISHER_INTERIOR_MECHANISM_GAIN'))
        (RUN/'public/FINAL_REPORT_ZH.md').write_text((RUN/'public/MECHANISM_REPORT_ZH.md').read_text()+'\n机制门槛未通过，完整训练/生成/评分未启动；按有限协议结束。\n')
        c.write(RUN/'private/COMPLETE.json',dict(status='COMPLETE',mechanism_only=True,epoch=time.time()))
        return
    assert c.read(RUN/'public/FULL_CPU_CHECKS.json')['status']=='PASS'
    with (RUN/'private/FULL_ADMISSION_ONCE').open('x'):pass
    admission()
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');assert all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    manifest=c.read(RUN/'RUN_MANIFEST.json');manifest['Judge_limit']=17330+1158;c.write(RUN/'RUN_MANIFEST.json',manifest)
    plan=c.read(RUN/'PLAN_CONFIG.json');plan.update(total_NN_backward_limit=141776,maximum_NN_backwards=141776,
        operator_backward_limit=30304,local_solve_limit=45328,maximum_non_generation_forwards=1020128,
        maximum_generations=1158,maximum_new_Judge=1158,full_consumer_implemented=True)
    c.write(RUN/'PLAN_CONFIG.json',plan)
    c.write(RUN/'private/FULL_READY.json',dict(status='PASS',single_arm=True,new_proposals=939))
    c.write(RUN/'public/PROGRESS.json',dict(status='FISHER_MIN_FULL160_TRAINING_RUNNING'))
    j.stage('paired_train',range(3),file='full_stage.py')
    for owner in range(1,9):
        result=c.read(RUN/'private/results'/NEW/f'{owner}.json')
        assert len(result['curve'])==result['scalar_optimizer_state']['step']==160 and result['save_load_exact']
    c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(candidates=8,selection=False))
    c.write(RUN/'public/PROGRESS.json',dict(status='FISHER_MIN_FROZEN_EVALUATION_RUNNING'))
    j.stage('paired_evaluate',range(3),file='full_stage.py')
    j.stage('paired_route',[0],file='full_stage.py')
    outputs=list((RUN/'private/outputs').glob('*/*/*.json'));assert len(outputs)==1158
    counts=[c.read(v) for v in (RUN/'private').glob('COUNTS_*.json')]
    totals={k:sum(v[k] for v in counts) for k in counts[0]}
    assert totals['updates']==1280 and totals['backwards']<=141776 and totals['operator_backwards']<=30304
    assert totals['local_solves']<=45328 and totals['non_generation_forwards']<=1020128 and totals['generation_attempts']<=1158
    c.write(RUN/'private/MODULE_GENERATION_COMPLETE.json',dict(outputs=1158,counts=counts))
    import pipeline
    pipeline.wait([pipeline.launch('full_queue.py','ingest')])
    c.write(RUN/'public/PROGRESS.json',dict(status='FISHER_MIN_GENERATED_SCORING_PENDING'))
    root=RUN/'private/judge_fisher_min_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(20)
    fd,file=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(file).write_text(report_source())
    subprocess.run([os.environ['TRAIN_PYTHON'],file],check=True)
    full=c.read(RUN/'public/FULL_RESULTS.json');panels=full['panels'][NEW]
    lines=['# Fisher内部选择完整验证','',f"完整无缺失：{full['gates']['complete']}；兼顾判定：{full['decision']}。",'',
        '| 方法 | FIT /32 | T2G /32 | 新损伤 /496 |','|---|---:|---:|---:|']
    for method in ('M1','M3','M13','E','MULTI',NEW):
        v=full['panels'][method];lines.append(f"| {method} | {v['FIT']['candidate_correct']} | {v['T2G']['candidate_correct']} | {v['FP32_QUALIFIED']['new_damage']} |")
    lines+=['','只读复用全部精确合法历史对照，保留逐owner门槛与自然R0。全部DEV，T2L两探针/Base正确支持0，保持率NA。']
    (RUN/'public/FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    c.write(RUN/'public/PAIRED_FULL_RESULTS.json',dict(status='COMPLETE',scientific_success=full['gates']['complete'] and all(full['compatibility'].values()),
        complete_no_missing=full['missing_consumers']==0,compatibility=full['compatibility'],resource=full['resource']))
    retire()
    c.write(RUN/'private/COMPLETE.json',dict(status='COMPLETE',epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status='FISHER_MIN_COMPLETE_PUBLICATION_PENDING'))


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='fisher_min_full_cpu':cpu()
        elif action=='fisher_min_full_controller':controller()
        elif action=='paired_train':p.train()
        elif action=='paired_evaluate':r.evaluate(expected_candidates=8)
        elif action=='paired_route':
            import module_route
            module_route.main(expected_candidates=8)
        else:raise ValueError(action)
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f"FAILURE_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        raise
