"""One finite Fisher-interior mechanism test; completed MULTI stays read-only."""
import copy
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import paired_run as p

j,g,r,c,RUN=p.j,p.g,p.r,p.c,p.RUN
BASE,NEW='MULTI','FISHER_MIN'
original_trial=p.trial
original_solve=p.cm.solve


def trial(*args,**kwargs):
    args=list(args);arm=args[-2]
    assert arm in (BASE,NEW)
    args[-2]=BASE
    if arm==NEW:
        p.cm.solve=lambda **kw:original_solve(**kw,fisher_objective=True)
    try:return original_trial(*args,**kwargs)
    finally:p.cm.solve=original_solve


def count(*_):
    c.budget();r.COUNTS['forwards']+=1
    if not r.GENERATING:
        r.COUNTS['non_generation_forwards']+=1
        assert r.COUNTS['non_generation_forwards'] <= (128 if os.environ['ACTION']=='module_fisher' else 20000)
r.count=count


def diagnostics(runtime,hook,expert,task,step,before,pure,scalar_before,scalar_after,
                sources,protection,refs,cache,budget,losses,old_decision,before_rng,numpy_before,consumer=None):
    owner=task['order'];cpu_cache=p.cache_cpu(cache)
    payload=dict(before=r.s.previous.weights(before),pure=r.s.previous.weights(pure),
        scalar_before=scalar_before,scalar_after=scalar_after,rng=before_rng,old_decision=old_decision,
        numpy_rng=dict(kind=numpy_before[0],keys=numpy_before[1].tolist(),position=numpy_before[2],
                       has_gauss=numpy_before[3],cached_gaussian=numpy_before[4]),seed=task['seed'],step=step,
        source_order=[x['query_id'] for x in r.s.source(task)],budget=budget,source_losses=losses,
        scope_lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json')),target_lock=c.digest(c.read(RUN/'private/TARGETS.json')))
    path=RUN/'private/probes'/f'{owner}.pt';assert not path.exists();c.save(path,payload)
    saved=r.p.load_state(path)
    assert all(torch.equal(saved[k][key],payload[k][key]) for k in ('before','pure') for key in r.mm.KEYS)
    assert saved['scalar_after']==scalar_after and saved['scalar_before']==scalar_before
    assert saved['seed']==task['seed'] and saved['source_order']==payload['source_order']
    assert all(torch.equal(saved['rng'][k],before_rng[k]) for k in ('torch_rng','cuda_rng'))
    assert saved['rng']['python_rng']==before_rng['python_rng']
    output={};numpy_state=np.random.get_state()
    try:
        with c.diagnostic_scope(hook):
            for arm in (BASE,NEW):
                rows=[];selected=None
                for fraction in j.FRACTIONS:
                    state,record=trial(runtime,hook,expert,before,pure,step,sources,losses,cache,cpu_cache,budget,arm,fraction)
                    if state is not None:
                        file=RUN/'private/diagnostic_states'/f'{arm}_{owner}_{fraction}.pt'
                        assert not file.exists()
                        c.save(file,dict(state=r.s.previous.weights(state),seed=task['seed'],step=step,fraction=fraction))
                        loaded=r.p.load_state(file);assert all(torch.equal(loaded['state'][k],state[k].cpu()) for k in r.mm.KEYS)
                        if selected is None:selected=dict(path=str(file),fraction=fraction,kind=arm+'_WITNESS')
                    rows.append(record)
                output[arm]=dict(trials=rows,selected=selected)
    finally:
        np.random.set_state(numpy_state);expert.load_state_dict(before);expert.zero_grad(set_to_none=True)
    c.write(RUN/'private/paired'/f'{owner}.json',dict(owner=owner,step=step,arms=output,
        save_load_exact=True,state_rng_RMS_fixed=True,heldout_accessed=False))
j.diagnostics=diagnostics


def report():
    rows=[c.read(RUN/'private/paired'/f'{i}.json') for i in range(1,9)]
    prior=Path(os.environ['MULTICORE_PARENT'])
    improved=valid_pairs=0;first_improved=0;parity=0
    for row in rows:
        assert row['save_load_exact'] and row['state_rng_RMS_fixed'] and not row['heldout_accessed']
        old=c.read(prior/'private/paired'/f"{row['owner']}.json")['arms']['MULTI']
        a,b=row['arms'][BASE],row['arms'][NEW]
        assert tuple(v['fraction'] for v in a['trials'])==tuple(v['fraction'] for v in b['trials'])==j.FRACTIONS
        assert [v['original_rule_pass'] for v in a['trials']]==[v['original_rule_pass'] for v in old['trials']]
        for index,(x,y) in enumerate(zip(a['trials'],b['trials'])):
            if x['original_rule_pass']:
                j.close([x['attempts'][-1][k] for k in ('source_gain','Fisher_ratio','norm_ratio')],
                    [old['trials'][index]['attempts'][-1][k] for k in ('source_gain','Fisher_ratio','norm_ratio')],atol=1e-5)
            if x['original_rule_pass'] and y['original_rule_pass']:
                valid_pairs+=1
                lower=y['attempts'][-1]['Fisher_ratio'] < x['attempts'][-1]['Fisher_ratio']-1e-4
                improved+=lower
                if index==0:first_improved+=lower
        parity+=1
    counts=[c.read(f) for f in (RUN/'private').glob('COUNTS_*.json')]
    totals={k:sum(v[k] for v in counts) for k in counts[0]}
    assert totals['updates']==341 and totals['backwards']<=3743
    assert totals['operator_backwards']<=256 and totals['local_solves']<=256
    assert totals['non_generation_forwards']<=60128 and totals['generation_attempts']==0
    assert all(c.read(RUN/'private/C_FISHER_PARITY'/f'{i}.json')['status']=='PASS' for i in range(1,9))
    passed=improved>0 and first_improved>0 and parity==8
    result=dict(status='COMPLETE',mechanism_gate=passed,fixed_states=8,paired_cases=32,
        valid_both_pairs=valid_pairs,real_lower_Fisher_pairs=improved,first_fraction_lower_Fisher=first_improved,
        MULTI_reconstruction_parity=parity,original_real_rules_unchanged=True,counts=totals,
        generation=0,new_Judge=0,full_training_started=False,independent_confirmation=False)
    c.write(RUN/'public/MECHANISM_RESULTS.json',result)
    c.write(RUN/'private/MECHANISM_COMPLETE.json',dict(result,epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status='MECHANISM_PASS_FULL_IMPLEMENTATION_PENDING' if passed else 'STOP_NO_FISHER_INTERIOR_BENEFIT',full_training_started=False))
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    c.write(RUN/'public/RESOURCE_SUMMARY.json',dict(new_GPU_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
        cumulative_GPU_hours=ledger['gpu_seconds_used']/3600,cumulative_Judge=ledger['Judge_attempts']))
    (RUN/'public/MECHANISM_REPORT_ZH.md').write_text(f'# Fisher内部选择的固定机制验证\n\n真实低Fisher配对见证{improved}/32，第一fraction改善{first_improved}/8；机制门槛{passed}。\n\n八冻结状态、原MULTI重建、C标定、完整保存加载与预算检查通过。0生成/0新Judge；不能据此宣称FIT/T2G或语义保护改善。完整训练尚未启动，必须先实现并核验唯一后续消费者。全部DEV。\n')
    if not passed:p.retire('Finite Fisher-interior mechanism gate failed')
    else:c.write(RUN/'private/RETAINED_FOR_FULL.json',dict(reason='Declared pending single-arm full consumer',owned_packages=1+8+8+sum(1 for _ in (RUN/'private/diagnostic_states').glob('*.pt'))))


def controller():
    with (RUN/'private/ONCE').open('x'):pass
    from guard_admission import require_reference
    require_reference();assert c.read(RUN/'public/CPU_CHECKS.json')['status']=='PASS'
    c.write(RUN/'public/PROGRESS.json',dict(status='FISHER_REBUILD_RUNNING',full_training_started=False))
    j.stage('module_fisher',[0],file='mechanism.py')
    c.write(RUN/'public/PROGRESS.json',dict(status='FROZEN_PREFIX_AND_FISHER_INTERIOR_DIAGNOSTICS_RUNNING',full_training_started=False))
    j.stage('paired_probe',range(3),file='mechanism.py')
    report()


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='module_fisher':r.GPUS=(int(os.environ['GPU']),);r.fisher()
        elif action=='paired_probe':j.probe()
        elif action=='fisher_min_controller':controller()
        else:raise ValueError(action)
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f"FAILURE_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        raise
