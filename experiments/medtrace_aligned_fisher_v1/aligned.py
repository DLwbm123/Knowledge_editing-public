"""Finite same-state alignment test; no semantic evaluation or grid."""
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import mechanism as m

p,j,r,c,RUN=m.p,m.j,m.r,m.c,m.RUN
m.BASE='FISHER_MIN';m.NEW='ALIGNED_MIN'
original_trial=m.original_trial
original_solve=m.original_solve
original_tangent=p.cm.tangent
original_finite=p.f.finite
FLOOR=.9


def alignment(before,candidate,pure,radius):
    def factors(x):return r.mm.factors({k:v.detach().cpu().double() for k,v in x.items()})
    b,a=factors(before);u,v=factors(candidate);x,y=factors(pure)
    return float(p.cm.inner([(u,v),(-b,a)],[(x,y),(-b,a)]))/radius**2


def trial(*args,**kwargs):
    args=list(args);arm=args[-2];before,pure=args[3],args[4]
    assert arm in (m.BASE,m.NEW)
    args[-2]='MULTI';context={}
    radius=r.fm.map_norm(before,pure)
    def tangent(*a,**kw):
        model,build,space=original_tangent(*a,**kw,alignment_reference=pure)
        def candidate(z):
            state=build(z);context['state']=state;return state
        return model,candidate,space
    def solve(**kw):
        direction=kw.pop('alignment');offset=kw.pop('alignment_offset')
        kw['source']=np.vstack((kw['source'],direction))
        kw['required']=np.array([kw['required'],FLOOR+1e-5-offset])
        return original_solve(**kw,fisher_objective=True)
    def finite(*a):
        return original_finite(*a) and alignment(before,context['state'],pure,radius)>=FLOOR-1e-7
    p.cm.solve=(lambda **kw:original_solve(**kw,fisher_objective=True)) if arm==m.BASE else solve
    if arm==m.NEW:p.cm.tangent=tangent;p.f.finite=finite
    try:
        state,record=original_trial(*args,**kwargs)
        if state is not None:
            record['attempts'][-1]['operator_alignment']=alignment(before,state,pure,radius)
        record['alignment_required']=FLOOR if arm==m.NEW else None
        return state,record
    finally:p.cm.solve=original_solve;p.cm.tangent=original_tangent;p.f.finite=original_finite
m.trial=trial


def report():
    rows=[c.read(RUN/'private/paired'/f'{i}.json') for i in range(1,9)]
    parent=Path(os.environ['BASE_PARENT']);improved=first=valid=0
    all_align=[];all_fisher=[]
    for row in rows:
        assert row['save_load_exact'] and row['state_rng_RMS_fixed'] and not row['heldout_accessed']
        a,b=row['arms'][m.BASE],row['arms'][m.NEW]
        old=c.read(parent/'private/paired'/f"{row['owner']}.json")['arms'][m.BASE]
        assert tuple(v['fraction'] for v in a['trials'])==tuple(v['fraction'] for v in b['trials'])==j.FRACTIONS
        assert [v['original_rule_pass'] for v in a['trials']]==[v['original_rule_pass'] for v in old['trials']]
        for index,(x,y) in enumerate(zip(a['trials'],b['trials'])):
            if x['original_rule_pass']:
                j.close([x['attempts'][-1][k] for k in ('source_gain','Fisher_ratio','norm_ratio')],
                    [old['trials'][index]['attempts'][-1][k] for k in ('source_gain','Fisher_ratio','norm_ratio')])
            if y['original_rule_pass']:
                z=y['attempts'][-1];assert z['operator_alignment']>=FLOOR-1e-7
                valid+=1;all_align.append(z['operator_alignment']);all_fisher.append(z['Fisher_ratio'])
                benefit=x['original_rule_pass'] and x['attempts'][-1]['operator_alignment']<FLOOR-1e-4
                improved+=benefit;first+=benefit and index==0
    counts=[c.read(f) for f in (RUN/'private').glob('COUNTS_*.json')]
    totals={k:sum(v[k] for v in counts) for k in counts[0]}
    assert totals['updates']==341 and totals['backwards']<=3743 and totals['operator_backwards']<=256
    assert totals['local_solves']<=256 and totals['non_generation_forwards']<=60128 and totals['generation_attempts']==0
    assert all(c.read(RUN/'private/C_FISHER_PARITY'/f'{i}.json')['status']=='PASS' for i in range(1,9))
    passed=improved>0 and first>0
    result=dict(status='COMPLETE',mechanism_gate=passed,fixed_states=8,paired_cases=32,
        aligned_real_candidates=valid,restored_direction_pairs=improved,first_fraction_direction_restored=first,
        baseline_reconstruction_parity=8,alignment_floor=FLOOR,counts=totals,new_generation=0,new_Judge=0,
        alignment_range=[min(all_align),max(all_align)] if all_align else None,
        Fisher_ratio_range=[min(all_fisher),max(all_fisher)] if all_fisher else None,
        full_training_started=False,independent_confirmation=False)
    c.write(RUN/'public/MECHANISM_RESULTS.json',result)
    c.write(RUN/'private/MECHANISM_COMPLETE.json',dict(result,epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status='ALIGNED_MECHANISM_PASS_FULL_CONSUMER_PENDING' if passed else 'STOP_NO_ALIGNED_MECHANISM_BENEFIT'))
    (RUN/'public/MECHANISM_REPORT_ZH.md').write_text(f'# 源方向保持的有限机制\n\n32配对中{improved}组恢复方向、第一fraction改善{first}组，真实通过{valid}，机制门槛{passed}。\n\n原预算/收益/步幅、来源/秩/seed不变。0生成/0Judge；机制不是性能成功，完整消费者尚待实施/准入。全部DEV。\n')
    if not passed:
        p.retire('Finite aligned mechanism failed')
        c.write(RUN/'private/COMPLETE.json',dict(status='COMPLETE',mechanism_only=True,epoch=time.time()))
    else:c.write(RUN/'private/RETAINED_FOR_FULL.json',dict(reason='Declared pending single-arm consumer'))


def controller():
    with (RUN/'private/ONCE').open('x'):pass
    from guard_admission import require_reference
    require_reference();assert c.read(RUN/'public/CPU_CHECKS.json')['status']=='PASS'
    c.write(RUN/'public/PROGRESS.json',dict(status='ALIGNED_FISHER_REBUILD_RUNNING'))
    j.stage('module_fisher',[0],file='aligned.py')
    c.write(RUN/'public/PROGRESS.json',dict(status='ALIGNED_FIXED_PREFIX_DIAGNOSTICS_RUNNING'))
    j.stage('paired_probe',range(3),file='aligned.py');report()


if __name__=='__main__':
    try:
        action=os.environ['ACTION']
        if action=='module_fisher':r.GPUS=(int(os.environ['GPU']),);r.fisher()
        elif action=='paired_probe':j.probe()
        elif action=='aligned_controller':controller()
        else:raise ValueError(action)
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f"FAILURE_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        raise
