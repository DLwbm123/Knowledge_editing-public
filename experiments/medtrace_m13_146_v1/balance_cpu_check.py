"""Small numerical and real-reuse checks; never loads the language model."""
import copy
import importlib.util
import os
from pathlib import Path
import sqlite3
import sys
import time
import torch
import balance_protocol as f
import balanced_fisher


def main():
    run=Path(os.environ['RUN_ROOT']);tools=run/'private/tools'
    import balance_runtime as w
    sys.path.insert(0,str(tools))
    import training_math as math,selected_report146 as report
    assert Path(math.__file__).parent==tools and Path(w.w.trajectory.__file__).parent==tools
    assert balanced_fisher.selfcheck()['status']=='PASS'
    assert balanced_fisher.direction_selfcheck()['status']=='PASS'
    oldroot=Path(f.read(run/'private/BALANCE_ADMISSION1.json')['evaluation_parent'])
    spec=importlib.util.spec_from_file_location('prior_math',oldroot/'private/tools/training_math.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rng=torch.Generator().manual_seed(20261017)
    template=w.r.p.tr.TT4(20261017,8,8).state_dict()
    raw={k:.1*torch.randn(v.shape,generator=rng,dtype=torch.float64) for k,v in template.items()}
    state=w.r.mm.canonicalize(raw)[0];g={k:torch.randn(v.shape,generator=rng,dtype=v.dtype) for k,v in state.items()}
    b,a=w.r.mm.factors(state)
    cache=dict(x=torch.randn(9,a.shape[1],generator=rng,dtype=torch.float64),y=torch.randn(9,4,b.shape[0],generator=rng,dtype=torch.float64),token_counts=[4,5])
    scalar=dict(step=0,second_moments=[0.,0.])
    for arm,coef in [('M1',0.),('M13',.001)]+([] if f.DIAGNOSE_ENDPOINT else [('M13B',.001)]):
        left=old.step(w.r,state,g,cache,coef,copy.deepcopy(scalar),.01,arm)
        right=math.step(w.r,state,g,cache,coef,copy.deepcopy(scalar),.01,arm)
        assert left[1:]==right[1:]
        assert all(torch.equal(left[0][k],right[0][k]) for k in state)
    new=math.step(w.r,state,g,cache,1.,copy.deepcopy(scalar),.01,f.CANDIDATE)
    assert new[1]['step']==1 and new[2]['balance']['active'] and new[2]['operator_backwards']==1
    assert abs(w.r.fm.map_norm(state,new[0])-.01)<=.00002
    p=w.w.packet();assert w.w.contract.source_admission(p)['status']=='PASS'
    assert [len(w.w.tasks()[i::2]) for i in range(2)]==[73,73]
    from m13_only_continue import complete
    for arm in ('C','M1','M3','M13'):
        for owner in range(1,147):complete(run,arm,owner)
    records={};scores={}
    with sqlite3.connect('file:'+str(oldroot/'private'/f.PARENT_QUEUE/'queue.sqlite')+'?mode=ro',uri=True) as db:
        for path,status,correct in db.execute('select c.path,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
            d=f.read(path)
            if d['method'] not in ('C',f.REFERENCE,'BASE'):continue
            k=tuple(d[x] for x in ('method','mode','prefix','owner'))+(d['binding']['input']['query_id'],);records[k]=d;scores[k]=correct if status=='FORMAT_VALID' else None
    result,_=report.calculate(f.read(run/'private/ORIGINAL146_TASKS.json'),f.read(run/'private/ORIGINAL_BASE_MASK146.json'),records,scores,methods=('C',f.REFERENCE),single_only=True)
    old=f.read(oldroot/'public'/f.PARENT_REPORT)['contrasts']['C_to_'+f.REFERENCE]
    for m in ('C',f.REFERENCE):assert result['panels'][m]['single_1']==old['panels'][m]['single_1']
    renamed={(f.CANDIDATE,)+k[1:] if k[0]==f.REFERENCE else k:v for k,v in records.items()};renamed_scores={(f.CANDIDATE,)+k[1:] if k[0]==f.REFERENCE else k:v for k,v in scores.items()}
    check,_=report.calculate(f.read(run/'private/ORIGINAL146_TASKS.json'),f.read(run/'private/ORIGINAL_BASE_MASK146.json'),renamed,renamed_scores,methods=('C',f.CANDIDATE),single_only=True)
    assert check['panels'][f.CANDIDATE]==result['panels'][f.REFERENCE]
    identity={k:v for k,v in records.items() if k[0] in (f.REFERENCE,'BASE')}
    identity_scores={k:v for k,v in scores.items() if k in identity}
    for k,v in list(identity.items()):
        if k[0]==f.REFERENCE:
            key=(f.CANDIDATE,)+k[1:];identity[key]=v;identity_scores[key]=scores[k]
    identical,_=report.calculate(f.read(run/'private/ORIGINAL146_TASKS.json'),f.read(run/'private/ORIGINAL_BASE_MASK146.json'),identity,identity_scores,methods=(f.REFERENCE,f.CANDIDATE),single_only=True)
    terms=identical['paired_native_FIT160_contrasts']['single_1']
    assert terms['T2G'][f.CANDIDATE+'_minus_'+f.REFERENCE+'_macro_pp']==0
    for role in ('FIT','T2L'):assert terms[role][f.CANDIDATE+'_preserves_each_'+f.REFERENCE+'_owner']
    assert terms[f.CANDIDATE+'_damage_no_worse_than_'+f.REFERENCE+'_per_owner']
    for t in w.w.tasks():
        for row in w.w.trajectory.source_rows(p,t['order']):assert Path(row['image_path']).is_file()
    receipt=dict(status='PASS',candidate=f.CANDIDATE,cap_edgecases=True,direction_edgecases=True,legacy_M1_M13_step_exact=True,legacy_M13B_step_exact=not f.DIAGNOSE_ENDPOINT,balanced_step_finite_and_matched=True,source_admission=True,original_completed_states=584,single_report_exact=True,new_method_report_supported=True,identity_primary_requires_strict_improvement=True,partitions=[73,73],consumers=f.CONSUMERS,new_generation=2956,epoch=time.time())
    w.c.write(run/'private/BALANCE_CPU_CHECK1.json',receipt);print(receipt)

if __name__=='__main__':main()
