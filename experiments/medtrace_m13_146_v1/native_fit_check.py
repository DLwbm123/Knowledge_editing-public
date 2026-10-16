"""Five-source CPU recipe and immutable native/FIT coverage checks; no model load."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import contract


def rejected(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Changed original training binding accepted')


def check(packet):
    contract.selfcheck();gate=contract.source_admission(packet)
    assert gate['status']=='PASS' and len(gate['qualified_owners'])==146
    assert gate['qualified_anchor_count']==0 and not gate['training_ready']
    for field in ('seed','native','fit_questions'):
        bad=copy.deepcopy(packet);task=bad['original_tasks'][0]
        if field=='seed':task[field]+=1
        elif field=='native':task[field]['reference']='Changed target'
        else:task[field][0]='Changed FIT question'
        rejected(lambda:contract.source_admission(bad))
    bad=copy.deepcopy(packet);bad['source_weights']=[.25,.25,.5]
    rejected(lambda:contract.source_admission(bad))
    bad=copy.deepcopy(packet)
    bad['native_fit_availability'][bad['original_tasks'][-1]['native']['image_path']]=False
    assert contract.source_admission(bad)['status']=='SOURCE_ADMISSION_PENDING'
    plan=contract.coverage_reservation(packet['original_tasks'],packet['base_mask'])
    assert len(plan)==24575 and not any(row['query'].get('unfilled_source_slot') for row in plan)
    assert not any(row['query'].get('audit_role')=='GFIT' for row in plan)
    return dict(status='PASS',original_edits=146,FIT_questions=584,
        independent_cross_image_training=False,consumers=len(plan),source_binding_changes_rejected=True,
        original_T2L_31_edits_54_probes=True,new_GPU=0,new_VLM=0,new_Judge=0,training_ready=False)


def gradient_check(packet):
    import torch
    import trajectory
    assert not torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__();self.a=torch.nn.Parameter(torch.ones(2,dtype=torch.float64))
    expert=Expert();counts={'backwards':0};calls=[]
    def loss(batch):
        calls.append(batch.labels)
        return (batch.labels+1)*expert.a.square().sum()
    r=SimpleNamespace(COUNTS=counts,mm=SimpleNamespace(KEYS=('a',)),
        set_gradient=lambda e,g:setattr(e.a,'grad',g.clone()))
    runtime=SimpleNamespace(compute_loss=loss);hook=SimpleNamespace(set_teacher_routing=lambda _:None)
    batches=[SimpleNamespace(labels=i) for i in range(5)]
    for step in range(1,161):
        i=1+(step-1)%4
        trajectory.source_gradients(r,runtime,hook,expert,batches,step)
        assert torch.equal(expert.a.grad,torch.full_like(expert.a,2*.5*(1+i+1)))
    assert counts['backwards']==320
    assert calls==[i for step in range(1,161) for i in (0,1+(step-1)%4)]
    trajectory.source_gradients(r,runtime,hook,expert,batches,1,True)
    assert counts['backwards']==325 and torch.equal(expert.a.grad,torch.full_like(expert.a,3.))
    for owner in range(1,147):
        rows=trajectory.source_rows(packet,owner)
        assert [row['audit_role'] for row in rows]==['NATIVE']+['FIT']*4
    return dict(status='PASS',analytic_gradient_exact=True,source_backwards=320,
        individual_gradients_checked=True,source_rows=730,no_GFIT_batches=True,
        new_GPU=0,new_VLM=0,new_Judge=0)


if __name__=='__main__':
    packet=json.loads(Path(os.environ['NATIVE_FIT_PACKET']).read_text())
    result=check(packet)
    if os.environ.get('NATIVE_FIT_GRADIENT_CHECK')=='1':result['gradient']=gradient_check(packet)
    Path(os.environ['NATIVE_FIT_CHECK_OUTPUT']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
