"""Finite CPU checks of the actual M13 math and fail-closed source admission."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import torch
import contract


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    torch.set_num_threads(2)
    mm=load('m13_math',contract.ROOT/'experiments/medtrace_module_transfer_v1/module_math.py')
    fm=load('categorical_math',contract.ROOT/'experiments/medtrace_fisher_response_v1/fisher_math.py')
    math_checks=mm.selfcheck();fisher_checks=fm.selfcheck();contract.selfcheck()
    packet=json.loads(Path(sys.argv[1]).read_text());baseline=contract.source_admission(packet)
    assert len(baseline['qualified_owners'])==7 and len(baseline['missing_owners'])==139
    changed=copy.deepcopy(packet);changed['original_tasks'][0]['seed']+=1
    assert 1 not in contract.source_admission(changed)['qualified_owners']
    changed=copy.deepcopy(packet);changed['reviews']['owners']['1'][0]['relation_status']='UNKNOWN'
    assert 1 not in contract.source_admission(changed)['qualified_owners']
    changed=copy.deepcopy(packet);changed['image_availability']={k:False for k in changed['image_availability']}
    assert not contract.source_admission(changed)['qualified_owners']
    assert len(contract.coverage_reservation(packet['original_tasks'],packet['base_mask']))==24867
    # The actual composed-map penalty is gauge-invariant; its factor gradients need not be zero.
    rng=torch.Generator().manual_seed(20261014)
    shapes=((1,3,2),(2,4,2),(2,5,2),(2,6,1))
    state=mm.canonicalize({k:torch.randn(s,generator=rng,dtype=torch.float64)*.1 for k,s in zip(mm.KEYS,shapes)})[0]
    state={k:v.requires_grad_() for k,v in state.items()}
    cache=dict(x=torch.randn(5,30,generator=rng,dtype=torch.float64),
        y=torch.randn(5,4,12,generator=rng,dtype=torch.float64),token_counts=[2,3])
    b,a=mm.factors(state);source=(b@a).square().mean()
    source_grad=dict(zip(mm.KEYS,torch.autograd.grad(source,list(state.values()),retain_graph=True)))
    penalty=mm.operator_penalty(state,cache)
    total_grad=dict(zip(mm.KEYS,torch.autograd.grad(source+.2*penalty,list(state.values()))))
    init=dict(step=0,second_moments=[0.,0.])
    _,m1=mm.canonical_rms_step(state,source_grad,init);_,m13=mm.canonical_rms_step(state,total_grad,init)
    assert m1['step']==m13['step']==1 and m1['second_moments']!=m13['second_moments']
    result=dict(status='PASS',canonical_and_operator_checks=math_checks,categorical_Fisher_checks=fisher_checks,
        M13_RMS_uses_combined_gradient=True,missing_unknown_changed_seed_missing_image_rejected=True,
        reserved_consumers=24867,new_VLM_calls=0,new_Judge=0,full_runtime_integration=False)
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
