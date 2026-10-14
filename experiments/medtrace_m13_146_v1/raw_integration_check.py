"""CPU C/M3 multi-step parity and Adam save/load continuity on actual 7168 cores."""
import copy
import json
import os
from pathlib import Path
import torch
import module_run as r
import training_math as train


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    torch.set_num_threads(2);rng=torch.Generator().manual_seed(20261014)
    states=r.p.load_state(Path(os.environ['M13_CPU_REFERENCE']))['states']
    frozen=torch.nn.Linear(1,1);frozen.requires_grad_(False);cases=[]
    for name in ('BASE','FINAL'):
        initial=states[name];assert sum(v.numel() for v in initial.values())==7168
        assert name!='BASE' or torch.count_nonzero(initial['G1'])==0
        cache=dict(x=torch.randn(4,14336,generator=rng),y=torch.randn(4,2,4096,generator=rng),token_counts=[2,2])
        for arm in ('C','M3'):
            expert=r.p.expert(initial,20261014);old=r.p.expert(initial,20261014)
            opt=r.p.optimizer(expert,frozen);reference=r.p.optimizer(old,frozen)
            for step in range(1,4):
                grad={k:torch.randn(v.shape,generator=rng,dtype=v.dtype)*.01 for k,v in initial.items()}
                lam,target=(0.,None) if arm=='C' else (.2,.001)
                selected,audit=train.raw_step(r,expert,opt,grad,cache,lam,target,arm)
                before=r.old.snapshot(old);params=dict(old.named_parameters())
                reference.zero_grad(set_to_none=True)
                for k,v in params.items():v.grad=grad[k].clone()
                if arm=='M3':(lam*r.mm.operator_penalty(params,cache)).backward()
                torch.nn.utils.clip_grad_norm_(old.parameters(),1.)
                reference.step();r.q.d.apply_direction(reference,{id(v):before[k] for k,v in params.items()},'RAW')
                expected=r.old.snapshot(old)
                if arm=='M3':expected,_=r.match_proposal(before,expected,target)
                old.load_state_dict(expected)
                assert all(torch.equal(selected[k],expected[k]) for k in initial)
                assert same(opt.state_dict(),reference.state_dict())
                assert all(int(v['step'])==step for v in opt.state.values())
                assert audit['projection'] is None and audit['source_RMS_steps']==0
                if step==2:
                    restored=r.p.expert(selected,20261014);restored_opt=r.p.optimizer(restored,frozen)
                    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
                    assert same(restored_opt.state_dict(),opt.state_dict())
                    expert,opt=restored,restored_opt
            cases.append(dict(state=name,arm=arm,parameters=7168,steps=3,
                exact_core_and_Adam_parity=True,save_load_continuity=True))
    for prior in (float('inf'),float('nan'),0.,-1.):
        try:train.coefficient(2.,4.,prior)
        except AssertionError:pass
        else:raise AssertionError('Nonfinite or nonpositive calibration accepted')
    result=dict(status='PASS',real_7168_raw_math_parity=cases,original_Adam_RAW_groups=True,
        C_zero_function_initialization_preserved=True,M3_combined_gradient_moments=True,
        invalid_frozen_coefficient_rejected=True,new_GPU_leases=0,new_VLM_calls=0,new_Judge=0,
        training_ready=False,full_training_dispatch_implemented=False)
    Path(os.environ['M13_CPU_REPORT']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
