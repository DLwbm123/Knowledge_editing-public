"""CPU-only parity against frozen module_run, using a retained 7168-parameter C state."""
import json
import os
from pathlib import Path
import torch
import module_run as r
import training_math as train
import reporting


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    torch.set_num_threads(2);rng=torch.Generator().manual_seed(20261014)
    states=r.p.load_state(Path(os.environ['M13_CPU_REFERENCE']))['states'];parity=[]
    for name in ('BASE','FINAL'):
        before=r.mm.canonicalize(states[name])[0];assert sum(v.numel() for v in before.values())==7168
        grad={k:torch.randn(v.shape,generator=rng,dtype=v.dtype)*.01 for k,v in before.items()}
        cache=dict(x=torch.randn(4,14336,generator=rng),y=torch.randn(4,2,4096,generator=rng),token_counts=[2,2])
        scalar=dict(step=0,second_moments=[0.,0.]);target=.001
        for arm in ('M1','M13'):
            lam=.2 if arm=='M13' else 0.
            selected,actual,audit=train.step(r,before,grad,cache,lam,scalar,target,arm)
            expert=r.p.expert(before,20261014);params=dict(expert.named_parameters())
            for k,v in params.items():v.grad=grad[k].clone()
            if arm=='M13':(lam*r.mm.operator_penalty(params,cache)).backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
            proposed,expected=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in params.items()},scalar)
            proposed,_=r.match_proposal(before,proposed,target);old=r.mm.canonicalize(proposed)[0]
            assert actual==expected and all(torch.equal(selected[k],old[k]) for k in before)
            assert audit['projection'] is None and audit['finite']['status']=='NOT_APPLICABLE' and scalar['step']==0
            parity.append(dict(state=name,arm=arm,parameters=7168,exact_core_and_RMS_parity=True))
    assert train.coefficient(2.,4.)==.05 and train.coefficient(2.,4.,.05)==.05
    try:train.coefficient(2.,4.,.06)
    except AssertionError:pass
    else:raise AssertionError('Calibration drift not rejected')
    result=dict(status='PASS',real_7168_math_parity=parity,reporting=reporting.selfcheck(),
        frozen_calibration_drift_rejected=True,original_soft_regularization_and_total_RMS=True,
        new_VLM_calls=0,new_Judge=0,new_GPU_leases=0,training_ready=False)
    Path(os.environ['M13_CPU_REPORT']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
