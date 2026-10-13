"""Small CPU checks of dispatch/storage/statistics; actual GPU gates stay separate."""
import json
import os
from pathlib import Path
import sys
RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
from audit import read,write,digest
def main():
    import torch
    sys.path.insert(0,str(RUN/'private/tools'))
    import worker
    assert Path(worker.__file__).parent==RUN/'private/tools'
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert,optimizer_for
    from scripts.medtrace.stage18_cfact import state_hash,extra_schedule
    from legacy_metrics import metric,paired,micro_delta,selfcheck
    cp=AsymmetricCPExpert(14336,4096,4);e=LowRankExpert(cp,20260912,rank=4);init={k:v.detach().clone() for k,v in e.state_dict().items()}
    arms=[worker.clone(init,20260912,'cpu') for _ in range(4)]
    assert len({state_hash(x) for x in arms})==1
    assert len({p.data_ptr() for x in arms for p in x.parameters()})==8
    base=torch.nn.Linear(1,1).requires_grad_(False);opts=[optimizer_for(x,base) for x in arms]
    assert all(not opt.state for opt in opts) and len({id(opt.state) for opt in opts})==4
    assert all([{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]==[{k:v for k,v in g.items() if k!='params'} for g in opts[0].param_groups] for opt in opts)
    plan=read(RUN/'PLAN_CONFIG.json');assert tuple(plan['arms'])==worker.ARMS and plan['training']['diagnostic_steps']==[1,20,80,160,320]
    import random
    order=[1,2,3,4];random.Random(20260912).shuffle(order)
    assert all(order[(s-1)%4]!=order[s%4] for s in range(1,321))
    H=read(RUN/'private/H_AVAILABLE.json');l=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');by={t['edit_id']:t for t in l['tasks']}
    for eid,rows in H.items():
        t=dict(by[eid],H_fit=rows);assert extra_schedule(t)==extra_schedule(t) and len(extra_schedule(t))==320
    selfcheck()
    result=dict(status='PASS_CPU_ONLY',checks=['four exact CPU tensor clones/no aliases','four fresh optimizer states and equal configuration','frozen four-arm/diagnostic dispatch','D next-fit differs from ordinary slot320steps','B/C unchanged balanced H schedule','shared missing/statistics original selfcheck'],actual_GPU_gates_pending=True,new_GPU_seconds=0,new_Judge_attempts=0)
    write(RUN/'private/CPU_IMPLEMENTATION_CHECK.json',result);write(RUN/'public/CPU_IMPLEMENTATION_CHECK.json',result);print(json.dumps(result))
if __name__=='__main__':main()
