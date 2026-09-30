"""Finish beta=0 S parity against the audited historical S; no scoring."""
import os,sys,time
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,session

def main():
    import torch
    from scope_worker import configure,setup,load_expert
    from training import stage
    with session('MECHANICAL_S_PARITY',900):
        runtime,counter=configure();start=time.time();t=next(t for t in read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'] if t['order']==4);s=setup(runtime,t);ex=s['ex']
        stage(runtime,t,ex,'LR','continuation',s['seed'],'SMOKE_NEW_S',s['protect']('S'),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],0.),diagnostic=s['diagnostic'])
        historical,_=load_expert(t,'S');assert all(torch.equal(v,historical.state_dict()[k]) for k,v in ex.state_dict().items()),'beta0 S failed archived final tensor parity'
        assert runtime.base_guard.verify()['unchanged']
        report=read(ROOT/'public/SMOKE_PARITY.json');report.update(steps=report['steps']+80,continuations=report['continuations']+1,beta0_S_historical_tensor_exact=True,model_calls=report['model_calls']+counter[0],seconds=report['seconds']+time.time()-start);report['methods'].append('new S80 beta0 + diagnostics');write(ROOT/'public/SMOKE_PARITY.json',report)
        print('S_PARITY_PASS',flush=True)
if __name__=='__main__':main()
