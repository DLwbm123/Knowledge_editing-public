"""Only two missing factorial arms, with shared frozen training and inference code."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import runtime146 as w
import factorial_protocol as f
from prefix_cache import PrefixCache

r,c,RUN=w.r,w.c,w.RUN

def main():
    f.admit(RUN);w.configure();r.NATIVE_FORWARD_CAP=80000
    original_budget=c.budget
    def budget():
        original_budget();f.project_budget(RUN)
        assert not list((RUN/'private').glob('FAILURE_factorial*.json'))
    c.budget=budget
    action=os.environ['ACTION'];part=int(os.environ['PARTITION'])
    with (RUN/'private'/f'FACTORIAL_ONCE_{action}_{part}').open('x'):pass
    w.require_ready=lambda:f.admit(RUN)
    if action=='factorial_train':
        original=w.trajectory.run;checked=False
        def cached(*args,**kwargs):
            nonlocal checked
            runtime,hook,expert,batches=args[1:5]
            if not checked:
                before=r.old.snapshot(expert)
                with c.diagnostic_scope(hook):
                    rng=c.rng();_,a=w.trajectory.source_gradients(r,runtime,hook,expert,batches,1)
                    grad={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                    c.restore_rng(rng);expert.load_state_dict(before)
                    _,b=w.trajectory.source_gradients(r,runtime,hook,expert,batches,1)
                    assert a==b and all(torch.equal(grad[k],v.grad) for k,v in expert.named_parameters())
                expert.load_state_dict(before);checked=True
                c.write(RUN/'private'/f'FACTORIAL_PARITY_{part}.json',dict(status='PASS',duplicate_source_gradients=True,epoch=time.time()))
            with PrefixCache(runtime,batches) as cache:
                result=original(*args,**kwargs)
                assert cache.hits==315 and cache.misses==5
                c.write(RUN/'private/prefix_cache_receipts'/f'{args[5]}_{args[9]["source_order"][0]}.json',dict(hits=cache.hits,misses=cache.misses))
                return result
        w.trajectory.run=cached
        load=r.load
        def math_load(gpu):
            runtime,bindings,handle=load(gpu);r.old.fp32(runtime);return runtime,bindings,handle
        r.load=math_load
        w.train(arms=('M1','M3'),partitions=2)
        n=len(w.tasks()[part::2]);assert r.COUNTS['updates']==n*320 and r.COUNTS['backwards']==n*640+4 and r.COUNTS['operator_backwards']==n*160
    else:
        assert action=='m13_146_single'
        sys.path.insert(0,str(Path(__file__).parent));import bank146
        bank146.evaluate(292,2,lambda:f.admit(RUN),'FACTORIAL_FROZEN1.json','SCOPE_LOCK.json','GPU_SOURCE_VERSION.json','FACTORIAL_NEW_PLAN1.json')
        assert r.COUNTS['updates']==r.COUNTS['backwards']==0
    c.write(RUN/'private'/f'FACTORIAL_COMPLETE_{action}_{part}.json',dict(status='COMPLETE',counts=r.COUNTS,epoch=time.time()))

if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;c.write(RUN/'private'/f'FAILURE_factorial_{os.environ["ACTION"]}_{os.environ["PARTITION"]}.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:
        r.save_counts();c.write(RUN/'private'/f'FACTORIAL_EXIT_{os.environ["ACTION"]}_{os.environ["PARTITION"]}.json',dict(exit_code=code,epoch=time.time()))
