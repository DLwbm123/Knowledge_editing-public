"""One source-balanced Fisher arm, with shared frozen training and inference code."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import runtime146 as w
import balance_protocol as f
from prefix_cache import PrefixCache

r,c,RUN=w.r,w.c,w.RUN

def main():
    f.admit(RUN);w.configure();r.NATIVE_FORWARD_CAP=50000
    original_budget=c.budget
    def budget():
        original_budget();f.project_budget(RUN)
        assert not list((RUN/'private').glob('FAILURE_balance*.json'))
    c.budget=budget
    action=os.environ['ACTION'];part=int(os.environ['PARTITION'])
    with (RUN/'private'/f'BALANCE_ONCE_{action}_{part}').open('x'):pass
    w.require_ready=lambda:f.admit(RUN)
    if action=='balance_train':
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
                expert.load_state_dict(before);c.restore_rng(rng);checked=True
                c.write(RUN/'private'/f'BALANCE_PARITY_{part}.json',dict(status='PASS',duplicate_source_gradients=True,epoch=time.time()))
            if f.DIAGNOSE_ENDPOINT:
                # Fixed endpoint diagnostic on the immutable M13 state; no heldout inputs.
                before=r.old.snapshot(expert);rng=c.rng()
                owner=next(t['order'] for t in w.tasks() if w.trajectory.source_rows(w.packet(),t['order'])[0]['query_id']==args[9]['source_order'][0])
                state=r.p.load_state(w.point({'order':owner},'M13'))['states']['FINAL']
                expert.load_state_dict(state)
                with c.diagnostic_scope(hook):
                    _,losses=w.trajectory.source_gradients(r,runtime,hook,expert,batches,160)
                    source={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                    params={k:v.detach().clone().requires_grad_(True) for k,v in expert.named_parameters()}
                    penalty=r.mm.operator_penalty(params,args[6])
                    fish=dict(zip(params,torch.autograd.grad(args[7]*penalty,tuple(params.values()))))
                    r.COUNTS['operator_backwards']+=1
                    from balanced_fisher import cap_gradient
                    _,audit=cap_gradient(source,fish,r.mm.GROUPS)
                c.write(RUN/'private/baseline_gradient_diagnostics'/f'{owner}.json',dict(audit,source_NLL=losses,penalty=float(penalty),step=160))
                expert.load_state_dict(before);c.restore_rng(rng)
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
        w.train(arms=(f.CANDIDATE,),partitions=2)
        n=len(w.tasks()[part::2]);assert r.COUNTS['updates']==n*160 and r.COUNTS['backwards']==n*(322 if f.DIAGNOSE_ENDPOINT else 320)+4 and r.COUNTS['operator_backwards']==n*(161 if f.DIAGNOSE_ENDPOINT else 160)
    else:
        assert action=='m13_146_single'
        sys.path.insert(0,str(Path(__file__).parent));import bank146
        bank146.evaluate(146,2,lambda:f.admit(RUN),'BALANCE_FROZEN1.json','SCOPE_LOCK.json','GPU_SOURCE_VERSION.json','BALANCE_NEW_PLAN1.json')
        assert r.COUNTS['updates']==r.COUNTS['backwards']==0
    c.write(RUN/'private'/f'BALANCE_COMPLETE_{action}_{part}.json',dict(status='COMPLETE',counts=r.COUNTS,epoch=time.time()))

if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;c.write(RUN/'private'/f'FAILURE_balance_{os.environ["ACTION"]}_{os.environ["PARTITION"]}.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:
        r.save_counts();c.write(RUN/'private'/f'BALANCE_EXIT_{os.environ["ACTION"]}_{os.environ["PARTITION"]}.json',dict(exit_code=code,epoch=time.time()))
