"""Mechanical canary and cross-device numerical floor on identical saved experts."""
import os,json
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT'])
def crosscheck(runtime,tasks,job,folder):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from scripts.medtrace.run_selective_write import save
    from scripts.medtrace import stage15
    from worker_r2 import load_expert
    import worker_v3 as old
    from freshstart.runtime import write,LAYER
    for t in tasks:
        ex=load_expert(runtime,t,job['seed'],'P',320);ex.requires_grad_(True)
        batch=runtime.build_edit_batch(old.record(t));hook=MedTraceLayerHook(runtime.get_module(LAYER),ex)
        hook.attach()
        try:
            hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch)
            gradients=torch.autograd.grad(loss,tuple(ex.parameters()))
            raw,_,binding=stage15.prepared(runtime,t['native'],old.record(t))
            out=stage15.generate(runtime,raw,binding,hook)
        finally:hook.detach()
        save(folder/f'e{t["order"]:03d}.pt',dict(loss=float(loss.detach()),grads=[g.cpu() for g in gradients],
            tokens=out['raw_token_ids'],eos_in_edit_target=runtime.adapter.tokenizer.eos_token_id in batch.target_token_ids,
            base_frozen=runtime.base_guard.verify()['unchanged']))
    write(folder/'CROSS_GENERATED.json',dict(status='COMPLETE',gpu=os.environ['CUDA_VISIBLE_DEVICES']))
