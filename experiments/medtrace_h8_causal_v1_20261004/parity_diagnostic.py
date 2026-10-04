"""One frozen mechanical edit, repeated unchanged one-step updates to locate failure."""
import os
import sys
from pathlib import Path
import json
import time
RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import worker as w
from audit import read,write
def main():
    import torch
    from dataclasses import replace
    from methods.medtrace import MedTraceLayerHook
    from methods.medtrace.selective_write import optimizer_for
    import random
    with w.legacy.lease(4):
        runtime,bindings=w.legacy.load(4)
        metadata=dict(attention_class=type(runtime.model.model.layers[-1].self_attn).__name__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark)
        if os.environ.get('DIAGNOSTIC_DETERMINISM')=='1':
            write(RUN/'private/recovery/H_STEP_PARITY/DETERMINISM_PREFLIGHT.json',metadata)
            torch.use_deterministic_algorithms(True)
        from scripts.medtrace import stage18_cfact as cf
        from m3bench_repro.editors.llava_runtime import seed_everything
        l=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');t=next(t for t in l['tasks'] if t['order']==31);record=w.legacy.record_for(t)
        init=torch.load(RUN/'private/edits/e031/W_init.pt',map_location='cpu',weights_only=True)['expert']
        cfg=dict(code_commit=read(RUN/'private/SOURCE_COMMIT.json')['commit'],runtime_lock=read(RUN/'private/cpu_gate/locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json'))
        role=os.environ.get('DIAGNOSTIC_ROLE','A_NO_H');assert role in ('A_NO_H','B_H1')
        rows=w.H[t['edit_id']] if role=='B_H1' else []
        task=dict(t,U_fit=[w.legacy.local_row(u) for u in t['U_fit']],H_fit=rows)
        teachers=cf.teachers_for(runtime,RUN,cfg,task,record)
        batches=[runtime.build_edit_batch(record)]+[runtime.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
        fit=[1,2,3,4];random.Random(t['seed']).shuffle(fit)
        H_batches=[cf.source_batch(runtime,record,h) for h in rows]
        hi=cf.extra_schedule(task)[0] if rows else None
        extra=(H_batches[hi],rows[hi]) if rows else None
        experts=[w.clone(init,t['seed'],runtime.device).requires_grad_(True) for _ in range(3)]
        results=[]
        for i,e in enumerate(experts):
            seed_everything(t['seed']);opt=optimizer_for(e,runtime.model);hook=MedTraceLayerHook(runtime.get_module(w.LAYER),e);hook.attach()
            before=cf.rng_state()
            try:
                item=cf.update(runtime,hook,e,opt,batches[0],batches[fit[0]],teachers[0],extra) if i<2 else w.logged_update(runtime,hook,e,opt,batches[0],batches[fit[0]],teachers[0],extra,1.,True)
                results.append(dict(kind='ORIGINAL_UNLOGGED' if i<2 else 'LOGGED',before_hash=cf.state_hash(w.clone(init,t['seed'],'cpu')),after_hash=cf.state_hash(e),item=item))
            finally:hook.detach()
        # Reproduce the failing validator's clone-after-update / RNG restoration
        # separately from logging. This is not another scientific arm.
        reference=w.clone(init,t['seed'],runtime.device).requires_grad_(True);ropt=optimizer_for(reference,runtime.model);w.reset_rng(before)
        rh=MedTraceLayerHook(runtime.get_module(w.LAYER),reference);rh.attach()
        try:reference_item=cf.update(runtime,rh,reference,ropt,batches[0],batches[fit[0]],teachers[0],extra)
        finally:rh.detach()
        results.append(dict(kind='REFERENCE_CLONED_AFTER',after_hash=cf.state_hash(reference),item=reference_item));experts.append(reference)
        comparisons=[]
        for i,j in [(0,1),(0,2),(1,2),(2,3)]:
            comparisons.append(dict(i=i,j=j,exact=all(torch.equal(a,b) for a,b in zip(experts[i].parameters(),experts[j].parameters())),max_abs_difference=[float((a-b).abs().max()) for a,b in zip(experts[i].parameters(),experts[j].parameters())],grad_max_abs_difference=[float((a.grad-b.grad).abs().max()) for a,b in zip(experts[i].parameters(),experts[j].parameters())]))
        name='RAW_PARITY_DIAGNOSTIC.json' if role=='A_NO_H' else 'RAW_H_PARITY_DIAGNOSTIC.json'
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):name='RAW_H_CUBLAS_ONLY_DIAGNOSTIC.json'
        if os.environ.get('DIAGNOSTIC_DETERMINISM')=='1':name='RAW_H_DETERMINISTIC_DIAGNOSTIC.json'
        write(RUN/'private/recovery/GPU_STEP_PARITY'/name,dict(results=results,comparisons=comparisons,metadata=metadata,model_training=runtime.model.training,seed=t['seed'],arm=role,scope='mechanical only; no accepted continuation outputs',epoch=time.time()))
        print(json.dumps(dict(comparisons=comparisons,losses=[{k:v['unweighted'] for k,v in r['item']['terms'].items()} for r in results])),flush=True)
if __name__=='__main__':main()
