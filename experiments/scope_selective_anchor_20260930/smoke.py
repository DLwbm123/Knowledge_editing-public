"""Separate no-Judge GPU mechanical smoke, before admission of any experiment block."""
import os,sys,time,json,traceback,importlib.util
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,session

def main():
    import torch
    from scope_worker import configure,setup,load_expert
    from training import stage,LAYER
    from anchor import predictor_mask,ScaledExpert,coefficients
    from methods.medtrace import MedTraceLayerHook
    from scripts.medtrace import stage15
    import worker_v3 as old
    spec=importlib.util.spec_from_file_location('legacy_training',ROOT/'legacy_training.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    with session('MECHANICAL_SMOKE',1800):
        runtime,counter=configure();began=time.time();t=next(t for t in read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'] if t['order']==4)
        s=setup(runtime,t);ex=s['ex'];layer=runtime.get_module(LAYER);batch=s['batches'][0]
        assert not any(p.requires_grad for p in runtime.model.parameters())
        captures=[]
        def capture(m,args):captures.append(args[0].detach().clone())
        h=layer.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():runtime.compute_loss(batch)
            hook=MedTraceLayerHook(layer,ex);hook.attach();hook.set_teacher_routing(batch.labels)
            try:
                assert torch.equal(hook.token_mask,predictor_mask(batch.labels,batch.attention_mask))
                with torch.no_grad():runtime.compute_loss(batch)
            finally:hook.detach()
        finally:h.remove()
        assert len(captures)==2 and torch.equal(*captures),'Upstream hidden states differ'
        del captures
        legacy.stage(runtime,t,ex,'LR','continuation',s['seed'],'SMOKE_LEGACY_H',s['protect']('H'),run_seed=20260929,structure='LR')
        legacy_state={k:v.detach().clone() for k,v in ex.state_dict().items()}
        historical,_=load_expert(t,'H')
        # The archived control is on the same frozen effective-residual path.
        z=torch.randn(16,14336,device='cuda',generator=torch.Generator(device='cuda').manual_seed(20260929))
        with torch.no_grad():history_diff=float((historical.residual(z)-ex.residual(z)).abs().max())
        assert torch.allclose(historical.residual(z),ex.residual(z),rtol=2e-5,atol=2e-5),f'Historical H not matched: {history_diff}'
        ex.load_state_dict(s['initial']['expert']);ex.requires_grad_(False)
        stage(runtime,t,ex,'LR','continuation',s['seed'],'SMOKE_NEW_H',s['protect']('H'),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],0.),diagnostic=s['diagnostic'])
        assert all(torch.equal(v,legacy_state[k]) for k,v in ex.state_dict().items()),'New beta0/diagnostics changed real trajectory'
        ex.load_state_dict(s['initial']['expert']);ex.requires_grad_(False)
        stage(runtime,t,ex,'LR','continuation',s['seed'],'SMOKE_SP_01',s['protect']('SP_01'),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],.1),diagnostic=s['diagnostic'])
        curves=read(ROOT/f'private/curves/s{s["seed"]}/SMOKE_SP_01/e4/continuation.json')['curve']
        assert any(c['positive_gradient_norm']>0 for c in curves[1:]) and all(torch.isfinite(p).all() for p in ex.parameters())
        row=next(r for r in t['evaluation'] if r['task']=='T0');raw,_,ib=stage15.prepared(runtime,row,old.record(t))
        # Explicit Base-off token execution, no hook and no zero-valued expert.
        assert not layer._forward_hooks and not layer._forward_pre_hooks
        off=stage15.generate(runtime,raw,ib,None);cached,_=old.base(runtime,row,old.record(t),score=False)
        assert off['raw_token_ids']==cached['raw_token_ids']
        outs=[]
        for expert in [historical,ScaledExpert(historical,1.)]:
            hook=MedTraceLayerHook(layer,expert);hook.attach()
            try:outs.append(stage15.generate(runtime,raw,ib,hook))
            finally:hook.detach()
        assert outs[0]['raw_token_ids']==outs[1]['raw_token_ids']
        assert runtime.base_guard.verify()['unchanged'] and not layer._forward_hooks and not layer._forward_pre_hooks
        write(ROOT/'public/SMOKE_PARITY.json',dict(status='PASS',edit_order=4,seed=20260929,steps=240,continuations=3,methods=['legacy H80','new H80 beta0 + diagnostics','new SP(beta=.1)80'],historical_H_max_residual_error=history_diff,new_vs_legacy_tensor_exact=True,upstream_hidden_exact=True,predictor_mask_exact=True,Base_off_token_exact=True,gamma1_token_exact=True,reference_no_grad=True,diagnostic_trajectory_exact=True,Judge_attempts=0,model_calls=counter[0],seconds=time.time()-began,not_promoted_to_formal_matrix=True))
        print('SMOKE_PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'public/SMOKE_PARITY.json',dict(status='FAILED',error=str(e),traceback=traceback.format_exc()));raise
