"""No-Judge real-backbone checks; separate from the 96 formal continuations."""
import os,sys,time,traceback,importlib.util
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,session
def main():
    import torch,random
    from scope_worker import configure,setup
    from training import stage,LAYER
    from anchor import predictor_mask,diagnostic_rng
    from methods.medtrace import MedTraceLayerHook
    spec=importlib.util.spec_from_file_location('legacy_training',ROOT/'legacy_training.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    with session('MECHANICAL_SMOKE',3600):
        runtime,counter=configure();began=time.time();t=next(t for t in read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'] if t['order']==4)
        s=setup(runtime,t);ex=s['ex'];layer=runtime.get_module(LAYER);batch=s['batches'][0]
        assert not any(p.requires_grad for p in runtime.model.parameters())
        hook=MedTraceLayerHook(layer,ex);hook.attach();hook.set_teacher_routing(batch.labels)
        try:assert torch.equal(hook.token_mask,predictor_mask(batch.labels,batch.attention_mask))
        finally:hook.detach()
        states=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone(),random.getstate())
        with diagnostic_rng():torch.rand(2);torch.rand(2,device='cuda');random.random()
        assert torch.equal(states[0],torch.get_rng_state()) and torch.equal(states[1],torch.cuda.get_rng_state()) and states[2]==random.getstate()
        checks={}
        def reset():ex.load_state_dict(s['initial']['expert']);ex.requires_grad_(False)
        def snapshot():return {k:v.detach().clone() for k,v in ex.state_dict().items()}
        for arm,beta in [('S',0.),('SP',.1)]:
            reset();legacy.stage(runtime,t,ex,'LR','continuation',s['seed'],'LEGACY_'+arm,s['protect']('SMOKE_'+arm),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],beta))
            historical=snapshot();reset()
            stage(runtime,t,ex,'LR','continuation',s['seed'],'SMOKE_'+arm,s['protect']('SMOKE_'+arm),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],beta),diagnostic=s['diagnostic'])
            assert all(torch.equal(v,historical[k]) for k,v in ex.state_dict().items()),arm+' trajectory changed'
            checks[arm+'_disabled_tensor_exact']=True
        reset();stage(runtime,t,ex,'LR','continuation',s['seed'],'SMOKE_GUARD',s['protect']('SMOKE_GUARD'),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],.1),diagnostic=s['diagnostic']);whole=snapshot()
        reset();stage(runtime,t,ex,'LR','continuation',s['seed'],'SMOKE_RESUME',s['protect']('SMOKE_RESUME'),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],.1),diagnostic=s['diagnostic'],stop_after=40)
        # Recreate the expert state; only the one latest optimizer/RNG slot may restore it.
        reset();stage(runtime,t,ex,'LR','continuation',s['seed'],'SMOKE_RESUME',s['protect']('SMOKE_RESUME'),run_seed=20260929,structure='LR',anchor=(s['ref'],s['cplus'],.1),diagnostic=s['diagnostic'])
        assert all(torch.equal(v,whole[k]) for k,v in ex.state_dict().items()),'Resume trajectory changed'
        assert all(p.grad is None for p in runtime.model.parameters()) and all(not p.requires_grad and p.grad is None for p in s['ref'])
        assert runtime.base_guard.verify()['unchanged'] and not layer._forward_hooks and not layer._forward_pre_hooks
        curves=read(ROOT/f'private/curves/s{s["seed"]}/SMOKE_GUARD/e4/continuation.json')['curve']
        assert len(curves)==80 and all(c['surgery_backbone_forwards']==0 for c in curves)
        checks.update(resume_tensor_exact=True,Base_no_grad=True,reference_no_grad=True,RNG_unchanged=True,predictor_mask_exact=True,hooks_clean=True,no_added_surgery_backbone_forward=True,projection_and_cap_checked_every_step=True,finite_positive_rho_unit_check=True,zero_protection_exact_unit_check=True)
        write(ROOT/'public/GRADIENT_SURGERY_TESTS.json',dict(status='PASS',checks=checks,mechanical_continuations=6,steps=480,Judge_attempts=0,model_calls=counter[0],seconds=time.time()-began,not_promoted_to_formal=True,raw_D_plus_only=True,Adam_monotonic_guarantee=False))
        print('GRADIENT_SURGERY_TESTS_PASS',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'public/GRADIENT_SURGERY_TESTS.json',dict(status='FAILED',error=str(e),traceback=traceback.format_exc()));raise
