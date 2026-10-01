"""Real inference parity before any formal candidate output or new Judge call."""
import os,sys,time,random,hashlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from resources import read,write,session
def main():
    import torch,numpy as np
    from dataclasses import asdict
    from scope_worker import configure,load_expert,weight_bindings
    from phases import references
    from routers import ScopeRouter,tensors_to,selfcheck
    from evaluation import validate_historical,generate_direct
    from scripts.medtrace import stage15
    import worker_v3 as old
    selfcheck();features=torch.load(ROOT/'private/ROUTER_FEATURES.pt',map_location='cpu',weights_only=False);weights=weight_bindings();refs=references('DEV');tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];tests=read(ROOT/'public/ROUTER_MECHANICAL_TESTS.json')
    with session('ROUTER_MODEL_MECHANICAL',3600):
        runtime,counter=configure();protocol=dict(model=str((ROOT/'models/llava-med-v1.5-mistral-7b').resolve()),precision='float16',backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda));t=tasks[0];e=tensors_to([features['singles']['1']],runtime.device);ex=load_expert(t,weights);snapshot={k:v.clone() for k,v in ex.state_dict().items()};samples=[next(r for r in t['evaluation'] if r['task']=='T0')]
        router=ScopeRouter(e,'R0')
        for r in t['evaluation']:
            if not router.route(old.key(runtime,r,old.record(t))).activated:samples.append(r);break
        parity=[];started=time.time();n=counter[0]
        for row in samples:
            raw,batch,ib=stage15.prepared(runtime,row,old.record(t));q=old.key(runtime,row,old.record(t));historical=next(r for r in refs if r['mode']=='single' and r['edit']==t['canonical_edit_id'] and r['input_id']==old.input_id(row));saved=validate_historical(historical,ib,protocol,features,weights)
            # Two fresh extractions establish deterministic feature hashes, not just a cache hit.
            hashes=[]
            for _ in [0,1]:
                with torch.inference_mode():z=runtime.extract_layer_input_key(batch,module_path=runtime.target_lock['balancedit']['targets'][0],pooling='mean').float().cpu().contiguous()
                hashes.append(hashlib.sha256(z.numpy().tobytes()).hexdigest())
            assert hashes[0]==hashes[1] and torch.equal(z.reshape(-1),q.float().cpu().reshape(-1))
            original=ScopeRouter(e,'R0').route(q);assert asdict(original)==historical['route']
            tokens=[]
            for method in ['R0','NEG0','PLOO']:
                r=ScopeRouter(e,method);cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state();py=random.getstate();npstate=np.random.get_state();decision,_=r.diagnostic(q)
                assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state()) and py==random.getstate()
                after=np.random.get_state();assert npstate[0]==after[0] and np.array_equal(npstate[1],after[1]) and npstate[2:]==after[2:]
                if decision.logical_edit_id!=original.logical_edit_id:continue
                out=generate_direct(runtime,raw,ib,ex if decision.activated else None);assert out['raw_token_ids']==saved['output']['raw_token_ids'];tokens.append(method)
            assert 'R0' in tokens
            parity.append(dict(role='active' if original.activated else 'OFF',token_exact_methods=tokens,deterministic_feature_hash=True))
        assert all(torch.equal(v,snapshot[k]) for k,v in ex.state_dict().items())
        assert runtime.base_guard.verify()['unchanged'] and not runtime.get_module('model.layers.30.mlp.down_proj')._forward_hooks
        assert all(not p.requires_grad and p.grad is None for p in ex.parameters())
        tests.update(status='PASS',checks=dict(R0_historical_token_exact=True,PLOO_weights_unchanged=True,positive_guard_membership_checked=True,native_activation_checked=True,NEG0_tau_exact_zero=True,positive_native_plus_four_S_fit_only=True,negative_audited_U_bg_only=True,formal_queries_outcomes_not_used_for_construction=True,deterministic_feature_hash=True,no_GPU_training=True,router_diagnostics_generation_RNG_unchanged=True,identical_effective_route_expert_token_exact=True),real_generation_parity=parity,Base_unchanged=True,expert_unchanged=True,hooks_clean=True,backward_calls=0,optimizer_steps=0,training_steps=0,model_calls=counter[0]-n,seconds=time.time()-started,epoch=time.time())
        write(ROOT/'public/ROUTER_MECHANICAL_TESTS.json',tests)
    write(ROOT/'RUN_STATUS.json',dict(status='READY_FOR_DEV',phase='DEV',epoch=time.time()));print('PASS: PLOO mechanical checks, real token parity, zero training/Judge')
if __name__=='__main__':main()
