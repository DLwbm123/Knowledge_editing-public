"""Small counterexamples plus actual worker/Adam/checkpoint paths, CPU only."""
import json
import os
from pathlib import Path
import tempfile
import time
import torch
from .guard import InputGuard, compress
from .summary import weighted_rows, collect


def run():
    torch.set_num_threads(2)
    torch.manual_seed(18)
    started = time.monotonic()
    checks = {}
    flat = torch.zeros(1024,96)
    flat[:65,:65] = torch.eye(65)
    kept,info = compress(flat)
    assert kept.norm() < 1e-6 and abs(info['tail_loss']-1) < 1e-10 and abs(info['shrink_loss']-64) < 1e-10
    near = flat.clone(); near[:64,:64] *= 1.001
    kept,info = compress(near)
    assert info['nonzero_rank'] == 64 and abs(info['shrink_loss']-64) < 1e-10
    assert abs(info['before_energy']-info['after_energy']-info['tail_loss']-info['shrink_loss']) < 1e-5
    checks['flat65_zero_and_nearflat65_scale_loss'] = info
    g = InputGuard(torch.zeros(32,96));g.begin(torch.zeros(4,96)); event = g.add(torch.cat((flat,torch.zeros(1,96))))
    end = g.end()
    assert end['history_rank'] == 0 and end['effective_new_history_rank'] == 0 and len(event['compressions']) == 1
    restored = InputGuard.restore(g.state())
    assert restored.basis().shape[1] == 0
    try:InputGuard.restore({k:v for k,v in g.state().items() if k != 'compressions'})
    except ValueError:pass
    else:raise AssertionError('Legacy schema accepted')
    checks['zero_spectrum_no_arbitrary_basis_and_restore'] = True
    h = torch.randn(12,96);h[0] = 0;h[1] *= 1e-20
    valid = torch.ones(12,dtype=torch.bool);valid[-1] = False
    strata = torch.arange(12)%3;u=torch.arange(12,dtype=torch.float64)
    sampled=torch.tensor([0,2,4,6,8,10])
    for arm in ('B','C','N','D','E'):
        rows,info=weighted_rows(h,u,valid,strata,sampled,arm,18)
        assert abs(info['weighted_input_energy']-5) < 1e-5
        if arm!='B':assert len(rows)==10
        zeros,info=weighted_rows(torch.zeros_like(h),u,valid,strata,sampled,arm,18)
        assert zeros.count_nonzero()==0
        zeros,info=weighted_rows(h,u,valid,strata,torch.tensor([0]),arm,18)
        assert zeros.count_nonzero()==0 and info['zero_reference']
    f,_=weighted_rows(h,u,valid,strata,sampled,'C',18)
    zero,_=weighted_rows(h,torch.zeros_like(u),valid,strata,sampled,'D',18)
    assert torch.equal(f,zero)
    a,_=weighted_rows(h,u,valid,strata,sampled,'D',18)
    b,_=weighted_rows(h,u*9,valid,strata,sampled,'D',18)
    assert torch.allclose(a,b,atol=1e-7,rtol=1e-6)
    e,_=weighted_rows(h,u,valid,strata,sampled,'E',18)
    ids=(valid & (h.abs().amax(1)>0)).nonzero().flatten()
    for group in (0,1,2):
        mask=strata[ids]==group
        assert torch.allclose(a[mask].square().sum(1).sort().values,e[mask].square().sum(1).sort().values,atol=2e-7)
    checks['energy_zero_small_nonzero_fallback_shuffle_and_mean_cancellation'] = True
    # Distinguish the per-token squared-gradient sum from gradient-of-sum squared.
    aa=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    def margins(x):return torch.stack((torch.tanh(x.sum(0)).sum(),torch.sin((x*torch.tensor([1.,-1.])).sum())))
    mm=margins(aa); grads=torch.stack([torch.autograd.grad(v,aa,retain_graph=True)[0] for v in mm])
    eps=1e-6; finite=torch.zeros_like(grads)
    for i in range(3):
        for j in range(2):
            delta=torch.zeros_like(aa);delta[i,j]=eps
            finite[:,i,j]=(margins(aa.detach()+delta)-margins(aa.detach()-delta))/(2*eps)
    assert torch.allclose(grads,finite,atol=1e-9,rtol=1e-6)
    assert not torch.allclose(grads.square().sum(2).mean(0),grads.sum(0).square().sum(1)/2)
    checks['nonlinear_per_token_VJP_finite_difference'] = True
    with tempfile.TemporaryDirectory(prefix='run.') as temp:
        root=Path(temp);os.environ['RUN_ROOT']=str(root)
        from .run import install
        install()
        from experiments.fixed_input_guard8_v1 import worker,launch
        from experiments.fixed_input_guard8_v1.cpu_check import Runtime,make_record,exact
        from experiments.fixed_input_guard8_v1.executor import Budget,atomic_json,rng_state
        from .editor import Editor,summary_admission
        assert len([s for s in launch.plan() if s[1]=='learn']) == 40
        assert len([s for s in launch.plan() if s[1]=='diagnostic']) == 15
        calibration=torch.zeros(32,14336)
        reports=[]
        for arm in ('B','C','N','D','E'):
            folder=root/arm
            for owner in (1,2):
                runtime=Runtime(torch.float16)
                result=worker.run_current(runtime,arm,calibration,make_record(owner),['f0','f1','f2','f3'],owner,folder,100+owner)
                state=torch.load(folder/'active.pt',map_location='cpu',weights_only=True)
                assert state['arm']==arm and state['owner']==owner and state['boundary']
                assert state['guard']['sketch'].shape==(1024,14336)
                assert len(runtime.model.layer._forward_hooks)==0 and len(runtime.model._forward_pre_hooks)==0
                atomic_json(folder/'CONSUMED.json',dict(owner=owner,insertion_complete=True,prefix_complete=False))
            fresh=Runtime(torch.float16);worker.load_native(fresh,folder/'native.pt',arm,2)
            assert torch.equal(fresh.model.layer.weight,runtime.model.layer.weight)
            assert result['counters']['attempt_slots']==40
            reports.append(dict(arm=arm,slots=40,qualified=state['guard']['counts'][3].item(),
                                counters=result['counters'],protection_bytes=result['summary']['protection_bytes']))
            print('CPU worker passed',arm,flush=True)
        checks['actual_five_arm_worker_save_reload_200_slots'] = reports
        editor=Editor(Runtime(torch.float16),'D',calibration,Budget(root/'isolated/budget.json'))
        editor.prepare_source(make_record(1),['f0','f1','f2','f3'],1)
        before=editor.state()
        h,energy,valid,strata,metrics=collect(editor.adapter,editor.views[0],True)
        after=editor.state()
        for key in ('native','master','optimizer','guard','rng'):
            assert exact(before[key],after[key]),key
        assert energy.sum()>0 and all(metrics['gradient_nonzero_elements'])
        editor.adapter.close()
        admission=worker.admit_current(Runtime(torch.float16),'D',calibration,make_record(1),['f0','f1','f2','f3'],root/'admit')
        assert admission['status']=='PASS' and admission['summary_admission']['status']=='PASS'
        checks['summary_does_not_mutate_training_state_and_admission'] = True
        from .diagnostic import full_response,token_metrics
        delta=torch.randn(8,96);h=torch.randn(70,96)
        assert torch.allclose(full_response(delta,h,'cpu'),h@delta.T,atol=1e-5)
        assert token_metrics(torch.tensor([[2.,1.]]),torch.tensor([0]),1)['target_margin']==[1.]
        # Exercise all three independent evaluator stages and their retirement barrier.
        import contextlib
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from experiments.fixed_input_guard8_v1.cpu_check import record_type
        from experiments.fixed_input_guard8_v1.diagnostic import save_snapshot
        from .diagnostic import evaluate
        folder=root/'private/fixed8/N'
        runtime=Runtime(torch.float16)
        runtime.adapter=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=7))
        runtime.generation_config=dict(max_new_tokens=12)
        for owner,slot,change in ((6,20,0.),(7,1,.1),(8,2,.2)):
            save_snapshot(runtime.model.layer.weight.detach()+change,folder,owner,slot)
            atomic_json(folder/'steps'/str(owner)/f'{slot}.json',dict(accepted=True))
        row=dict(dataset='SYNTHETIC',question='source 6',reference='6',image_path='/synthetic/input',original_image_path='/synthetic/input')
        atomic_json(root/'private/BENCHMARK146_QUEUE.json',dict(tasks=[{}]*5+[dict(edit_id='6',native=row,fit_questions=['fit'])]))
        ledger=dict(gpu_sessions=[dict(pid=os.getpid(),started_epoch=0,arm='N')])
        @contextlib.contextmanager
        def resources():yield ledger
        c=SimpleNamespace(read=lambda p:json.loads(Path(p).read_text()),lease=lambda _:contextlib.nullcontext(),resources=resources,load=lambda _:(runtime,{}))
        def generate(runtime,bindings,row,destination,phase,budget):
            budget.consume('generation_attempts')
            value=dict(R0=dict(raw_token_ids=[1,2] if phase['owner']==6 else [1]*12,raw_answer='synthetic'))
            atomic_json(destination,value);return value
        with patch.dict(sys.modules,{'m3bench_repro.editors.llava_runtime':SimpleNamespace(EditorRecord=record_type())}), \
             patch.object(launch,'helpers',return_value=(c,root)),patch.object(launch,'install_budget',return_value=Budget), \
             patch.object(launch,'image_path',side_effect=Path),patch.object(launch,'generate',side_effect=generate):
            for owner in (6,7,8):evaluate('N',owner,dict(source6_diagnostic=True,physical_GPU=7,implementation_commit='CPU'))
        output=root/'private/evaluation/source6_diagnostic/N'
        assert not list((folder/'diagnostic_weights').glob('*.pt')) and not list(output.rglob('*.pt'))
        assert c.read(output/'FIRST_CHANGE.json')['first_generation_divergence']==dict(owner=7,slot=1)
        assert len(c.read(output/'7_01/METRICS.json')['fixed_competitor_margin'])==3
        checks['independent_three_stage_diagnostic_and_weight_retirement'] = True
    full=InputGuard(torch.zeros(32,14336))
    assert full.check_storage() < 1_000_000_000
    return dict(status='PASS',scope='CPU_SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE',checks=checks,
                persistent_protection_bytes=full.check_storage(),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    print('RESULT_JSON='+json.dumps(run()))
