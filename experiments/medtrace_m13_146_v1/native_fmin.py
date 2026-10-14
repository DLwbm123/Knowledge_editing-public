"""First two native/FIT FISHER_MIN consumers, using the original correction."""
import copy
import os
import time
import traceback
import numpy as np
import torch
import full_stage as original
import runtime146 as worker
import contract
import trajectory

p,g,r,c,RUN=original.p,original.g,original.r,original.c,original.RUN


def configure():
    worker.configure();g.SOURCE_WEIGHTS=contract.SOURCE_WEIGHTS
    counted=r.count;forward_start=r.COUNTS['non_generation_forwards'];solve_start=r.COUNTS['local_solves']
    def count(*args):
        counted(*args);assert r.COUNTS['non_generation_forwards']-forward_start<=100000
    r.count=count
    def solve_count():
        r.COUNTS['local_solves']+=1
        assert r.COUNTS['local_solves']-solve_start<=7680
    p.solve_count=solve_count


def validate(saved,binding):
    assert saved['binding']==binding and 0<saved['step']<=160
    assert saved['scalar']['step']==saved['step'] and len(saved['scalar']['second_moments'])==2
    assert all(np.isfinite(v) and v>=0 for v in saved['scalar']['second_moments'])
    assert len(saved['curve'])==saved['step'] and [v['step'] for v in saved['curve']]==list(range(1,saved['step']+1))
    assert set(saved['state'])==set(r.mm.KEYS) and sum(v.numel() for v in saved['state'].values())==7168
    assert all(torch.isfinite(v).all() for v in saved['state'].values())
    assert saved['counts']['updates']-saved.get('counts_start',{}).get('updates',0)==saved['step'] and saved['counts']['backwards']-saved.get('counts_start',{}).get('backwards',0)>=2*saved['step']
    assert set(saved['rng'])>=set(('torch_rng','python_rng','cuda_rng')) and len(saved['numpy_rng'])==5


def main(full=False,loaded=None):
    gate,scope_file,source_file=worker.stage_admission('FISHER_MIN',full);scope=c.read(RUN/'private'/scope_file)
    assert gate['status']=='PASS'
    if not full:assert gate['owners']==[1,2]
    assert gate['packet_binding']==c.digest(worker.packet()) and gate['scope_binding']==c.digest(scope)
    assert c.read(RUN/'private/NATIVE_FMIN_CPU_CHECKS1.json')['status']=='PASS'
    assert c.read(RUN/'private/NATIVE_FOLLOWUP_EXIT1.json')['exit_code']==0
    assert c.read(RUN/'public/FISHER_ADMISSION.json')['status']=='PASS'
    assert contract.source_admission(worker.packet())['status']=='PASS'
    owner=int(os.environ['OWNER']);gpu=int(os.environ['GPU']);assert owner in gate['owners'] and gpu in (0,1)
    if not full:assert owner==gpu+1 and owner in (1,2)
    with (RUN/'private'/f'NATIVE_FMIN_ONCE_{owner}').open('x'):pass
    configure();task=worker.tasks()[owner-1];rows=trajectory.source_rows(worker.packet(),owner)
    assert not worker.point(task,'FISHER_MIN').exists()
    calibration=c.read(RUN/'private/calibration146'/f'{owner}.json')
    assert calibration['source_binding']==c.digest(contract.original.source_binding(rows))
    budget=.1*calibration['C_operator_penalty'];assert budget>0
    reference=c.read(RUN/'private/results/C'/f'{owner}.json')
    from m3bench_repro.editors.llava_runtime import seed_everything
    counts_start=dict(r.COUNTS)
    with worker.network(gpu,loaded) as (runtime,bindings):
        try:
            cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v
                for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()}
            cpu_cache=p.cache_cpu(cache)
            runtime.model.half();prepared=r.s.prepare(runtime,bindings,task,rows,True)
            r.old.fp32(runtime);sources=[r.old.batch_on(runtime,v[1]) for v in prepared]
            seed_everything(task['seed']);initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
            saved=r.p.load_state(worker.point(task,'C'))['states']
            assert all(torch.equal(initial[k],saved['BASE'][k]) for k in initial)
            penalty=float(r.mm.operator_penalty({k:v.to(runtime.device) for k,v in saved['FINAL'].items()},cache))
            p.j.close(penalty,calibration['C_operator_penalty'])
            base=r.mm.canonicalize(initial)[0];expert=r.p.expert(base,task['seed'],runtime.device)
            hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
            try:
                refs=[r.q.logits(runtime,hook,b,verify_base=True) for b in sources]
                before=r.old.snapshot(expert)
                with c.diagnostic_scope(hook):
                    rng=c.rng();_,first=trajectory.source_gradients(r,runtime,hook,expert,sources,1)
                    gradient={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                    cuda_after=c.rng()['cuda_rng'];c.restore_rng(rng);expert.load_state_dict(before)
                    _,second=trajectory.source_gradients(r,runtime,hook,expert,sources,1)
                    assert first==second and all(torch.equal(gradient[k],v.grad) for k,v in expert.named_parameters())
                    assert torch.equal(cuda_after,c.rng()['cuda_rng'])
                expert.load_state_dict(before)
                c.write(RUN/'private'/f'NATIVE_FMIN_NETWORK_ADMISSION_{owner}.json',dict(status='PASS',
                    actual_zero_function_Base=True,source_gradient_and_GPU_RNG_exact=True,
                    C_Fisher_budget_parity=True,validation_backwards=4,scientific_updates=0,epoch=time.time()))
                binding=dict(arm='FISHER_MIN',seed=task['seed'],source_recipe=contract.SOURCE_RECIPE,
                    source_order=[v['query_id'] for v in rows],source_binding=calibration['source_binding'],
                    scope_binding=c.digest(scope),source_version=c.read(RUN/'private'/source_file)['commit'],
                    budget=budget,steps=160,parameters=7168)
                scalar=dict(step=0,second_moments=[0.,0.]);curve=[];start=dict(r.COUNTS)
                resume=RUN/'private/resume146'/f'FISHER_MIN_{owner}.pt';assert not resume.exists()
                for step in range(1,161):
                    c.budget();before=r.old.snapshot(expert)
                    _,losses=trajectory.source_gradients(r,runtime,hook,expert,sources,step)
                    norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
                    assert torch.isfinite(norm) and all(torch.isfinite(v.grad).all() for v in expert.parameters())
                    pure,scalar=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in expert.named_parameters()},scalar)
                    assert scalar['step']==step
                    pure=g.canonical_match(before,pure,reference['curve'][step-1]['operator_step_norm'])
                    numpy_before=np.random.get_state();correction_rng=c.rng()
                    with c.diagnostic_scope(hook):
                        selected,audit=original.correction(runtime,hook,expert,before,pure,step,sources,losses,
                            cache,cpu_cache,budget,'FISHER_MIN')
                    after_rng=c.rng()
                    assert after_rng['python_rng']==correction_rng['python_rng']
                    assert all(torch.equal(after_rng[k],correction_rng[k]) for k in ('torch_rng','cuda_rng'))
                    np.random.set_state(numpy_before);expert.load_state_dict(selected)
                    assert scalar['step']==step
                    if step==1:
                        before_loss=r.losses_at(runtime,hook,expert,before,g.source_batches(sources,step))
                        p.j.close(before_loss,losses)
                        pure_gain=g.gain(losses,r.losses_at(runtime,hook,expert,pure,g.source_batches(sources,step)))
                        selected_gain=g.gain(losses,r.losses_at(runtime,hook,expert,selected,g.source_batches(sources,step)))
                        baseline=audit['baseline']['original'];p.j.close(pure_gain,baseline['pure_source_gain'])
                        candidate_penalty=float(r.mm.operator_penalty(selected,cache))
                        finite=p.f.finite(selected_gain,pure_gain,candidate_penalty,budget,
                            r.fm.map_norm(before,selected),r.fm.map_norm(before,pure))
                        unchanged=all(torch.equal(before[k],selected[k]) for k in before)
                        assert finite or (unchanged and abs(selected_gain)<=1e-6)
                        c.write(RUN/'private'/f'NATIVE_FMIN_FIRST_STEP_REAL_{owner}.json',dict(status='PASS',
                            before_pure_selected_losses_verified=True,pure_gain=pure_gain,selected_gain=selected_gain,
                            Fisher_ratio=candidate_penalty/budget,source_RMS_clock=step,
                            selected_finite_rule_pass=finite,unchanged_skip=unchanged,diagnostic_GPU_RNG_exact=True,epoch=time.time()))
                    expert.load_state_dict(selected);r.COUNTS['updates']+=1
                    assert r.COUNTS['backwards']-counts_start['backwards']<=16004 and r.COUNTS['operator_backwards']-counts_start['operator_backwards']<=5120
                    curve.append(dict(step=step,source_NLL=losses,decision=audit,
                        operator_step_norm=r.fm.map_norm(before,selected)))
                    if step%20==0:
                        state=dict(state=r.old.snapshot(expert),scalar=copy.deepcopy(scalar),step=step,
                            curve=copy.deepcopy(curve),binding=binding,counts=dict(r.COUNTS),rng=c.rng(),
                            numpy_rng=copy.deepcopy(np.random.get_state()),counts_start=counts_start)
                        validate(state,binding);c.save(resume,state);r.save_counts()
                r.finish_candidate(runtime,hook,expert,task,'FISHER_MIN',base,sources,refs,curve,start,scalar,
                    mechanism=dict(budget=budget,C_operator_penalty=calibration['C_operator_penalty'],
                        source_recipe=contract.SOURCE_RECIPE,source_binding=binding['source_binding'],
                        final_penalty=float(r.mm.operator_penalty(r.old.snapshot(expert),cache)),alignment_rule=False))
                c.write(RUN/'private'/f'NATIVE_FMIN_COMPLETE_{owner}.json',dict(status='COMPLETE',steps=160,
                    last_consumers=['native/FIT single performance','native/FIT real R0 bank'],generation=0,Judge=0,epoch=time.time()))
            finally:hook.detach()
        finally:r.save_counts()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f'FAILURE_native_fmin_{os.environ.get("OWNER","0")}.json',
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False));raise
