"""Consume the first two native/FIT calibrations; never launch the full146 lane."""
import os
from pathlib import Path
import time
import traceback
import torch
import runtime146 as worker
import contract
import trajectory
import training_math

r,c,RUN=worker.r,worker.c,worker.RUN


def require_admission(full=False):
    gate,scope_file,source_file=worker.stage_admission('M13',full)
    assert gate['status']=='PASS'
    if not full:assert gate['owners']==[1,2]
    assert gate['packet_binding']==c.digest(worker.packet())
    assert gate['scope_binding']==c.digest(c.read(RUN/'private'/scope_file))
    assert c.read(RUN/'private/NATIVE_TRAJECTORY_CPU_CHECKS1.json')['status']=='PASS'
    assert contract.source_admission(worker.packet())['status']=='PASS'
    for owner in (1,2):
        assert c.read(RUN/'private'/f'CALIBRATION_COMPLETE_{owner}.json')['steps']==160
        assert c.read(RUN/'private'/f'NETWORK_ADMISSION_{owner}.json')['status']=='PASS'
    worker.configure()
    counted=r.count;forward_start=r.COUNTS['non_generation_forwards']
    def count(*args):
        counted(*args)
        assert r.COUNTS['non_generation_forwards']-forward_start<=5000
    r.count=count
    return gate,scope_file,source_file


def fisher():
    require_admission()
    assert os.environ['GPU']=='0'
    with (RUN/'private/NATIVE_FISHER_ONCE1').open('x'):pass
    # The original Fisher only consumes a verified zero-function C BASE.
    first=worker.tasks()[0]
    saved=r.p.load_state(worker.point(first,'C'))['states']['BASE']
    expected=r.p.tr.TT4(first['seed'],8,8).state_dict()
    assert all(torch.equal(saved[k],expected[k]) for k in expected)
    c.write(RUN/'private/REFERENCE_EXACT.json',dict(status='PASS',owners=[1,2],
        zero_function_C_BASE_exact=True,not_full146_reference=True))
    r.fisher()
    assert r.COUNTS['backwards']==1952 and r.COUNTS['updates']==0
    c.write(RUN/'private/NATIVE_FISHER_COMPLETE1.json',dict(status='COMPLETE',
        same_frozen_basis=True,source_training_used=False,epoch=time.time()))


def train(full=False,loaded=None):
    gate,scope_file,source_file=require_admission(full)
    owner=int(os.environ['OWNER']);gpu=int(os.environ['GPU'])
    assert owner in gate['owners'] and gpu in (0,1)
    if not full:assert owner==gpu+1 and owner in (1,2)
    assert c.read(RUN/'public/FISHER_ADMISSION.json')['status']=='PASS'
    assert c.read(RUN/'private/NATIVE_FISHER_COMPLETE1.json')['status']=='COMPLETE'
    with (RUN/'private'/f'NATIVE_M13_ONCE_{owner}').open('x'):pass
    task=worker.tasks()[owner-1];rows=trajectory.source_rows(worker.packet(),owner)
    assert not worker.point(task,'M13').exists()
    from m3bench_repro.editors.llava_runtime import seed_everything
    counts_start=dict(r.COUNTS)
    with worker.network(gpu,loaded) as (runtime,bindings):
        try:
            cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v
                for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()}
            runtime.model.half();prepared=r.s.prepare(runtime,bindings,task,rows,True)
            r.old.fp32(runtime);batches=[r.old.batch_on(runtime,v[1]) for v in prepared]
            initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
            saved=r.p.load_state(worker.point(task,'C'))['states']
            assert all(torch.equal(initial[k],saved['BASE'][k]) for k in initial)
            ref=c.read(RUN/'private/results/C'/f'{owner}.json')
            assert len(ref['values'])==5 and len(ref['curve'])==160
            assert c.read(RUN/'private'/f'CALIBRATION_COMPLETE_{owner}.json')['source_binding']==c.digest(contract.original.source_binding(rows))
            gain=-.5*ref['values'][0]['NLL_change']-.125*sum(v['NLL_change'] for v in ref['values'][1:])
            penalty=float(r.mm.operator_penalty({k:v.to(runtime.device) for k,v in saved['FINAL'].items()},cache))
            coefficient=training_math.coefficient(gain,penalty)
            targets=[v['operator_step_norm'] for v in ref['curve']]
            c.write(RUN/'private/calibration146'/f'{owner}.json',dict(C_source_gain=gain,
                C_operator_penalty=penalty,coefficient=coefficient,FISHER_MIN_budget=.1*penalty,
                source_binding=c.digest(contract.original.source_binding(rows)),heldout_used=False))
            seed_everything(task['seed']);base=r.mm.canonicalize(initial)[0]
            expert=r.p.expert(base,task['seed'],runtime.device)
            hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
            try:
                refs=[r.q.logits(runtime,hook,b,verify_base=True) for b in batches]
                before=r.old.snapshot(expert)
                with c.diagnostic_scope(hook):
                    rng=c.rng();_,first=trajectory.source_gradients(r,runtime,hook,expert,batches,1)
                    gradients={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                    cuda_after=c.rng()['cuda_rng'];c.restore_rng(rng);expert.load_state_dict(before)
                    _,second=trajectory.source_gradients(r,runtime,hook,expert,batches,1)
                    assert first==second and all(torch.equal(gradients[k],v.grad) for k,v in expert.named_parameters())
                    assert torch.equal(cuda_after,c.rng()['cuda_rng'])
                expert.load_state_dict(before)
                c.write(RUN/'private'/f'NATIVE_M13_NETWORK_ADMISSION_{owner}.json',dict(status='PASS',
                    canonical_zero_function_Base=True,source_gradient_and_GPU_RNG_exact=True,
                    validation_backwards=4,scientific_updates=0,epoch=time.time()))
                binding=dict(arm='M13',seed=task['seed'],steps=160,parameters=7168,
                    source_recipe=contract.SOURCE_RECIPE,source_order=[v['query_id'] for v in rows],
                    source_binding=c.digest(contract.original.source_binding(rows)),
                    scientific_lock=c.digest(c.read(RUN/'private'/scope_file)),
                    source_version=c.read(RUN/'private'/source_file)['commit'],
                    budget_binding=c.digest(c.read(RUN/'PLAN_CONFIG.json')))
                start=dict(r.COUNTS);resume=RUN/'private/resume146'/f'M13_{owner}.pt'
                assert not resume.exists(),'Interrupted state requires an independent audit'
                def checkpoint(state):c.save(resume,state);r.save_counts()
                final=trajectory.run(r,runtime,hook,expert,batches,'M13',cache,coefficient,targets,binding,checkpoint)
                assert r.COUNTS['updates']-counts_start['updates']==160 and r.COUNTS['backwards']-counts_start['backwards']==324
                assert r.COUNTS['operator_backwards']-counts_start['operator_backwards']==160 and r.COUNTS['local_solves']==counts_start['local_solves']
                r.finish_candidate(runtime,hook,expert,task,'M13',base,batches,refs,final['curve'],start,final['scalar'],
                    mechanism=dict(coefficient=coefficient,C_operator_penalty=penalty,C_source_gain=gain,
                        source_recipe=contract.SOURCE_RECIPE,source_binding=binding['source_binding'],
                        soft_Fisher_only=True,no_guard_or_alignment_rejection=True))
                c.write(RUN/'private'/f'NATIVE_M13_COMPLETE_{owner}.json',dict(status='COMPLETE',steps=160,
                    last_consumers=['native/FIT single performance','native/FIT real R0 bank'],
                    generation=0,Judge=0,epoch=time.time()))
            finally:hook.detach()
        finally:r.save_counts()


if __name__=='__main__':
    try:
        if os.environ['ACTION']=='native_fisher':fisher()
        elif os.environ['ACTION']=='native_m13':train()
        else:raise ValueError(os.environ['ACTION'])
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f'FAILURE_{os.environ["ACTION"]}_{os.environ.get("OWNER","0")}.json',
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),retry=False))
        raise
