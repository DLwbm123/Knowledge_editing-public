"""Bounded first two native/FIT C trajectories; no mechanism, generation, or Judge."""
import copy
import os
from pathlib import Path
import time
import traceback
import torch
import contract
import trajectory
import runtime146 as worker

r,c,RUN=worker.r,worker.c,worker.RUN


def main(full=False,loaded=None):
    packet=worker.packet();gate=contract.source_admission(packet)
    assert packet['source_recipe']==contract.SOURCE_RECIPE and gate['status']=='PASS'
    admission,scope_file,source_file=worker.stage_admission('C',full)
    assert admission['status']=='PASS' and admission['packet_binding']==c.digest(packet)
    assert admission['physical_GPUs']==[0,1]
    if not full:assert admission['owners']==[1,2]
    gpu=int(os.environ['GPU']);owner=int(os.environ['OWNER']);assert owner in admission['owners'] and gpu in (0,1)
    if not full:assert owner==gpu+1
    with (RUN/'private'/f'CALIBRATION_ONCE_{owner}').open('x'):pass
    assert c.read(RUN/'private/NATIVE_FIT_CPU_CHECKS.json')['gradient']['status']=='PASS'
    task=packet['original_tasks'][owner-1];rows=trajectory.source_rows(packet,owner)
    assert all(Path(row['image_path']).is_file() for row in rows)
    worker.configure()
    counted=r.count;forward_start=r.COUNTS['non_generation_forwards']
    def count(*args):
        counted(*args);assert r.COUNTS['non_generation_forwards']-forward_start<=2000
    r.count=count
    from m3bench_repro.editors.llava_runtime import seed_everything
    counts_start=dict(r.COUNTS)
    with worker.network(gpu,loaded) as (runtime,bindings):
        try:
            runtime.model.half();prepared=r.s.prepare(runtime,bindings,task,rows,True)
            r.old.fp32(runtime);batches=[r.old.batch_on(runtime,v[1]) for v in prepared]
            seed_everything(task['seed']);initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
            assert sum(v.numel() for v in initial.values())==7168 and torch.count_nonzero(initial['G1'])==0
            expert=r.p.expert(initial,task['seed'],runtime.device)
            hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
            try:
                refs=[r.q.logits(runtime,hook,b,verify_base=True) for b in batches]
                before=r.old.snapshot(expert)
                with c.diagnostic_scope(hook):
                    rng=c.rng();_,first=trajectory.source_gradients(r,runtime,hook,expert,batches,1)
                    gradients={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                    cuda_after=c.rng()['cuda_rng'];c.restore_rng(rng)
                    expert.load_state_dict(before)
                    _,second=trajectory.source_gradients(r,runtime,hook,expert,batches,1)
                    assert first==second and all(torch.equal(gradients[k],v.grad) for k,v in expert.named_parameters())
                    assert torch.equal(cuda_after,c.rng()['cuda_rng'])
                expert.load_state_dict(before)
                c.write(RUN/'private'/f'NETWORK_ADMISSION_{owner}.json',dict(status='PASS',
                    actual_zero_function_Base=True,five_source_gradients_and_GPU_RNG_exact=True,
                    validation_backwards=4,scientific_updates=0,epoch=time.time()))
                binding=dict(arm='C',seed=task['seed'],steps=160,parameters=7168,
                    source_recipe=contract.SOURCE_RECIPE,source_order=[row['query_id'] for row in rows],
                    source_binding=c.digest(contract.original.source_binding(rows)),
                    scientific_lock=c.digest(c.read(RUN/'private'/scope_file)),
                    source_version=c.read(RUN/'private'/source_file)['commit'],
                    budget_binding=c.digest(c.read(RUN/'PLAN_CONFIG.json')))
                start=dict(r.COUNTS)
                resume=RUN/'private/resume146'/f'C_{owner}.pt'
                def checkpoint(saved):c.save(resume,saved);r.save_counts()
                final=trajectory.run(r,runtime,hook,expert,batches,'C',None,0.,None,binding,checkpoint)
                assert r.COUNTS['updates']-counts_start['updates']==160 and r.COUNTS['backwards']-counts_start['backwards']==324
                assert all(r.COUNTS[k]==counts_start[k] for k in ('operator_backwards','generation_attempts'))
                r.finish_candidate(runtime,hook,expert,task,'C',before,batches,refs,final['curve'],start,None,
                    mechanism=dict(source_recipe=contract.SOURCE_RECIPE,source_binding=binding['source_binding'],
                        native_calibration_first_two=True,reusable_only_with_exact_recipe_and_seed=True))
                c.write(RUN/'private'/f'CALIBRATION_COMPLETE_{owner}.json',dict(status='COMPLETE',
                    steps=160,seed=task['seed'],source_binding=binding['source_binding'],
                    last_consumers=['same-source M13 C-step matching','same-source FISHER_MIN C budget'],
                    generation=0,Judge=0,epoch=time.time()))
            finally:hook.detach()
        finally:r.save_counts()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f'FAILURE_native_calibration_{os.environ["OWNER"]}.json',
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False))
        raise
