"""New M13 original146 worker; explicit admissions precede every GPU load."""
import os
from contextlib import contextmanager
from pathlib import Path
import time
import traceback
import torch
import contract
import trajectory
import training_math
import module_run as r

c,RUN=r.c,r.RUN


def packet():return c.read(RUN/'private/M13_SOURCE_PACKET.json')
def tasks():return packet()['original_tasks']
def point(task,arm):return RUN/'private/candidates'/arm/f"{task['order']}.pt"
def dest(row):return RUN/'private/outputs'/f"{row['method']}_{row['mode']}_p{row['prefix']}"/str(row['owner'])/(c.digest(row['query']['query_id'])+'.json')
def base_path(query):return dest(dict(method='BASE',mode='base',prefix=0,owner=0,query=query))


def require_ready():
    admission=c.read(RUN/'private/M13_146_RUNTIME_ADMISSION.json')
    assert admission['status']=='PASS' and admission['launch_authorized'] is True
    assert admission['source_commit']==c.read(RUN/'private/GPU_SOURCE_VERSION.json')['commit']
    p=packet();assert contract.source_admission(p)['status']=='PASS'
    assert p['source_recipe']==contract.SOURCE_RECIPE
    assert admission['packet_binding']==c.digest(p)
    assert admission['scope_binding']==c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))
    assert admission['plan_binding']==c.digest(c.read(RUN/'private/EVAL_PLAN146.json'))
    scope=c.read(RUN/'private/SCOPE_LOCK.json')
    assert scope['method_flags']=={k:list(v) for k,v in contract.FLAGS.items()}
    assert scope['source_recipe']==contract.SOURCE_RECIPE
    assert scope['source_weights']==list(contract.SOURCE_WEIGHTS) and scope['steps']==160 and scope['parameters']==7168
    assert c.read(RUN/'PLAN_CONFIG.json')['maximum_counts']==contract.CAPS
    assert admission['physical_source_paths_checked'] and admission['cold_initialization_exact']
    assert admission['original_R0_bindings_checked'] and admission['storage_write_read']
    assert c.read(RUN/'private/SOURCE_ADMISSION146.json')['status']=='PASS'
    for task in tasks():
        for row in trajectory.source_rows(p,task['order']):assert Path(row['image_path']).is_file()


def configure():
    r.s.tasks=tasks;r.s.source=lambda task:trajectory.source_rows(packet(),task['order']);r.s.point=point
    r.s.baseline=lambda task,row:base_path(row)
    r.GPUS=(int(os.environ.get('GPU','0')),)
    r.COUNTS.setdefault('local_solves',0)
    r.source_gradients=lambda *args,**kwargs:trajectory.source_gradients(r,*args,**kwargs)
    def count(*_):
        c.budget();r.COUNTS['forwards']+=1
        if not r.GENERATING:r.COUNTS['non_generation_forwards']+=1
        assert r.COUNTS['non_generation_forwards']<=getattr(r,'NATIVE_FORWARD_CAP',contract.CAPS['non_generation_forwards'])
    r.count=count


@contextmanager
def network(gpu,loaded=None):
    # A full-lane worker owns one lease/model; each task still gets a fresh expert.
    if loaded is not None:
        yield loaded
        return
    with r.p.lease(gpu):
        runtime,bindings,handle=r.load(gpu)
        try:yield runtime,bindings
        finally:handle.remove()


def stage_admission(stage,full):
    if full:
        from selected_training146 import require_admission,source_file
        return require_admission(), 'SELECTED_SCOPE1.json', source_file()
    names={'C':'NATIVE_CALIBRATION','M13':'NATIVE_FOLLOWUP','FISHER_MIN':'NATIVE_FMIN'}
    name=names[stage]
    return c.read(RUN/'private'/f'{name}_ADMISSION1.json'), (
        'SCOPE_LOCK.json' if stage=='C' else name+'_SCOPE1.json'), (
        'GPU_SOURCE_VERSION.json' if stage=='C' else name+'_SOURCE1.json')


def train(arms=None,partitions=3):
    from m3bench_repro.editors.llava_runtime import seed_everything
    require_ready();configure();part,gpu=int(os.environ['PARTITION']),int(os.environ['GPU'])
    reference=os.environ['ACTION']=='m13_146_reference'
    if not reference:assert c.read(RUN/'public/FISHER_ADMISSION.json')['status']=='PASS'
    with r.p.lease(gpu):
        runtime,bindings,handle=r.load(gpu)
        cache=None if reference else {k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v
            for k,v in r.p.load_state(RUN/'private/OPERATOR_FISHER.pt').items()}
        try:
            p=packet()
            for task in tasks()[part::partitions]:
                rows=trajectory.source_rows(p,task['order'])
                runtime.model.half();prepared=r.s.prepare(runtime,bindings,task,rows,True)
                frozen=[item[1] for item in prepared]
                r.old.fp32(runtime);batches=[r.old.batch_on(runtime,v) for v in frozen]
                initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
                assert sum(v.numel() for v in initial.values())==7168 and torch.count_nonzero(initial['G1'])==0
                targets=None;coefficient=0.
                if not reference:
                    ref=c.read(RUN/'private/results/C'/f"{task['order']}.json")
                    saved=r.p.load_state(point(task,'C'))['states']
                    assert all(torch.equal(initial[k],saved['BASE'][k]) for k in initial),'Cold seed/init changed'
                    targets=[v['operator_step_norm'] for v in ref['curve']]
                    assert len(ref['values'])==contract.SOURCE_COUNT
                    gain=-.5*ref['values'][0]['NLL_change']-.125*sum(v['NLL_change'] for v in ref['values'][1:])
                    penalty=float(r.mm.operator_penalty({k:v.to(runtime.device) for k,v in saved['FINAL'].items()},cache))
                    prior=c.read(RUN/'private/PRIOR_CALIBRATIONS.json').get(str(task['order']))
                    if prior:
                        assert prior['source_binding']==c.digest(contract.original.source_binding(rows))
                        assert abs(penalty-prior['C_operator_penalty'])<=1e-12+1e-4*abs(prior['C_operator_penalty'])
                    coefficient=training_math.coefficient(gain,penalty,prior['coefficient'] if prior else None)
                    path=RUN/'private/calibration146'/f"{task['order']}.json";assert not path.exists()
                    c.write(path,dict(C_source_gain=gain,C_operator_penalty=penalty,coefficient=coefficient,heldout_used=False))
                for arm in ('C',) if reference else (arms or ('M1','M3','M13')):
                    seed_everything(task['seed']);assert not point(task,arm).exists(),'No trajectory restart'
                    resume=RUN/'private/resume146'/f"{arm}_{task['order']}.pt";assert not resume.exists(),'Interrupted state requires an independent audit'
                    expert=r.p.expert(initial,task['seed'],runtime.device)
                    base=r.mm.canonicalize(initial)[0] if arm in ('M1','M13') else initial
                    expert.load_state_dict(base)
                    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                    try:
                        refs=[r.q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(batches)]
                        expert.load_state_dict(initial)
                        binding=dict(arm=arm,seed=task['seed'],steps=160,parameters=7168,source_recipe=contract.SOURCE_RECIPE,
                            source_order=[v['query_id'] for v in rows],source_binding=c.digest(contract.original.source_binding(rows)),
                            scientific_lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json')),
                            source_version=c.read(RUN/'private/GPU_SOURCE_VERSION.json')['commit'],budget_binding=c.digest(c.read(RUN/'PLAN_CONFIG.json')['maximum_counts']))
                        start=dict(r.COUNTS)
                        def checkpoint(saved):c.save(resume,saved);r.save_counts()
                        final=trajectory.run(r,runtime,hook,expert,batches,arm,cache,
                            coefficient if arm in ('M3','M13') else 0.,targets,binding,checkpoint)
                        r.finish_candidate(runtime,hook,expert,task,arm,base,batches,refs,final['curve'],start,final['scalar'],
                            mechanism=dict(coefficient=coefficient if arm in ('M3','M13') else 0.,source_binding=binding['source_binding']))
                        c.write(RUN/'private/retired146'/f"{arm}_{task['order']}.json",dict(bytes=resume.stat().st_size,last_consumer='160 final/save-load and durable diagnostics'))
                        resume.unlink()
                    finally:hook.detach()
            c.write(RUN/'private'/f"DONE_{os.environ['ACTION']}_{part}.json",dict(status='COMPLETE',epoch=time.time()))
        finally:handle.remove();r.save_counts()


def main():
    require_ready();configure();action=os.environ['ACTION']
    if action in ('m13_146_reference','m13_146_train'):train()
    elif action=='m13_146_fisher':r.fisher()
    elif action in ('m13_146_single','m13_146_bank'):
        from bank146 import evaluate
        evaluate()
    elif action=='m13_146_controller':
        from controller146 import controller
        controller()
    else:raise ValueError(action)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        r.save_counts();c.write(RUN/'private'/f"FAILURE_m13_146_{os.environ.get('ACTION')}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),automatic_retry=False));raise
