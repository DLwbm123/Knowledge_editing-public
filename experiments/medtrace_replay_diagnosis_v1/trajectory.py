"""Fixed-prefix replay diagnosis; historical endpoints and routes stay read-only."""
import json
import os
import random
import sqlite3
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import replay as parent_code

p,c=parent_code.p,parent_code.c
RUN=parent_code.RUN
PARENT=Path(os.environ['REPLAY_PARENT'])
ARMS=parent_code.ARMS
NODES=(12,48,96)
GPUS=(4,5,6,7)


def label(arm,node):return f'{arm}_s{node:03d}'
def point(t,arm,node):return RUN/'private/weights'/t['anonymous_edit']/arm/f'step{node:03d}.pt'
def output(arm,node,qid):return RUN/'private/outputs'/label(arm,node)/(c.digest(qid)+'.json')
def lock():return c.read(RUN/'private/TRAJECTORY_LOCK.json')


def prepare():
    checks=parent_code.selfcheck();tasks=p.tasks();indices=list(range(0,32,2));selected=[tasks[i] for i in indices]
    ids={t['edit_id'] for t in selected};baseline=c.read(PARENT/'private/BASELINE.json');queries=p.queries(146)
    main={qid:row for qid,row in queries.items() if c.read(baseline[qid]['source_path'])['effective_expert'] in ids}
    fresh=[x for x in c.read(PARENT/'private/REPLAY_CHECK.json') if x['forced_expert_index'] in indices]
    assert len(fresh)==48 and len({x['source_group'] for x in fresh})==16
    jobs=[]
    for qid,row in main.items():
        d=c.read(baseline[qid]['source_path']);jobs.append(dict(row=row,expert=d['effective_expert'],fresh=False))
    for row in fresh:jobs.append(dict(row=row,expert=tasks[row['forced_expert_index']]['edit_id'],fresh=True))
    expected=len(jobs)*2*len(NODES)
    frozen=dict(selected_indices=indices,selected_edits=[t['edit_id'] for t in selected],nodes=list(NODES),
        displayed_nodes=[0,12,48,96,192],arms=list(ARMS),GPUs=list(GPUS),main_queries=len(main),CHECK_queries=len(fresh),
        new_outputs=expected,training_trajectories=32,training_updates=3264,node_weights=96,
        parent_gpu_binding=c.read(PARENT/'private/GPU_SOURCE_VERSION.json'),
        selection='even zero-based indices among original first32; no outcome selection',
        reference_mask='all original queries whose frozen selected expert is in fixed16; conditional routed cohort')
    path=RUN/'private/TRAJECTORY_LOCK.json'
    if path.exists():assert c.read(path)==frozen
    else:c.write(path,frozen)
    c.write(RUN/'private/TRAJECTORY_JOBS.json',jobs)
    # Freeze every consumer before creating new weights.
    old_db=sqlite3.connect('file:'+str(PARENT/'private/judge_replay_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
    old_db.row_factory=sqlite3.Row
    old={(x['method'],x['query_id']):x['path'] for x in old_db.execute('SELECT * FROM consumer')}
    consumers=[]
    for job in jobs:
        row=job['row'];qid=row['query_id'];mode='forced_source_CHECK' if job['fresh'] else 'bank_R0'
        for arm in ARMS:
            for node in (0,)+NODES+(192,):
                path=old['W0' if node==0 else arm,qid] if node in (0,192) else str(output(arm,node,qid))
                consumers.append(dict(arm=label(arm,node),mode=mode,query_id=qid,path=path))
        if job['fresh']:consumers.append(dict(arm='BASE',mode=mode,query_id=qid,path=old['BASE',qid]))
    assert len(consumers)==len(jobs)*10+48
    c.write(RUN/'private/CONSUMERS.json',consumers)
    public={k:v for k,v in frozen.items() if k not in ('selected_edits','parent_gpu_binding')}
    public.update(status='PASS',checks=checks,consumers=len(consumers),new_Judge_upper_bound=expected)
    c.write(RUN/'public/ADMISSION.json',public)
    return selected


def train(runtime,t,maxstep):
    from m3bench_repro.editors.llava_runtime import seed_everything
    receipt=RUN/'private/training'/t['anonymous_edit']/'TRAINING.json'
    if receipt.exists():
        d=c.read(receipt);assert d['maxstep']==maxstep
        if maxstep==192:assert d['state_parity_at192'] is True
        assert all(point(t,a,n).exists() for a in ARMS for n in NODES)
        return d
    initial=p.load_state(p.initial(t));native=runtime.build_edit_batch(c.record(t))
    fits=[runtime.build_edit_batch(replace(c.record(t),question=q)) for q in t['fit_questions']]
    replay_rows=c.read(PARENT/'private/REPLAY_FIT.json')
    replay_batches=[runtime.build_edit_batch(parent_code.record(row,p.tasks()[0])) for row in replay_rows]
    assert len(replay_batches)==192 and all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in replay_batches)
    order=list(range(4));random.Random(t['seed']).shuffle(order)
    groups=list(range(64));random.Random(t['seed']).shuffle(groups)
    replay_order=[3*g+k for g in groups for k in range(3)]
    assert sorted(replay_order)==list(range(192))
    result=[]
    for arm in ARMS:
        original_receipt=c.read(PARENT/'private/weights'/arm/t['anonymous_edit']/'TRAINING.json')
        assert original_receipt['binding']['fit_order']==order
        assert original_receipt['binding']['replay_order']==(replay_order if arm==ARMS[1] else [])
        e=p.expert(initial['expert'],t['seed'],runtime.device);opt=p.optimizer(e,runtime.model);seed_everything(t['seed'])
        binding=dict(trajectory_lock=c.digest(lock()),W0=initial['state_hash'],arm=arm,edit_id=t['edit_id'],maxstep=maxstep,
            fit_order=order,replay_order=replay_order if arm==ARMS[1] else [],execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'))
        latest=RUN/'private/weights'/t['anonymous_edit']/arm/'latest.pt';curve=[];start=0
        if latest.exists():
            saved=p.load_state(latest);assert saved['binding']==binding
            e.load_state_dict(saved['expert']);opt.load_state_dict(saved['optimizer']);c.restore_rng(saved);curve=saved['curve'];start=saved['step']
        hook=parent_code.MedTraceLayerHook(runtime.get_module(c.LAYER),e);hook.attach()
        try:
            for step in range(start+1,maxstep+1):
                c.budget();batch=replay_batches[replay_order[step-1]] if arm==ARMS[1] else None
                item=parent_code.update(runtime,hook,e,opt,native,fits[order[(step-1)%4]],batch)
                curve.append(dict(step=step,**item))
                if step in NODES:
                    state_hash=c.state_hash(e)
                    delta={k:float((v.detach().cpu()-initial['expert'][k]).norm()) for k,v in e.state_dict().items()}
                    c.save(point(t,arm,step),dict(binding=binding,expert=e.state_dict(),step=step,state_hash=state_hash))
                    seen=replay_order[:step] if arm==ARMS[1] else []
                    c.write(point(t,arm,step).with_suffix('.json'),dict(step=step,state_hash=state_hash,parameter_L2_delta=delta,
                        replay_samples=len(seen),replay_source_groups=len({replay_rows[i]['source_group'] for i in seen}),
                        replay_answer_types={kind:sum(replay_rows[i]['answer_kind']==kind for i in seen) for kind in ('yes','no','open')}))
                if step%32==0 or step in NODES:
                    c.save(latest,dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),curve=curve,step=step,**c.rng()))
                    print('TRAIN',t['order'],arm,step,flush=True)
            parity=None
            if maxstep==192:
                qid=t['edit_id'];baseline=c.read(PARENT/'private/outputs'/arm/(c.digest(qid)+'.json'))
                expected=baseline['binding']['phase']['weights'][t['edit_id']]['hash'];actual=c.state_hash(e)
                assert actual==expected,'Fixed-prefix reconstruction differs from historical192 state; stop without relaxing validation'
                parity=True
            result.append(dict(arm=arm,updates=maxstep,curve=curve,state_parity_at192=parity,resume_removed_bytes=latest.stat().st_size))
        finally:hook.detach()
        assert all(point(t,arm,n).exists() for n in NODES)
        latest.unlink();del e,opt
    out=dict(maxstep=maxstep,arms=result,state_parity_at192=all(x['state_parity_at192'] for x in result) if maxstep==192 else None)
    c.write(receipt,out);return out


def evaluate(runtime,t):
    jobs=[x for x in c.read(RUN/'private/TRAJECTORY_JOBS.json') if x['expert']==t['edit_id']]
    assert jobs
    for arm in ARMS:
        for node in NODES:
            weight=point(t,arm,node);done=weight.with_suffix('.CONSUMED.json')
            if done.exists():assert not weight.exists();continue
            saved=p.load_state(weight);e=p.expert(saved['expert'],t['seed'],runtime.device)
            assert c.state_hash(e)==saved['state_hash'],'Saved node state does not match retained binding'
            mixture=p.Mixture([e]);mixture.ids=[0];mixture.weights=[1.]
            hook=parent_code.MedTraceLayerHook(runtime.get_module(c.LAYER),mixture)
            completed=[]
            for job in jobs:
                row=job['row'];qid=row['query_id'];dest=output(arm,node,qid)
                if dest.exists():
                    existing=c.read(dest);assert existing['binding']['phase']['state_hash']==saved['state_hash'];completed.append(str(dest));continue
                oldarm='W0' if job['fresh'] else arm
                oldpath=PARENT/'private/outputs'/oldarm/(c.digest(qid)+'.json')
                old=c.read(oldpath);binding=old['binding']['judge_input']
                raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
                assert raw['image_sha256']==binding['image_sha256'] and raw['input_ids'].tolist()==[binding['prompt_ids']]
                assert raw['attention_mask'].tolist()==[binding['attention_mask']] and runtime.generation_config==binding['generation']
                row=dict(row,image_sha256=binding['image_sha256'])
                phase=dict(arm=label(arm,node),node=0,prefix=146,slot=0,steps=node,state_hash=saved['state_hash'],
                    execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),trajectory_lock=c.digest(lock()))
                hook.attach();began=time.time()
                try:
                    with torch.inference_mode(),hook.generation_request():
                        g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                finally:hook.detach()
                out=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids))
                d=dict(binding=dict(input=row,judge_input=binding,phase=phase,arm=label(arm,node),
                    mode='forced_source_CHECK' if job['fresh'] else 'bank_R0',prefix=146,owner_order=146),R0=out,
                    effective_expert=t['edit_id'],route=dict(activated=True,logical_edit_id=t['edit_id']),
                    U_KL=None,Base_token_consistency=None,seconds=time.time()-began,
                    active_target=old['active_target'],original_route_preserved=not job['fresh'],forced_expert_stress_test=job['fresh'],
                    output_shape=dict(tokens=len(out['raw_token_ids']),empty=not out['raw_answer'].strip(),
                        first_token=out['raw_token_ids'][0] if out['raw_token_ids'] else None,eos_token=runtime.adapter.tokenizer.eos_token_id))
                c.write(dest,d);completed.append(str(dest))
                print('EVAL',t['order'],arm,node,len(completed),len(jobs),flush=True)
            assert len(completed)==len(jobs) and all(Path(x).is_file() for x in completed)
            assert not weight.is_symlink() and weight.resolve().is_relative_to((RUN/'private/weights').resolve())
            size=weight.stat().st_size
            c.write(done,dict(status='CONSUMED',consumers=completed,state_hash=saved['state_hash'],bytes=size,bindings_durable=True,rebuild='retrain'))
            weight.unlink();del e,mixture,hook
    assert not any(x.grad is not None for x in runtime.model.parameters())


def mechanical():
    tasks=p.tasks();t=tasks[lock()['selected_indices'][0]]
    with p.lease(4):
        runtime,_=c.load(4);r=train(runtime,t,192)
        peak=torch.cuda.max_memory_allocated();assert r['state_parity_at192']
        c.write(RUN/'public/GPU_MECHANICAL.json',dict(status='PASS',fixed_expert_index=0,arms=2,
            original192_state_reproduced=True,updates=384,included_in_formal_training=True,peak_allocated_bytes=peak,Base_gradient=False))
    p.done('MECHANICAL_COMPLETE')


def worker():
    part=int(os.environ['PARTITION']);gpu=int(os.environ['GPU']);assert gpu==GPUS[part]
    assert c.read(RUN/'public/GPU_MECHANICAL.json')['status']=='PASS'
    selected=[p.tasks()[i] for i in lock()['selected_indices']]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t in selected[part::4]:
            receipt=RUN/'private/training'/t['anonymous_edit']/'TRAINING.json'
            if not receipt.exists():train(runtime,t,96)
            evaluate(runtime,t)
    p.done('WORKER_'+str(part))


def controller():
    import pipeline
    assert c.read(RUN/'public/ADMISSION.json')['status']=='PASS' and c.read(RUN/'public/GPU_MECHANICAL.json')['status']=='PASS'
    p.progress('TRAJECTORY_GPU_RUNNING')
    pipeline.wait([pipeline.launch('trajectory.py','trajectory_worker',g,i) for i,g in enumerate(GPUS)])
    rows=c.read(RUN/'private/CONSUMERS.json');assert all(Path(x['path']).is_file() for x in rows)
    assert not list((RUN/'private/weights').rglob('*.pt'))
    consumed=list((RUN/'private/weights').rglob('*.CONSUMED.json'));assert len(consumed)==96
    training=[c.read(x) for x in (RUN/'private/training').glob('*/TRAINING.json')]
    p.done('DELETION',dict(node_files=96,node_bytes=sum(c.read(x)['bytes'] for x in consumed),
        resume_files=32,resume_bytes=sum(z['resume_removed_bytes'] for x in training for z in x['arms']),
        all_consumers_complete=True,historical_assets_untouched=True))
    p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('trajectory_queue.py','trajectory_ingest')])
    p.progress('TRAJECTORY_ASTRA_SCORING');root=RUN/'private/judge_trajectory_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('trajectory_report.py','trajectory_report')])


if __name__=='__main__':
    try:
        action=os.environ['ACTION'];{'prepare':prepare,'trajectory_mechanical':mechanical,'trajectory_worker':worker,'trajectory_controller':controller}[action]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
