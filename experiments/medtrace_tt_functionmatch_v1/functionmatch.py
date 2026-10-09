"""Fixed historical function-norm schedules crossed with Adam/RAW direction policies."""
import os
import time
import sqlite3
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import direction as d
import function_math as fm

p,c,RUN,scoped=d.p,d.c,d.RUN,d.scoped
PARENT=Path(os.environ['DIRECTION_PARENT'])
ARMS=('RAW_A_SCHEDULE','ADAM_R_SCHEDULE')
LABELS=('W0','RAW_DIR_540')+ARMS
TRAIN_GPUS=EVAL_GPUS=(2,3,4,5,6,7)
PRIMARY=ARMS[0]
selected,point,output=d.selected,d.point,d.output
# Reuse the unchanged frozen-route evaluation and its negative control.
d.ARMS,d.LABELS,d.TRAIN_GPUS,d.EVAL_GPUS=ARMS,LABELS,TRAIN_GPUS,EVAL_GPUS
evaluate=d.evaluate


def schedule(t,mode):return RUN/'private/schedules'/f"{t['order']}_{mode}.json"


def plan():
    assert (PARENT/'private/REPORT_COMPLETE.json').exists()
    checks=fm.selfcheck();db=sqlite3.connect('file:'+str(PARENT/'private/judge_direction_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
    old={(a,m,q):path for a,m,q,path in db.execute('SELECT method,mode,query_id,path FROM consumer')};db.close()
    routes=c.read(RUN/'private/ROUTES.json');chosen={t['edit_id'] for t in selected()}
    consumers=[];jobs=[]
    def add(arm,mode,row,expert,path=None):
        dest=str(path or output(arm,row['query_id']))
        consumers.append(dict(arm=arm,mode=mode,query_id=row['query_id'],path=dest,produced_arm=arm))
        if path is None:jobs.append(dict(arm=arm,mode=mode,row=row,expert=expert,path=dest))
    for q,row in p.queries(146).items():
        expert=routes[q]['effective_expert'];affected=expert in chosen
        for label,parent_label in [(x,x) for x in LABELS if x not in ARMS]:
            add(label,'bank_R0',row,expert,old[parent_label,'bank_R0',q])
        for arm in ARMS:add(arm,'bank_R0',row,expert,None if affected else old['W0','bank_R0',q])
    for row in c.read(RUN/'private/SOURCE_CHECK_ROWS.json'):
        q=row['query_id'];expert=p.tasks()[row['forced_expert_index']]['edit_id'];affected=expert in chosen
        for label,parent_label in [(x,x) for x in ('BASE','W0','RAW_DIR_540')]:
            add(label,'forced_source_CHECK',row,expert if label!='BASE' else None,old[parent_label,'forced_source_CHECK',q])
        for arm in ARMS:add(arm,'forced_source_CHECK',row,expert,None if affected else old['W0','forced_source_CHECK',q])
        expert=routes[q]['effective_expert'];affected=expert in chosen
        for label,parent_label in [(x,x) for x in ('BASE_NAT','W0_NAT','RAW_DIR_540_NAT')]:
            add(label,'natural_source_CHECK',row,expert if label!='BASE_NAT' else None,old[parent_label,'natural_source_CHECK',q])
        for arm in ARMS:add(arm+'_NAT','natural_source_CHECK',row,expert,None if affected else old['W0_NAT','natural_source_CHECK',q])
    assert len(jobs)==220 and len(consumers)==7012 and len({j['path'] for j in jobs})==220
    assert all(Path(x['path']).exists() for x in consumers if not Path(x['path']).is_relative_to(RUN/'private/outputs'))
    # Resolve every Base reference before starting any GPU work.
    bindings=c.read(RUN/'private/EVAL_BINDINGS.json')
    unique={j['row']['query_id']:j['row'] for j in jobs}
    assert all(scoped.base_tokens(row,bindings) for row in unique.values())
    lock=dict(arms=list(ARMS),primary=PRIMARY,experts=8,selected_edits=list(chosen),
        phases=[['original_tail',320]],optimizer_steps=10240,training_forwards=20480,
        training_backwards=20480,endpoint_rebuild_extra_steps=5120,initial_state='original_STAGED_220',
        new_generations=220,Judge_upper_bound=220,consumers=7012,training_roles=['native','FIT'],replay=False,
        training_GPUs=list(TRAIN_GPUS),inference_GPUs=list(EVAL_GPUS),parameters=7168,final_TT=16,
        function_matching=dict(metric='TT normalized-input linear-map Frobenius change',RTOL=fm.RTOL,ATOL=fm.ATOL,max_scale=fm.MAX_SCALE,max_bisections=fm.MAX_BISECTIONS),
        reference_trajectories=16,reference_steps=5120,new_trajectories=16,diagnostic_forwards=160)
    c.write(RUN/'private/DIRECTION_LOCK.json',lock);c.write(RUN/'private/JOBS.json',jobs);c.write(RUN/'private/CONSUMERS.json',consumers)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**{k:v for k,v in lock.items() if k!='selected_edits'},selfcheck=checks,Base_lookup_unique_inputs=len(unique)))
    p.done('PLAN_COMPLETE')


def fit(runtime,t,mode):
    from m3bench_repro.editors.llava_runtime import seed_everything
    reference=mode.startswith('REFERENCE_')
    policy=mode.removeprefix('REFERENCE_') if reference else ('RAW' if mode==ARMS[0] else 'ADAM')
    assert policy in ('RAW','ADAM') and (reference or mode in ARMS)
    expert=p.expert(d.start_state(t),t['seed'],runtime.device)
    batches=[runtime.build_edit_batch(row) for row in [c.record(t)]+[replace(c.record(t),question=q) for q in t['fit_questions']]]
    assert len(batches)==5 and all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in batches)
    fixed=None if reference else c.read(schedule(t,'ADAM' if mode==ARMS[0] else 'RAW'))
    if fixed:assert fixed['endpoint_exact'] and len(fixed['curve'])==320
    binding=dict(mode=mode,seed=t['seed'],steps=320,roles=['native','FIT'],replay=False,
        execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),scientific_lock=c.digest(c.read(RUN/'private/DIRECTION_LOCK.json')),
        schedule_binding=None if reference else c.digest(fixed))
    hook=scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
    seed_everything(t['seed']);opt=p.optimizer(expert,runtime.model);curve=[];start=0
    latest=RUN/'private/resume'/f"{t['order']}_{mode}.pt"
    try:
        if not reference:x=p.tr.activations(runtime,hook,batches,[]);f0=p.tr.functional(expert,x)
        if latest.exists():
            saved=p.load_state(latest);assert saved['binding']==binding
            expert.load_state_dict(saved['expert']);opt.load_state_dict(saved['optimizer']);c.restore_rng(saved)
            curve=saved['curve'];start=saved['step']
        for step in range(start+1,321):
            c.budget();opt.zero_grad(set_to_none=True)
            before={k:v.detach().clone() for k,v in expert.state_dict().items()};losses=[]
            for i in [0,1+(step-1)%4]:
                hook.set_teacher_routing(batches[i].labels);loss=runtime.compute_loss(batches[i]);assert torch.isfinite(loss)
                (.5*loss).backward();losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);assert torch.isfinite(norm)
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
            assert not any(v.grad is not None for v in runtime.model.parameters())
            opt.step();groups=d.apply_direction(opt,{id(v):before[k] for k,v in expert.named_parameters()},policy)
            candidate={k:v.detach().clone() for k,v in expert.state_dict().items()}
            candidate_norm=fm.map_norm(before,candidate)
            row=dict(step=step,CE=losses,gradient_norm=float(norm),groups=groups,candidate_map_norm=candidate_norm)
            if reference:row['actual_map_norm']=candidate_norm
            else:
                target=fixed['curve'][step-1]['actual_map_norm'];state,match=fm.match(before,candidate,target)
                expert.load_state_dict(state);row.update(function_match=match,actual_map_norm=match['actual'],
                    net_function_change=float((p.tr.functional(expert,x)-f0).norm()))
            curve.append(row)
            if step%20==0:
                c.save(latest,dict(binding=binding,expert=expert.state_dict(),optimizer=opt.state_dict(),step=step,curve=curve,**c.rng()))
                print('TRAIN',t['order'],mode,step,flush=True)
        final=[]
        if not reference:
            for b in batches:
                hook.set_teacher_routing(b.labels)
                with torch.inference_mode():out=runtime.model(**b.forward_kwargs())
                mask=b.labels[:,1:]!=-100;target=b.labels[:,1:][mask];logits=out.logits[:,:-1].float()[mask]
                final.append(dict(nll=float(out.loss),all_argmax_correct=bool((logits.argmax(-1)==target).all()),tokens=len(target)))
        return expert,dict(status='COMPLETE',binding=binding,steps=320,forwards=640,backwards=640,curve=curve,
            diagnostic_forwards=0 if reference else 10,final=final,Base_gradient=False),latest
    finally:hook.detach()


def references():
    part=int(os.environ['PARTITION']);gpu=TRAIN_GPUS[part]
    jobs=[(t,mode) for t in selected() for mode in ('ADAM','RAW')]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t,mode in jobs[part::6]:
            path=schedule(t,mode)
            if path.exists():assert c.read(path)['endpoint_exact'];continue
            expert,audit,latest=fit(runtime,t,'REFERENCE_'+mode)
            old=p.initial(t) if mode=='ADAM' else PARENT/'private/weights'/t['anonymous_edit']/'RAW_DIR_540/FINAL.pt'
            original=p.load_state(old)['expert']
            assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in expert.state_dict().items()), 'Reference endpoint mismatch'
            audit.update(endpoint_exact=True,expert_order=t['order'],mode=mode);c.write(path,audit)
            d.consume_resume(latest);del expert
    p.done('REFERENCE_'+str(part))


def train():
    assert (RUN/'private/MECHANICAL_COMPLETE.json').exists()
    part=int(os.environ['PARTITION']);gpu=TRAIN_GPUS[part]
    jobs=[(t,arm) for t in selected() for arm in ARMS]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t,arm in jobs[part::6]:
            dest=point(t,arm)
            if (dest.parent/'TRAINING.json').exists():continue
            expert,audit,latest=fit(runtime,t,arm)
            c.save(dest,dict(expert={k:v.detach().cpu() for k,v in expert.state_dict().items()},binding=audit['binding'],state_hash=c.state_hash(expert)))
            saved=p.load_state(dest);assert all(torch.equal(saved['expert'][k],v.detach().cpu()) for k,v in expert.state_dict().items())
            audit.update(save_load_exact=True,parameters=7168,final_state_hash=c.state_hash(expert));c.write(dest.parent/'TRAINING.json',audit)
            d.consume_resume(latest);del expert
    p.done('TRAIN_'+str(part))


def controller():
    import pipeline
    assert (RUN/'private/PLAN_COMPLETE.json').exists()
    pipeline.wait([pipeline.launch('functionmatch.py','function_reference',g,i) for i,g in enumerate(TRAIN_GPUS) if not (RUN/'private'/f'REFERENCE_{i}.json').exists()])
    refs=[c.read(schedule(t,mode)) for t in selected() for mode in ('ADAM','RAW')]
    assert len(refs)==16 and all(x['endpoint_exact'] and len(x['curve'])==320 for x in refs)
    c.write(RUN/'private/ENDPOINT_PARITY.json',dict(status='PASS',exact_all_cores=True,reference_trajectories=16,extra_steps=5120,extra_forwards=10240,extra_backwards=10240))
    p.done('MECHANICAL_COMPLETE')
    pipeline.wait([pipeline.launch('functionmatch.py','function_train',g,i) for i,g in enumerate(TRAIN_GPUS) if not (RUN/'private'/f'TRAIN_{i}.json').exists()])
    pipeline.wait([pipeline.launch('functionmatch.py','function_eval',g,i) for i,g in enumerate(EVAL_GPUS) if not (RUN/'private'/f'EVAL_{i}.json').exists()])
    assert all(Path(x['path']).exists() for x in c.read(RUN/'private/CONSUMERS.json'))
    p.done('GENERATION_COMPLETE');pipeline.wait([pipeline.launch('function_queue.py','function_ingest')])
    root=RUN/'private/judge_function_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('function_report.py','function_report')])


if __name__=='__main__':
    try:{'function_plan':plan,'function_reference':references,'function_train':train,'function_eval':evaluate,'function_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise
