"""Fixed initial TT curriculum and consolidation ablation; native/FIT only."""
import os
import sqlite3
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import scoped

p,c,RUN=scoped.p,scoped.c,scoped.RUN
PARENT=Path(os.environ['SCOPED_PARENT'])
ARMS=('STAGED_220','JOINT_220')
LABELS=('W0','CE192','SCOPE_E')+ARMS
TRAIN_GPUS=(4,5)
EVAL_GPUS=(2,3,4,5,6,7)
PRIMARY='JOINT_220'


def selected():return p.tasks()[:8]
def point(t,arm):return RUN/'private/weights'/t['anonymous_edit']/arm/'FINAL.pt'
def output(arm,qid):return RUN/'private/outputs'/arm/(c.digest(qid)+'.json')


def phase_indices(arm,phase,step):
    assert arm in ARMS and phase in ('native','A2','W0') and step>=1
    return [0] if arm=='STAGED_220' and phase=='native' else [0,1+(step-1)%4]


def selfcheck():
    assert all(phase_indices('STAGED_220','native',s)==[0] for s in range(1,141))
    assert all(phase_indices('JOINT_220','native',s)==[0,1+(s-1)%4] for s in range(1,141))
    assert all(phase_indices(a,'A2',s)==[0,1+(s-1)%4] for a in ARMS for s in range(1,81))
    assert [sum(i in phase_indices('JOINT_220','native',s) for s in range(1,141)) for i in range(1,5)]==[35]*4
    return dict(status='PASS',identical_phase_lengths_and_resets=True,paired_fit_schedule=True,original_STAGED_indices=True)


def plan():
    assert (PARENT/'private/REPORT_COMPLETE.json').exists()
    checks=selfcheck();db=sqlite3.connect('file:'+str(PARENT/'private/judge_scoped_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
    old={(a,m,q):path for a,m,q,path in db.execute('SELECT method,mode,query_id,path FROM consumer')};db.close()
    routes=c.read(RUN/'private/ROUTES.json');chosen={t['edit_id'] for t in selected()}
    consumers=[];jobs=[]
    def add(arm,mode,row,expert,path=None):
        dest=str(path or output(arm,row['query_id']))
        consumers.append(dict(arm=arm,mode=mode,query_id=row['query_id'],path=dest,produced_arm=arm))
        if path is None:jobs.append(dict(arm=arm,mode=mode,row=row,expert=expert,path=dest))
    for q,row in p.queries(146).items():
        expert=routes[q]['effective_expert'];affected=expert in chosen
        for label,parent_label in [('W0','W0'),('CE192','CE192'),('SCOPE_E','E_SPLIT_SCOPE_GATE_s192')]:
            add(label,'bank_R0',row,expert,old[parent_label,'bank_R0',q])
        for arm in ARMS:add(arm,'bank_R0',row,expert,None if affected else old['W0','bank_R0',q])
    for row in c.read(RUN/'private/SOURCE_CHECK_ROWS.json'):
        q=row['query_id'];expert=p.tasks()[row['forced_expert_index']]['edit_id'];affected=expert in chosen
        for label,parent_label in [('BASE','BASE'),('W0','W0'),('CE192','CE192'),('SCOPE_E','E_SPLIT_SCOPE_GATE_s192')]:
            add(label,'forced_source_CHECK',row,expert if label!='BASE' else None,old[parent_label,'forced_source_CHECK',q])
        for arm in ARMS:add(arm,'forced_source_CHECK',row,expert,None if affected else old['W0','forced_source_CHECK',q])
        expert=routes[q]['effective_expert'];affected=expert in chosen
        for label,parent_label in [('BASE_NAT','BASE_NAT'),('W0_NAT','W0_NAT'),('SCOPE_E_NAT','E_SPLIT_SCOPE_GATE_s192_NAT')]:
            add(label,'natural_source_CHECK',row,expert if label!='BASE_NAT' else None,old[parent_label,'natural_source_CHECK',q])
        for arm in ARMS:add(arm+'_NAT','natural_source_CHECK',row,expert,None if affected else old['W0_NAT','natural_source_CHECK',q])
    assert len(jobs)==220 and len(consumers)==8621 and len({j['path'] for j in jobs})==220
    assert all(Path(x['path']).exists() for x in consumers if not Path(x['path']).is_relative_to(RUN/'private/outputs'))
    # Resolve every Base reference before starting any GPU work.
    bindings=c.read(RUN/'private/EVAL_BINDINGS.json')
    unique={j['row']['query_id']:j['row'] for j in jobs}
    assert all(scoped.base_tokens(row,bindings) for row in unique.values())
    lock=dict(arms=list(ARMS),primary=PRIMARY,experts=8,selected_edits=list(chosen),
        phases=[['native',140],['A2',80]],optimizer_steps=3840,training_forwards=6560,
        training_backwards=6560,endpoint_rebuild_extra_steps=320,initial_state='seeded_zero_TT88',
        new_generations=220,Judge_upper_bound=220,consumers=8621,training_roles=['native','FIT'],replay=False,
        training_GPUs=list(TRAIN_GPUS),inference_GPUs=list(EVAL_GPUS),parameters=7168,final_TT=16)
    c.write(RUN/'private/INITIALWRITE_LOCK.json',lock);c.write(RUN/'private/JOBS.json',jobs);c.write(RUN/'private/CONSUMERS.json',consumers)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**{k:v for k,v in lock.items() if k!='selected_edits'},selfcheck=checks,Base_lookup_unique_inputs=len(unique)))
    p.done('PLAN_COMPLETE')


def phases(runtime,t,arm,expert,names):
    from m3bench_repro.editors.llava_runtime import seed_everything
    records=[c.record(t)]+[replace(c.record(t),question=q) for q in t['fit_questions']]
    batches=[runtime.build_edit_batch(x) for x in records]
    assert all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in batches)
    hook=scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
    latest=point(t,arm).parent/('resume_'+names[0][0]+'_'+names[-1][0]+'.pt')
    binding=dict(arm=arm,seed=t['seed'],phases=names,execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'))
    saved=p.load_state(latest) if latest.exists() else None
    curves=saved['curve'] if saved else []
    if saved:
        assert saved['binding']==binding;expert.load_state_dict(saved['expert'])
    try:
        for phase_index,(phase,steps) in enumerate(names):
            if saved and phase_index<saved['phase_index']:continue
            seed_everything(t['seed'])
            opt=torch.optim.AdamW(expert.parameters(),lr=1e-3,weight_decay=0) if phase!='W0' else p.optimizer(expert,runtime.model)
            start=0
            if saved and phase_index==saved['phase_index']:
                opt.load_state_dict(saved['optimizer']);c.restore_rng(saved);start=saved['step']
            for step in range(start+1,steps+1):
                c.budget();opt.zero_grad(set_to_none=True);indices=phase_indices(arm,phase,step);losses=[]
                before={k:v.detach().clone() for k,v in expert.state_dict().items()}
                for i in indices:
                    hook.set_teacher_routing(batches[i].labels);loss=runtime.compute_loss(batches[i]);assert torch.isfinite(loss)
                    (loss/len(indices)).backward();losses.append(float(loss.detach()))
                norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);assert torch.isfinite(norm)
                assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in expert.parameters())
                assert not any(x.grad is not None for x in runtime.model.parameters())
                opt.step();assert all(torch.isfinite(x).all() for x in expert.parameters())
                curves.append(dict(phase=phase,step=step,indices=indices,CE=losses,gradient_norm=float(norm),
                    update_norms={k:float((v-before[k]).norm()) for k,v in expert.state_dict().items()},forwards=len(indices),backwards=len(indices)))
                if step%20==0 or step==steps:
                    c.save(latest,dict(binding=binding,expert=expert.state_dict(),optimizer=opt.state_dict(),phase_index=phase_index,
                        step=step,curve=curves,**c.rng()))
                    print('TRAIN',t['order'],arm,phase,step,flush=True)
    finally:hook.detach()
    return curves


def save_final(runtime,t,arm,expert,curve):
    binding=dict(arm=arm,source='SEEDED_TT88_ZERO',seed=t['seed'],steps=220,
        execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),scientific_lock=c.digest(c.read(RUN/'private/INITIALWRITE_LOCK.json')),
        roles=['native','FIT'],replay=False,optimizer_resets=[0,140])
    c.save(point(t,arm),dict(expert={k:v.detach().cpu() for k,v in expert.state_dict().items()},binding=binding,state_hash=c.state_hash(expert)))
    saved=p.load_state(point(t,arm));assert all(torch.equal(saved['expert'][k],v.detach().cpu()) for k,v in expert.state_dict().items())
    records=[c.record(t)]+[replace(c.record(t),question=q) for q in t['fit_questions']]
    hook=scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach();diagnostics=[]
    try:
        for row in records:
            b=runtime.build_edit_batch(row);hook.set_teacher_routing(b.labels)
            with torch.inference_mode():d=runtime.model(**b.forward_kwargs())
            mask=b.labels[:,1:]!=-100;target=b.labels[:,1:][mask];logits=d.logits[:,:-1].float()[mask]
            diagnostics.append(dict(nll=float(d.loss),all_argmax_correct=bool((logits.argmax(-1)==target).all()),tokens=len(target)))
    finally:hook.detach()
    c.write(point(t,arm).parent/'TRAINING.json',dict(status='COMPLETE',binding=binding,curve=curve,steps=220,
        forwards=sum(x['forwards'] for x in curve),backwards=sum(x['backwards'] for x in curve),diagnostic_forwards=5,
        final=diagnostics,Base_gradient=False,save_load_exact=True,parameters=7168,final_state_hash=c.state_hash(expert)))
    removed=[]
    for path in point(t,arm).parent.glob('resume_*.pt'):
        assert not path.is_symlink();removed.append(dict(name=path.name,bytes=path.stat().st_size));path.unlink()
    c.write(point(t,arm).parent/'RESUME_DELETION.json',dict(reason='Final weight and trajectory saved; resume state no longer needed',files=removed))


def mechanical():
    # Rebuild one original endpoint before admitting any remaining trajectory.
    t=selected()[0]
    with p.lease(TRAIN_GPUS[0]):
        runtime,_=c.load(TRAIN_GPUS[0]);expert=p.tr.TT4(t['seed'],8,8).to(runtime.device)
        assert torch.count_nonzero(expert.G1)==0
        curve=phases(runtime,t,'STAGED_220',expert,[('native',140),('A2',80)])
        at220={k:v.detach().clone() for k,v in expert.state_dict().items()}
        tail=phases(runtime,t,'STAGED_220',expert,[('W0',320)])
        original=p.load_state(p.initial(t))['expert']
        assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in expert.state_dict().items()),'Original 540-step endpoint differs; do not admit'
        c.write(RUN/'private/ENDPOINT_PARITY.json',dict(status='PASS',expert_order=t['order'],exact_all_cores=True,
            reconstruction_steps=540,extra_steps=320,extra_forwards=640,extra_backwards=640,tail_curve=tail))
        expert.load_state_dict(at220);save_final(runtime,t,'STAGED_220',expert,curve)
    p.done('MECHANICAL_COMPLETE')


def train():
    assert (RUN/'private/MECHANICAL_COMPLETE.json').exists()
    part=int(os.environ['PARTITION']);gpu=TRAIN_GPUS[part]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t in selected()[part::2]:
            for arm in ARMS:
                if (point(t,arm).parent/'TRAINING.json').exists():continue
                expert=p.tr.TT4(t['seed'],8,8).to(runtime.device)
                curve=phases(runtime,t,arm,expert,[('native',140),('A2',80)])
                save_final(runtime,t,arm,expert,curve);del expert
    p.done('TRAIN_'+str(part))


def evaluate():
    part=int(os.environ['PARTITION']);gpu=EVAL_GPUS[part];jobs=c.read(RUN/'private/JOBS.json')
    bindings=c.read(RUN/'private/EVAL_BINDINGS.json');base={q:scoped.base_tokens(row,bindings) for q,row in {j['row']['query_id']:j['row'] for j in jobs}.items()}
    ts={t['edit_id']:t for t in p.tasks()};chosen={t['edit_id'] for t in selected()};routes=c.read(RUN/'private/ROUTES.json')
    source=c.read(RUN/'private/GPU_SOURCE_VERSION.json');lock=c.digest(c.read(RUN/'private/INITIALWRITE_LOCK.json'))
    weights={};states={}
    for arm in ARMS:
        weights[arm]=dict(c.read(RUN/'private/REFERENCE_WEIGHTS.json'))
        for t in selected():
            d=p.load_state(point(t,arm));states[arm,t['edit_id']]=d['expert'];weights[arm][t['edit_id']]=dict(path=str(point(t,arm)),hash=d['state_hash'])
    with p.lease(gpu):
        runtime,_=c.load(gpu);loaded={}
        q=[q for q in p.queries(146) if routes[q]['effective_expert'] is not None and routes[q]['effective_expert'] not in chosen][part]
        old=c.read(scoped.historical('W0',q));t=ts[old['effective_expert']]
        control=p.expert(p.load_state(p.initial(t))['expert'],t['seed'],runtime.device).requires_grad_(False)
        hook=scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),control);hook.attach()
        try:
            row=old['binding']['input'];raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            with torch.inference_mode(),hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            assert list(g.raw_token_ids)==old['R0']['raw_token_ids']
        finally:hook.detach()
        p.done('NEGATIVE_CONTROL_'+str(part),dict(original_tokens_equal=True))
        for i,j in enumerate(jobs[part::len(EVAL_GPUS)]):
            c.budget();dest=Path(j['path']);row=j['row'];q=row['query_id'];arm=j['arm'].removesuffix('_NAT')
            phase=dict(arm=j['arm'],node=0,prefix=146,slot=0,weights=weights[arm],execution=source,initialwrite_lock=lock)
            if dest.exists():assert c.read(dest)['binding']['phase']==phase;continue
            old=c.read(scoped.historical('W0',q));bind=old['binding']['judge_input']
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            assert raw['image_sha256']==bind['image_sha256'] and raw['input_ids'][0].tolist()==bind['prompt_ids']
            assert raw['attention_mask'][0].tolist()==bind['attention_mask'] and runtime.generation_config==bind['generation']
            key=arm,j['expert']
            if key not in loaded:loaded[key]=p.expert(states[key],ts[j['expert']]['seed'],runtime.device).requires_grad_(False)
            hook=scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),loaded[key]);hook.attach()
            try:
                with torch.inference_mode(),hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            finally:hook.detach()
            actual=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids))
            c.write(dest,dict(binding=dict(input=row,judge_input=bind,phase=phase,arm=j['arm'],mode=j['mode']),R0=actual,
                effective_expert=j['expert'],route=routes[q],active_target=old.get('active_target',False),U_KL=None,
                Base_token_consistency=actual['raw_token_ids']==base[q]))
            if i%10==0:print('EVAL',part,i,len(jobs[part::len(EVAL_GPUS)]),flush=True)
    p.done('EVAL_'+str(part))


def controller():
    import pipeline
    assert (RUN/'private/PLAN_COMPLETE.json').exists()
    if not (RUN/'private/MECHANICAL_COMPLETE.json').exists():
        pipeline.wait([pipeline.launch('initialwrite.py','initialwrite_mechanical',TRAIN_GPUS[0],0)])
    pipeline.wait([pipeline.launch('initialwrite.py','initialwrite_train',g,i) for i,g in enumerate(TRAIN_GPUS) if not (RUN/'private'/('TRAIN_'+str(i)+'.json')).exists()])
    pipeline.wait([pipeline.launch('initialwrite.py','initialwrite_eval',g,i) for i,g in enumerate(EVAL_GPUS) if not (RUN/'private'/('EVAL_'+str(i)+'.json')).exists()])
    assert all(Path(x['path']).exists() for x in c.read(RUN/'private/CONSUMERS.json'))
    p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('initialwrite_queue.py','initialwrite_ingest')])
    root=RUN/'private/judge_initialwrite_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('initialwrite_report.py','initialwrite_report')])


if __name__=='__main__':
    try:
        {'initialwrite_plan':plan,'initialwrite_mechanical':mechanical,'initialwrite_train':train,'initialwrite_eval':evaluate,'initialwrite_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
