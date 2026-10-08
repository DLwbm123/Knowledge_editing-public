"""Fixed original tail: Adam step length with raw-gradient group direction."""
import os
import sqlite3
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import scoped

p,c,RUN=scoped.p,scoped.c,scoped.RUN
PARENT=Path(os.environ['INITIALWRITE_PARENT'])
ARMS=('RAW_DIR_540',)
LABELS=('W0','STAGED_220','JOINT_220')+ARMS
TRAIN_GPUS=(4,5)
EVAL_GPUS=(2,3,4,5,6,7)
PRIMARY='RAW_DIR_540'


def selected():return p.tasks()[:8]
def point(t,arm):return RUN/'private/weights'/t['anonymous_edit']/arm/'FINAL.pt'
def output(arm,qid):return RUN/'private/outputs'/arm/(c.digest(qid)+'.json')


def apply_direction(opt,before,mode):
    """Adam moments advance on current gradients; only its parameter write is replaced."""
    assert mode in ('RAW','ADAM')
    records=[]
    for group in opt.param_groups:
        ps=group['params'];old=[before[id(x)] for x in ps]
        gradient=torch.cat([x.grad.detach().double().flatten() for x in ps])
        candidate=torch.cat([(x.detach()-b).double().flatten() for x,b in zip(ps,old)])
        gn=float(gradient.norm());dn=float(candidate.norm());fallback=gn==0
        if mode=='RAW' and not fallback:
            with torch.no_grad():
                for x,b in zip(ps,old):x.copy_(b-x.grad*(dn/gn))
        actual=torch.cat([(x.detach()-b).double().flatten() for x,b in zip(ps,old)])
        norm=float(actual.norm());bound=4*torch.finfo(ps[0].dtype).eps*(float(torch.cat([b.double().flatten() for b in old]).norm())+dn)
        assert abs(norm-dn)<=bound+1e-14 and torch.isfinite(actual).all()
        records.append(dict(raw_gradient_norm=gn,Adam_candidate_norm=dn,actual_norm=norm,roundoff_bound=bound,
            Adam_negative_gradient_cosine=float(torch.dot(candidate,-gradient)/(candidate.norm()*gradient.norm())) if gn and dn else None,
            actual_negative_gradient_cosine=float(torch.dot(actual,-gradient)/(actual.norm()*gradient.norm())) if gn and norm else None,
            zero_gradient_Adam_fallback=fallback))
    return records


def selfcheck():
    ps=[torch.nn.Parameter(torch.tensor([2.,-3.,.5])),torch.nn.Parameter(torch.tensor([.1,-.2]))]
    opt=torch.optim.Adam([{'params':[ps[0]],'lr':1e-3},{'params':[ps[1]],'lr':1e-4}])
    for x in ps:x.grad=torch.arange(1,len(x)+1,dtype=x.dtype)
    before={id(x):x.detach().clone() for x in ps};opt.step();rows=apply_direction(opt,before,'RAW')
    assert all(x['actual_negative_gradient_cosine']>.99999 for x in rows)
    assert all(abs(x['actual_norm']-x['Adam_candidate_norm'])<=x['roundoff_bound'] for x in rows)
    for x in ps:x.grad=torch.zeros_like(x)
    before={id(x):x.detach().clone() for x in ps};opt.step();adam=[x.detach().clone() for x in ps];rows=apply_direction(opt,before,'RAW')
    assert all(torch.equal(x,a) for x,a in zip(ps,adam)) and all(x['zero_gradient_Adam_fallback'] for x in rows)
    return dict(status='PASS',group_step_norms_match_with_roundoff=True,raw_gradient_direction=True,zero_gradient_fallback_preserves_Adam=True)


def plan():
    assert (PARENT/'private/REPORT_COMPLETE.json').exists()
    checks=selfcheck();db=sqlite3.connect('file:'+str(PARENT/'private/judge_initialwrite_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
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
        for label,parent_label in [(x,x) for x in ('BASE','W0','STAGED_220','JOINT_220')]:
            add(label,'forced_source_CHECK',row,expert if label!='BASE' else None,old[parent_label,'forced_source_CHECK',q])
        for arm in ARMS:add(arm,'forced_source_CHECK',row,expert,None if affected else old['W0','forced_source_CHECK',q])
        expert=routes[q]['effective_expert'];affected=expert in chosen
        for label,parent_label in [(x,x) for x in ('BASE_NAT','W0_NAT','STAGED_220_NAT','JOINT_220_NAT')]:
            add(label,'natural_source_CHECK',row,expert if label!='BASE_NAT' else None,old[parent_label,'natural_source_CHECK',q])
        for arm in ARMS:add(arm+'_NAT','natural_source_CHECK',row,expert,None if affected else old['W0_NAT','natural_source_CHECK',q])
    assert len(jobs)==110 and len(consumers)==7012 and len({j['path'] for j in jobs})==110
    assert all(Path(x['path']).exists() for x in consumers if not Path(x['path']).is_relative_to(RUN/'private/outputs'))
    # Resolve every Base reference before starting any GPU work.
    bindings=c.read(RUN/'private/EVAL_BINDINGS.json')
    unique={j['row']['query_id']:j['row'] for j in jobs}
    assert all(scoped.base_tokens(row,bindings) for row in unique.values())
    lock=dict(arms=list(ARMS),primary=PRIMARY,experts=8,selected_edits=list(chosen),
        phases=[['original_tail',320]],optimizer_steps=2880,training_forwards=5760,
        training_backwards=5760,endpoint_rebuild_extra_steps=320,initial_state='original_STAGED_220',
        new_generations=110,Judge_upper_bound=110,consumers=7012,training_roles=['native','FIT'],replay=False,
        training_GPUs=list(TRAIN_GPUS),inference_GPUs=list(EVAL_GPUS),parameters=7168,final_TT=8)
    c.write(RUN/'private/DIRECTION_LOCK.json',lock);c.write(RUN/'private/JOBS.json',jobs);c.write(RUN/'private/CONSUMERS.json',consumers)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**{k:v for k,v in lock.items() if k!='selected_edits'},selfcheck=checks,Base_lookup_unique_inputs=len(unique)))
    p.done('PLAN_COMPLETE')


def start_state(t):
    return p.load_state(PARENT/'private/weights'/t['anonymous_edit']/'STAGED_220/FINAL.pt')['expert']


def fit(runtime,t,mode):
    from m3bench_repro.editors.llava_runtime import seed_everything
    expert=p.expert(start_state(t),t['seed'],runtime.device)
    records=[c.record(t)]+[replace(c.record(t),question=q) for q in t['fit_questions']]
    batches=[runtime.build_edit_batch(x) for x in records]
    assert all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in batches)
    hook=scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
    latest=RUN/'private/resume'/f"{t['order']}_{mode}.pt"
    binding=dict(mode=mode,seed=t['seed'],steps=320,start=c.state_hash(expert),roles=['native','FIT'],replay=False,
        execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),scientific_lock=c.digest(c.read(RUN/'private/DIRECTION_LOCK.json')))
    saved=p.load_state(latest) if latest.exists() else None
    seed_everything(t['seed']);opt=p.optimizer(expert,runtime.model);curve=[];start=0;diagnostic_forwards=0
    try:
        if mode=='RAW':
            x=p.tr.activations(runtime,hook,batches,[]);f0=p.tr.functional(expert,x);previous=f0;path=0.;diagnostic_forwards=5
        if saved:
            assert saved['binding']==binding;expert.load_state_dict(saved['expert']);opt.load_state_dict(saved['optimizer']);c.restore_rng(saved)
            curve=saved['curve'];start=saved['step']
            if mode=='RAW':previous=p.tr.functional(expert,x);path=saved['function_path']
        for step in range(start+1,321):
            c.budget();opt.zero_grad(set_to_none=True);before={id(x):x.detach().clone() for x in expert.parameters()};losses=[]
            for i in [0,1+(step-1)%4]:
                hook.set_teacher_routing(batches[i].labels);loss=runtime.compute_loss(batches[i]);assert torch.isfinite(loss)
                (.5*loss).backward();losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);assert torch.isfinite(norm)
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in expert.parameters())
            assert not any(x.grad is not None for x in runtime.model.parameters())
            opt.step();groups=apply_direction(opt,before,mode)
            row=dict(step=step,CE=losses,gradient_norm=float(norm),groups=groups,forwards=2,backwards=2)
            if mode=='RAW':
                current=p.tr.functional(expert,x);path+=float((current-previous).norm());previous=current
                row.update(function_path=path,net_function_change=float((current-f0).norm()))
            curve.append(row)
            if step%20==0:
                c.save(latest,dict(binding=binding,expert=expert.state_dict(),optimizer=opt.state_dict(),step=step,curve=curve,
                    function_path=path if mode=='RAW' else None,**c.rng()))
                print('TRAIN',t['order'],mode,step,flush=True)
        diagnostics=[]
        if mode=='RAW':
            for b in batches:
                hook.set_teacher_routing(b.labels)
                with torch.inference_mode():d=runtime.model(**b.forward_kwargs())
                mask=b.labels[:,1:]!=-100;target=b.labels[:,1:][mask];logits=d.logits[:,:-1].float()[mask]
                diagnostics.append(dict(nll=float(d.loss),all_argmax_correct=bool((logits.argmax(-1)==target).all()),tokens=len(target)))
            diagnostic_forwards+=5
        return expert,dict(status='COMPLETE',binding=binding,steps=320,forwards=640,backwards=640,curve=curve,
            diagnostic_forwards=diagnostic_forwards,final=diagnostics,Base_gradient=False),latest
    finally:hook.detach()


def consume_resume(path):
    assert path.is_file() and not path.is_symlink();receipt=dict(name=path.name,bytes=path.stat().st_size,reason='Final state and trajectory durable; no remaining resume consumer')
    path.unlink();c.write(path.with_suffix('.DELETED.json'),receipt)


def mechanical():
    t=selected()[0]
    with p.lease(TRAIN_GPUS[0]):
        runtime,_=c.load(TRAIN_GPUS[0]);expert,audit,latest=fit(runtime,t,'ADAM')
        original=p.load_state(p.initial(t))['expert']
        assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in expert.state_dict().items()),'Original Adam tail endpoint differs'
        c.write(RUN/'private/ENDPOINT_PARITY.json',dict(status='PASS',expert_order=t['order'],exact_all_cores=True,
            reconstruction_steps=320,extra_steps=320,extra_forwards=640,extra_backwards=640,audit=audit))
        consume_resume(latest)
    p.done('MECHANICAL_COMPLETE')


def train():
    assert (RUN/'private/MECHANICAL_COMPLETE.json').exists()
    part=int(os.environ['PARTITION']);gpu=TRAIN_GPUS[part]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t in selected()[part::2]:
            arm=ARMS[0];dest=point(t,arm)
            if (dest.parent/'TRAINING.json').exists():continue
            expert,audit,latest=fit(runtime,t,'RAW')
            c.save(dest,dict(expert={k:v.detach().cpu() for k,v in expert.state_dict().items()},binding=audit['binding'],state_hash=c.state_hash(expert)))
            saved=p.load_state(dest);assert all(torch.equal(saved['expert'][k],v.detach().cpu()) for k,v in expert.state_dict().items())
            audit.update(save_load_exact=True,parameters=7168,final_state_hash=c.state_hash(expert));c.write(dest.parent/'TRAINING.json',audit)
            consume_resume(latest);del expert
    p.done('TRAIN_'+str(part))


def evaluate():
    part=int(os.environ['PARTITION']);gpu=EVAL_GPUS[part];jobs=c.read(RUN/'private/JOBS.json')
    bindings=c.read(RUN/'private/EVAL_BINDINGS.json');base={q:scoped.base_tokens(row,bindings) for q,row in {j['row']['query_id']:j['row'] for j in jobs}.items()}
    ts={t['edit_id']:t for t in p.tasks()};chosen={t['edit_id'] for t in selected()};routes=c.read(RUN/'private/ROUTES.json')
    source=c.read(RUN/'private/GPU_SOURCE_VERSION.json');lock=c.digest(c.read(RUN/'private/DIRECTION_LOCK.json'))
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
            phase=dict(arm=j['arm'],node=0,prefix=146,slot=0,weights=weights[arm],execution=source,direction_lock=lock)
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
        pipeline.wait([pipeline.launch('direction.py','direction_mechanical',TRAIN_GPUS[0],0)])
    pipeline.wait([pipeline.launch('direction.py','direction_train',g,i) for i,g in enumerate(TRAIN_GPUS) if not (RUN/'private'/('TRAIN_'+str(i)+'.json')).exists()])
    pipeline.wait([pipeline.launch('direction.py','direction_eval',g,i) for i,g in enumerate(EVAL_GPUS) if not (RUN/'private'/('EVAL_'+str(i)+'.json')).exists()])
    assert all(Path(x['path']).exists() for x in c.read(RUN/'private/CONSUMERS.json'))
    p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('direction_queue.py','direction_ingest')])
    root=RUN/'private/judge_direction_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('direction_report.py','direction_report')])


if __name__=='__main__':
    try:
        {'direction_plan':plan,'direction_mechanical':mechanical,'direction_train':train,'direction_eval':evaluate,'direction_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
