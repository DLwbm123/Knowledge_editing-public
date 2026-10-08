"""Training-only feasible contraction of existing TT writes; no replay or SGD."""
import os
import sqlite3
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F
import scoped

p,c,RUN=scoped.p,scoped.c,scoped.RUN
PARENT=Path(os.environ['SCOPED_PARENT'])
ARMS=('NATIVE','NATIVE_FIT')
LABELS=('W0','CE192','SCOPE_E')+ARMS
CAL_GPUS=(4,5)
EVAL_GPUS=(2,3,4,5,6,7)
TAU=.001
ITERATIONS=12


def selected():return p.tasks()[:8]
def point(t,arm):return RUN/'private/weights'/t['anonymous_edit']/arm/'FINAL.pt'
def output(arm,qid):return RUN/'private/outputs'/arm/(c.digest(qid)+'.json')


def feasible(values,indices):
    return all(values[i]['nll']<=TAU and values[i]['all_argmax_correct'] for i in indices)


def bracket(test,zero,one):
    if not one:return 1.,'BASELINE_INFEASIBLE',[]
    if zero:return 0.,'ZERO_FEASIBLE',[]
    low,high=0.,1.;trace=[]
    for step in range(ITERATIONS):
        alpha=(low+high)/2;ok=test(alpha);trace.append(dict(step=step+1,alpha=alpha,feasible=ok))
        if ok:high=alpha
        else:low=alpha
    return high,'FEASIBLE_BRACKET',trace


def measures(logits,labels):
    target=labels[:,1:];mask=target!=-100;target=target[mask]
    scores=logits[:,:-1].float()[mask]
    assert len(target)>0 and torch.isfinite(scores).all()
    return dict(nll=float(F.cross_entropy(scores,target)),all_argmax_correct=bool((scores.argmax(-1)==target).all()),tokens=len(target))


def selfcheck():
    alpha,status,trace=bracket(lambda x:x>=.35,False,True)
    assert .35<=alpha<=.35+2**-ITERATIONS and len(trace)==ITERATIONS
    assert bracket(lambda x:False,False,False)==(1.,'BASELINE_INFEASIBLE',[])
    assert bracket(lambda x:True,True,True)==(0.,'ZERO_FEASIBLE',[])
    logits=torch.zeros(1,4,5);labels=torch.tensor([[-100,-100,2,4]])
    logits[0,1,2]=12;logits[0,2,4]=12
    assert feasible([measures(logits,labels)],[0])
    logits[0,2,3]=20;assert not feasible([measures(logits,labels)],[0])
    expert=p.tr.TT4(17,8,8)
    with torch.no_grad():expert.G1.normal_(0,.01)
    initial=expert.G2.detach().clone();x=torch.randn(3,14336);y=expert.residual(x)
    with torch.no_grad():expert.G2.copy_(initial*.37)
    assert torch.allclose(expert.residual(x),y*.37,atol=1e-5,rtol=1e-5)
    with torch.no_grad():expert.G2.zero_()
    assert torch.count_nonzero(expert.residual(x))==0
    return dict(status='PASS',bracket_endpoints=True,shifted_target_mask=True,TT_scaling=True,zero_residual=True)


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
    lock=dict(arms=list(ARMS),primary='NATIVE_FIT',experts=8,selected_edits=list(chosen),
        tau=TAU,iterations=ITERATIONS,optimizer_steps=0,calibration_forwards_upper=752,
        new_generations=220,Judge_upper_bound=220,consumers=8621,training_roles=['native','FIT'],replay=False,
        calibration_GPUs=list(CAL_GPUS),inference_GPUs=list(EVAL_GPUS),parameters=7168,final_TT=16)
    c.write(RUN/'private/MINWRITE_LOCK.json',lock);c.write(RUN/'private/JOBS.json',jobs);c.write(RUN/'private/CONSUMERS.json',consumers)
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**{k:v for k,v in lock.items() if k!='selected_edits'},selfcheck=checks,Base_lookup_unique_inputs=len(unique)))
    p.done('PLAN_COMPLETE')


def calibrate():
    part=int(os.environ['PARTITION']);gpu=CAL_GPUS[part];lock=c.read(RUN/'private/MINWRITE_LOCK.json')
    assert lock['tau']==TAU and lock['iterations']==ITERATIONS
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t in selected()[part::2]:
            if all((point(t,a).parent/'CALIBRATION.json').exists() for a in ARMS):continue
            initial=p.load_state(p.initial(t))['expert'];expert=p.expert(initial,t['seed'],runtime.device).requires_grad_(False)
            g2=expert.G2.detach().clone();hook=scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
            records=[c.record(t)]+[replace(c.record(t),question=q) for q in t['fit_questions']]
            batches=[runtime.build_edit_batch(x) for x in records];forwards=0
            def evaluate(alpha,indices):
                nonlocal forwards
                with torch.no_grad():expert.G2.copy_(g2*alpha)
                values={}
                for i in indices:
                    batch=batches[i];target=batch.labels[batch.labels!=-100]
                    assert target.tolist()==list(batch.target_token_ids) and int(target[-1])==runtime.adapter.tokenizer.eos_token_id
                    hook.set_teacher_routing(batch.labels)
                    with torch.inference_mode():out=runtime.model(**batch.forward_kwargs())
                    item=measures(out.logits,batch.labels)
                    assert abs(float(out.loss)-item['nll'])<1e-6
                    values[i]=item;forwards+=1
                return values
            try:
                one=evaluate(1.,range(5))
                assert all(torch.equal(v.cpu(),initial[k]) for k,v in expert.state_dict().items())
                zero=evaluate(0.,range(5))
                # At alpha=0 the hook must be identical to editing fully disabled.
                hook.set_teacher_routing(batches[0].labels)
                with torch.inference_mode():with_zero=runtime.model(**batches[0].forward_kwargs()).logits
                hook.detach()
                with torch.inference_mode():without=runtime.model(**batches[0].forward_kwargs()).logits
                assert torch.equal(with_zero,without);forwards+=2;del with_zero,without
                hook.attach()
                for arm in ARMS:
                    indices=[0] if arm=='NATIVE' else list(range(5));probes=[]
                    def test(alpha):
                        values=evaluate(alpha,indices);probes.append(dict(alpha=alpha,values=values));return feasible(values,indices)
                    alpha,status,trace=bracket(test,feasible(zero,indices),feasible(one,indices))
                    final=evaluate(alpha,range(5))
                    assert status=='BASELINE_INFEASIBLE' or feasible(final,indices)
                    assert all(torch.equal(expert.state_dict()[k].cpu(),initial[k]) for k in ('G1','G3','G4'))
                    assert all(v.grad is None for v in expert.parameters()) and all(v.grad is None for v in runtime.model.parameters())
                    binding=dict(arm=arm,alpha=alpha,source_W0=c.state_hash(p.expert(initial,t['seed'])),
                        execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),scientific_lock=c.digest(lock),roles=['native'] if arm=='NATIVE' else ['native','FIT'])
                    c.save(point(t,arm),dict(expert={k:v.detach().cpu() for k,v in expert.state_dict().items()},binding=binding,state_hash=c.state_hash(expert)))
                    saved=p.load_state(point(t,arm));assert all(torch.equal(saved['expert'][k],expert.state_dict()[k].cpu()) for k in initial)
                    c.write(point(t,arm).parent/'CALIBRATION.json',dict(status=status,binding=binding,initial=one,zero=zero,final=final,trace=trace,probes=probes,
                        no_optimizer_steps=True,zero_hook_parity=True,fixed_cores_unchanged=True,parameters=7168))
                    print('CALIBRATED',t['order'],arm,alpha,status,flush=True)
                assert forwards<=94
                c.write(RUN/'private/calibration_cost'/f"{t['order']}.json",dict(forwards=forwards,optimizer_steps=0))
            finally:hook.detach()
    p.done('CALIBRATION_'+str(part))


def evaluate():
    part=int(os.environ['PARTITION']);gpu=EVAL_GPUS[part];jobs=c.read(RUN/'private/JOBS.json')
    bindings=c.read(RUN/'private/EVAL_BINDINGS.json');base={q:scoped.base_tokens(row,bindings) for q,row in {j['row']['query_id']:j['row'] for j in jobs}.items()}
    ts={t['edit_id']:t for t in p.tasks()};chosen={t['edit_id'] for t in selected()};routes=c.read(RUN/'private/ROUTES.json')
    source=c.read(RUN/'private/GPU_SOURCE_VERSION.json');lock=c.digest(c.read(RUN/'private/MINWRITE_LOCK.json'))
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
            phase=dict(arm=j['arm'],node=0,prefix=146,slot=0,weights=weights[arm],execution=source,minwrite_lock=lock)
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
    pipeline.wait([pipeline.launch('minwrite.py','minwrite_calibrate',g,i) for i,g in enumerate(CAL_GPUS)])
    pipeline.wait([pipeline.launch('minwrite.py','minwrite_eval',g,i) for i,g in enumerate(EVAL_GPUS)])
    assert all(Path(x['path']).exists() for x in c.read(RUN/'private/CONSUMERS.json'))
    p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('minwrite_queue.py','minwrite_ingest')])
    root=RUN/'private/judge_minwrite_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('minwrite_report.py','minwrite_report')])


if __name__=='__main__':
    try:
        {'minwrite_plan':plan,'minwrite_calibrate':calibrate,'minwrite_eval':evaluate,'minwrite_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
