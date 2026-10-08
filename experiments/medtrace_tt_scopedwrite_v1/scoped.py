"""Bounded no-replay first-eight ScopedWrite pilot, using the native TT and frozen bank."""
import inspect
import os
import random
import time
import traceback
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import torch
import replay
import scoped_math as sm

p,c,RUN=replay.p,replay.c,replay.RUN
SOURCE=Path(os.environ['REPLAY_PARENT'])
GPUS=(4,5)
ARMS=sm.ARMS
NODES=(32,96,192)


def selected():return p.tasks()[:8]
def prepared(t):return RUN/'private/prepared'/f"{t['order']}.pt"
def node(t,arm,step):return RUN/'private/weights'/t['anonymous_edit']/arm/f'step{step:03d}.pt'
def output(label,qid):return RUN/'private/outputs'/label/(c.digest(qid)+'.json')
def label(arm,step):return arm+f'_s{step:03d}'


@lru_cache(maxsize=1)
def historical_index():return c.read(RUN/'private/HISTORICAL_OUTPUTS.json')


def historical(arm,qid):return Path(historical_index()[arm][qid])


def prepare():
    result=sm.selfcheck(p.tr.TT4)
    audits=[]
    for t in selected():
        e=p.expert(p.load_state(p.initial(t))['expert'],t['seed'])
        new,audit=sm.canonicalize(e.state_dict())
        assert type(e).__name__=='TT4' and sum(x.numel() for x in e.parameters())==7168
        inputs,outputs=e.parameter_groups()
        assert [id(x) for x in inputs]==[id(e.G3),id(e.G4)] and [id(x) for x in outputs]==[id(e.G1),id(e.G2)]
        audits.append(dict(expert_index=t['order'],**audit))
    c.write(RUN/'public/CPU_TESTS.json',result)
    c.write(RUN/'public/PARAMETER_AND_COORDINATE_AUDIT.json',dict(status='PASS',experts=audits,
        total_deployed_parameters=7168,trainable_parameters=2048,actual_class=p.tr.TT4.__qualname__))
    c.write(RUN/'private/RUNTIME_CLASS_BINDING.json',dict(class_file=inspect.getfile(p.tr.TT4),class_source=inspect.getsource(p.tr.TT4)))
    c.write(RUN/'public/DATA_ROLE_AUDIT.json',dict(status='PASS',training_roles=['native','FIT'],
        training_inputs_per_expert=5,source_replay_used=False,CHECK_used_for_training=False,
        coordinates='checkpoint only; invertible Cholesky gauge',history_key_dedup='image and question',
        case_independence=False,scope_qualification=False))


def fixed_expert(state,t,device):
    e=p.expert(state,t['seed'],device)
    for name,v in e.named_parameters():v.requires_grad_(name=='G2')
    return e


def question_features(runtime,record,a):
    from llava.constants import IMAGE_TOKEN_INDEX
    import structure
    batch=runtime.build_question_batch(replace(record,target='',official_rephrase=''))
    masks=structure.token_masks(batch.raw_input_ids[0],batch.inputs_embeds.shape[1],
        None if batch.attention_mask is None else batch.attention_mask[0],IMAGE_TOKEN_INDEX)
    values=[]
    def capture(_,args):
        x=args[0][0].float();x=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)
        z=x@a.T
        values.append(torch.stack([z[m].mean(0) for m in masks]).detach().cpu())
    handle=runtime.get_module(c.LAYER).register_forward_pre_hook(capture)
    try:
        with torch.no_grad():runtime.model(**batch.forward_kwargs())
    finally:handle.remove()
    assert len(values)==1
    _,visual,text=values[0];combined=.5*(visual+text)
    key=combined/combined.norm() if combined.norm()>1e-12 else torch.zeros_like(combined)
    return dict(key=key,gate=sm.gate(text[2:],visual[2:],bool(masks[1].any())),
        modality_norms=[float(text.norm()),float(visual.norm())],token_counts=[int(m.sum()) for m in masks],
        binding=c.digest([batch.image_sha256,record.question]),answer_used=False)


def update(runtime,hook,e,opt,batches,fit_index,history,features,arm):
    opt.zero_grad(set_to_none=True);losses={}
    for name,index in [('native',0),('fit',fit_index)]:
        batch=batches[index];hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch)
        assert torch.isfinite(loss);(.5*loss).backward();losses[name]=float(loss.detach())
    assert e.G2.grad is not None and torch.isfinite(e.G2.grad).all()
    assert all(v.grad is None for name,v in e.named_parameters() if name!='G2')
    assert all(v.grad is None for v in runtime.model.parameters())
    norm=torch.nn.utils.clip_grad_norm_([e.G2],1.);assert torch.isfinite(norm)
    before=e.G2.detach().clone();opt.step();candidate=(e.G2.detach()-before).view(512,4).cpu()
    gamma=1.
    if arm==ARMS[3]:gamma=.5*features[0]['gate']['gamma']+.125*sum(x['gate']['gamma'] for x in features[1:])
    if arm==ARMS[4]:gamma=.5*(features[0]['gate']['gamma']+features[fit_index]['gate']['gamma'])
    seen_before=len(history.keys)
    delta=sm.precondition(candidate,arm,history.joint,history.local,history.shared,gamma)
    with torch.no_grad():e.G2.copy_(before+delta.reshape_as(e.G2).to(e.G2.device))
    actual=(e.G2.detach()-before).view(512,4).cpu()
    assert torch.isfinite(actual).all()
    for index in (0,fit_index):
        f=features[index];history.absorb(f['binding'],f['key'],'native' if index==0 else 'FIT')
    return dict(loss=losses,preclip_norm=float(norm),gamma=gamma,candidate_norm=float(candidate.norm()),
        actual_norm=float(actual.norm()),local_norm=float(actual[:,:2].norm()),shared_norm=float(actual[:,2:].norm()),
        history_before=seen_before,history_after=len(history.keys),fit_index=fit_index)


def mechanical():
    part=int(os.environ['PARTITION']);gpu=GPUS[part];audits=[]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t in selected()[part::2]:
            original=p.load_state(p.initial(t))['expert'];state,audit=sm.canonicalize(original)
            records=[c.record(t)]+[replace(c.record(t),question=q) for q in t['fit_questions']]
            batches=[runtime.build_edit_batch(r) for r in records]
            assert all(b.target_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id for b in batches)
            e=p.expert(original,t['seed'],runtime.device);hook=replay.MedTraceLayerHook(runtime.get_module(c.LAYER),e);hook.attach()
            rows=[]
            try:
                for batch in batches:
                    hook.set_teacher_routing(batch.labels);e.load_state_dict(original);captures=[]
                    handle=runtime.get_module(c.LAYER).register_forward_pre_hook(lambda _,args:captures.append(args[0].detach().clone()))
                    try:
                        with torch.no_grad():old=runtime.model(**batch.forward_kwargs()).logits.detach().float()
                    finally:handle.remove()
                    with torch.no_grad():repeat=runtime.model(**batch.forward_kwargs()).logits.detach().float()
                    assert torch.equal(old,repeat),'Original precision is not repeatable'
                    h=captures[0];before=e.residual(h).detach();e.load_state_dict(state);after=e.residual(h).detach()
                    assert torch.allclose(before,after,atol=2e-5,rtol=2e-5),'Gauge residual parity failed'
                    with torch.no_grad():new=runtime.model(**batch.forward_kwargs()).logits.detach().float()
                    error=(new-old).norm()/old.norm().clamp_min(1e-12);maximum=(new-old).abs().max()
                    assert error<=1e-3 and maximum<=.25,'Gauge logits parity failed; do not relax tolerance'
                    rows.append(dict(logits_relative_L2=float(error),logits_max_abs=float(maximum),
                        residual_max_abs=float((before-after).abs().max()),repeat_max_abs=0.))
                raw=runtime.adapter.prepare_inputs(records[0].image_path,records[0].question,None);tokens=[]
                for version in (original,original,state):
                    e.load_state_dict(version)
                    with torch.inference_mode(),hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    tokens.append(list(g.raw_token_ids))
                assert tokens[0]==tokens[1]==tokens[2],'Gauge free-generation parity failed'
            finally:hook.detach()
            del e
            new=fixed_expert(state,t,runtime.device);_,a=new.factors()
            features=[question_features(runtime,rec,a.detach()) for rec in records]
            initial_rng=c.rng();snapshot={k:v.detach().cpu().clone() for k,v in new.state_dict().items()}
            test_hook=replay.MedTraceLayerHook(runtime.get_module(c.LAYER),new);test_hook.attach()
            try:
                history=sm.History();opt=p.optimizer(new,runtime.model)
                item=update(runtime,test_hook,new,opt,batches,1,history,features,ARMS[0])
                assert item['actual_norm']>0 and item['history_before']==0 and item['history_after']<=2
                assert all(torch.equal(snapshot[k],v.detach().cpu()) for k,v in new.state_dict().items() if k!='G2')
                assert all(v.grad is None for v in runtime.model.parameters())
            finally:test_hook.detach();c.restore_rng(initial_rng)
            # Export contains only the original four tensors and uses native TT inference.
            c.save(prepared(t),dict(expert=state,features=features,gauge=audit,state_hash=c.state_hash(p.expert(state,t['seed']))))
            loaded=p.load_state(prepared(t));restored=p.expert(loaded['expert'],t['seed'],runtime.device)
            check=p.expert(state,t['seed'],runtime.device)
            assert torch.equal(restored.residual(h),check.residual(h))
            audits.append(dict(expert_index=t['order'],parity=rows,generation_tokens_equal=True,
                fixed_cores_unchanged=True,Base_gradient=False,G2_nonzero_update=item['actual_norm'],
                save_load_native_TT_equal=True,gates=[x['gate'] for x in features],mechanical_updates=1))
            print('MECHANICAL',t['order'],'PASS',flush=True)
            del new,opt,restored,check,batches
    c.write(RUN/'public'/f'NATIVE_PARITY_{part}.json',dict(status='PASS',experts=audits))
    p.done('MECHANICAL_'+str(part))


def train():
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert (RUN/'private/PLAN_COMPLETE.json').exists()
    part=int(os.environ['PARTITION']);gpu=GPUS[part]
    with p.lease(gpu):
        runtime,_=c.load(gpu)
        for t in selected()[part::2]:
            init=p.load_state(prepared(t));features=init['features']
            batches=[runtime.build_edit_batch(c.record(t))]+[runtime.build_edit_batch(replace(c.record(t),question=q)) for q in t['fit_questions']]
            order=list(range(4));random.Random(t['seed']).shuffle(order)
            old=c.read(SOURCE/'private/weights/CE192'/t['anonymous_edit']/'TRAINING.json')
            assert old['binding']['fit_order']==order
            for arm in ARMS:
                directory=node(t,arm,192).parent;receipt=directory/'TRAINING.json';latest=directory/'latest.pt'
                if receipt.exists():
                    assert all(node(t,arm,n).exists() for n in NODES);continue
                e=fixed_expert(init['expert'],t,runtime.device);opt=p.optimizer(e,runtime.model);history=sm.History()
                hook=replay.MedTraceLayerHook(runtime.get_module(c.LAYER),e);hook.attach();seed_everything(t['seed'])
                binding=dict(arm=arm,W0=init['state_hash'],scientific_lock=c.digest(c.read(RUN/'private/SCOPED_LOCK.json')),
                    execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),fit_order=order,loss='0.5native+0.5FIT',replay=False)
                start=0;curve=[];fixed={k:v.detach().cpu().clone() for k,v in e.state_dict().items() if k!='G2'}
                if latest.exists():
                    saved=p.load_state(latest);assert saved['binding']==binding
                    e.load_state_dict(saved['expert']);opt.load_state_dict(saved['optimizer']);history.restore(saved['history'])
                    c.restore_rng(saved);start=saved['step'];curve=saved['curve']
                try:
                    for step in range(start+1,193):
                        c.budget();item=update(runtime,hook,e,opt,batches,1+order[(step-1)%4],history,features,arm)
                        if item['history_after']!=item['history_before'] or step in NODES:item['history_spectrum']=history.spectrum()
                        item['step']=step;curve.append(item)
                        if step in NODES:
                            assert all(torch.equal(fixed[k],v.detach().cpu()) for k,v in e.state_dict().items() if k!='G2')
                            c.save(node(t,arm,step),dict(expert=e.state_dict(),state_hash=c.state_hash(e),step=step,binding=binding))
                        if step%32==0:
                            c.save(latest,dict(expert=e.state_dict(),optimizer=opt.state_dict(),history=history.state(),step=step,curve=curve,binding=binding,**c.rng()))
                            print('TRAIN',t['order'],arm,step,flush=True)
                    c.write(receipt,dict(status='COMPLETE',binding=binding,updates=192,curve=curve,final_hash=c.state_hash(e),
                        fixed_cores_unchanged=True,Base_gradient=False,unique_history_keys=len(history.keys),history_spectrum=history.spectrum(),
                        gate_values=[x['gate']['gamma'] for x in features],key_norms=[float(x['key'].norm()) for x in features],
                        training_parameters=2048,deployed_parameters=7168))
                    latest.unlink()
                finally:hook.detach()
                del e,opt
    p.done('TRAIN_'+str(part))


def evaluate():
    part=int(os.environ['PARTITION']);gpu=GPUS[part];jobs=c.read(RUN/'private/JOBS.json')
    ts=p.tasks();byid={t['edit_id']:t for t in ts};chosen={t['edit_id'] for t in selected()}
    states={};weights={};reference=c.read(RUN/'private/REFERENCE_WEIGHTS.json')
    for job in jobs:
        arm=job['arm']
        if arm in weights:continue
        weights[arm]=dict(reference)
        if arm=='W0_NAT':continue
        target_arm=ARMS[4] if job['variant'] else next(a for a in ARMS if arm.startswith(a+'_s'))
        for t in selected():
            saved=p.load_state(node(t,target_arm,job['node']));state=saved['expert']
            if job['variant']:
                initial=p.load_state(prepared(t))['expert'];state={k:v.clone() for k,v in state.items()}
                sl=slice(2,4) if job['variant']=='local' else slice(0,2)
                state['G2'][:,:,sl]=initial['G2'][:,:,sl]
            states[arm,t['edit_id']]=state
            weights[arm][t['edit_id']]=dict(hash=c.state_hash(p.expert(state,t['seed'])),path=str(node(t,target_arm,job['node'])),variant=job['variant'])
    routes=c.read(RUN/'private/ROUTES.json');execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);loaded={}
        # A fresh unchanged native output checks the negative-control reuse contract.
        qid=[q for q in p.queries(146) if routes[q]['effective_expert'] not in chosen and routes[q]['effective_expert'] is not None][part]
        old=c.read(historical('W0',qid))
        owner=byid[old['effective_expert']];control_expert=p.expert(p.load_state(p.initial(owner))['expert'],owner['seed'],runtime.device)
        hook=replay.MedTraceLayerHook(runtime.get_module(c.LAYER),control_expert);hook.attach()
        try:
            row=old['binding']['input'];raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            with torch.inference_mode(),hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            assert list(g.raw_token_ids)==old['R0']['raw_token_ids']
        finally:hook.detach()
        p.done('NEGATIVE_CONTROL_'+str(part),dict(original_output_equal=True,expert_unchanged=True))
        for i,job in enumerate(jobs[part::2]):
            c.budget();dest=Path(job['path']);row=job['row'];qid=row['query_id'];expert=job['expert']
            phase=dict(arm=job['arm'],node=job['node'],prefix=146,slot=0,weights=weights[job['arm']],
                execution=execution,scoped_lock=c.digest(c.read(RUN/'private/SCOPED_LOCK.json')),variant=job['variant'])
            if dest.exists():assert c.read(dest)['binding']['phase']==phase;continue
            old=c.read(historical('W0',qid))
            bind=old['binding']['judge_input'];raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            assert raw['image_sha256']==bind['image_sha256'] and raw['input_ids'][0].tolist()==bind['prompt_ids']
            assert raw['attention_mask'][0].tolist()==bind['attention_mask'] and runtime.generation_config==bind['generation']
            hook=None
            if expert is not None:
                key=job['arm'],expert
                if key not in loaded:
                    t=byid[expert];state=states.get(key)
                    if state is None:
                        assert job['arm']=='W0_NAT';state=p.load_state(p.initial(t))['expert']
                    loaded[key]=p.expert(state,t['seed'],runtime.device).requires_grad_(False)
                hook=replay.MedTraceLayerHook(runtime.get_module(c.LAYER),loaded[key]);hook.attach()
            try:
                with torch.inference_mode():
                    if hook:
                        with hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    else:g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
            finally:
                if hook:hook.detach()
            actual=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids))
            base_path=historical_index().get('BASE',{}).get(qid)
            base_tokens=c.read(base_path)['R0']['raw_token_ids'] if base_path else bindings[qid]['output']['raw_generated_token_ids']
            c.write(dest,dict(binding=dict(input=row,judge_input=bind,phase=phase,arm=job['arm'],mode=job['mode']),
                R0=actual,effective_expert=expert,route=routes[qid],active_target=old.get('active_target',False),U_KL=None,
                Base_token_consistency=actual['raw_token_ids']==base_tokens))
            if i%25==0:print('EVAL',part,i,len(jobs[part::2]),flush=True)
    p.done('EVAL_'+str(part))


def controller():
    import pipeline
    assert (RUN/'private/PLAN_COMPLETE.json').exists()
    pipeline.wait([pipeline.launch('scoped.py','scoped_train',g,i) for i,g in enumerate(GPUS)])
    pipeline.wait([pipeline.launch('scoped.py','scoped_eval',g,i) for i,g in enumerate(GPUS)])
    consumers=c.read(RUN/'private/CONSUMERS.json');assert all(Path(x['path']).exists() for x in consumers)
    removed=[]
    for t in selected():
        for arm in ARMS:
            for step in (32,96):
                path=node(t,arm,step);assert path.is_file() and not path.is_symlink()
                removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    # Route features are no longer consumers of training or scoring; final TT states stay retained.
    for path in (RUN/'private/route_features').glob('*.pt'):
        assert not path.is_symlink();removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    c.write(RUN/'private/DELETION.json',dict(files=removed,final_TT_retained=40,historical_assets_untouched=True))
    p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('scoped_queue.py','scoped_ingest')])
    root=RUN/'private/judge_scoped_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('scoped_report.py','scoped_report')])


if __name__=='__main__':
    try:
        {'prepare':prepare,'scoped_mechanical':mechanical,'scoped_train':train,'scoped_eval':evaluate,'scoped_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','all')+'.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
