"""One bounded GPU job. Old experts are frozen; each bank prefix is regenerated."""
import copy
import fcntl
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['RUN_ROOT'])
sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(ROOT/'source_patch'))
from freshstart import runtime as rt
rt.ROOT=ROOT;rt.GPU=os.environ.get('PINNED_GPU_UUID',json.loads((ROOT/'CONFIG_LOCK.json').read_text())['gpu_uuid'])
from freshstart.runtime import read,write
from scripts.medtrace.stage17_prepare import digest


def request(row,output):
    record=dict(question=row['question'],gold_answer=row['reference'],raw_base_answer=output['raw_answer'])
    key=digest(dict(record=record,protocol='MEDTRACE_STAGE17_SOURCE_AGREEMENT_V1',run='textjoint0924'))
    record['opaque_query_id']=key
    path=ROOT/'private/judge/pending'/f'{key}.json'
    if not path.exists():write(path,dict(key=key,record=record))
    return key


def input_id(row):return digest([row['image_sha256'],row['question']])


def record(t):
    from m3bench_repro.editors.llava_runtime import EditorRecord
    n=t['native']
    return EditorRecord(t['canonical_edit_id'],n['dataset'],n['question'],n['reference'],t['fit_questions'][0],
        Path(n['image_path']),'',t['order'],'SOURCE_LABEL','EXPOSED_DEV')


def rows(t):return t['evaluation']


def base(runtime,row,rec,score=True):
    from scripts.medtrace import stage15
    path=ROOT/'private/base'/f'{input_id(row)}.json'
    with (ROOT/'private/base/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            raw,_,binding=stage15.prepared(runtime,row,rec)
            write(path,stage15.generate(runtime,raw,binding))
        result=read(path)
    return result,request(row,result) if score else None


def key(runtime,row,rec):
    import torch
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import assert_base_off
    assert_base_off(runtime)
    path=ROOT/'private/keys'/f'{input_id(row)}.pt'
    with (ROOT/'private/keys/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():return torch.load(path,map_location=runtime.device,weights_only=True)
        _,batch,_=stage15.prepared(runtime,row,rec)
        with torch.inference_mode():value=runtime.extract_layer_input_key(batch,module_path=runtime.target_lock['balancedit']['targets'][0],pooling='mean')
        from scripts.medtrace.run_selective_write import save
        save(path,value.cpu());return value


def router_entry(runtime,t):
    from m3bench_repro.editors.routing import balanced_radius
    n=t['native'];rec=record(t);nk=key(runtime,n,rec)
    positive=key(runtime,dict(n,question=t['fit_questions'][0]),rec)
    black=runtime.make_black_image(rec,ROOT/'private/black'/str(t['order']))
    negative=key(runtime,dict(n,image_path=str(black),image_sha256='BLACK-'+n['image_sha256']),rec)
    return nk,balanced_radius(nk,positive,negative,alpha=.2,distance='euclidean')


def protection(runtime,t,run,urows):
    import torch
    from scripts.medtrace.run_selective_write import teacher_batch,save
    from scripts.medtrace.stage18_cfact import assert_base_off,check_cache
    from methods.medtrace.selective_write import full_vocab_kl
    assert_base_off(runtime);teachers=[];rec=record(t)
    for row in urows:
        b,_=base(runtime,row,rec,score=False)
        kwargs,labels,mask,binding=teacher_batch(runtime,dict(row,eqkey=input_id(row)),b['raw_token_ids'])
        binding.update(model='91bb16c122001ddc9cf1fd36ce1dae09448943a2',teacher='ALL_EXPERTS_OFF',direction='Base||student',layer=rt.LAYER)
        path=ROOT/'private/teachers'/f'{digest(binding)}.pt'
        if path.exists():saved=torch.load(path,weights_only=True,map_location='cpu')
        else:
            with torch.no_grad():logp=runtime.model(**kwargs).logits[mask].float().log_softmax(-1).cpu()
            saved=dict(binding=binding,logp=logp);save(path,saved)
        logp=check_cache(saved,binding)
        if logp.shape!=(int(mask.sum()),runtime.model.config.vocab_size):raise ValueError('KL mask/vocabulary mismatch')
        teachers.append((kwargs,labels,mask,logp))
    if not teachers:raise ValueError('No U: objective cannot silently drop protection')
    trace=[]
    def add(_runtime,hook,expert,step,stage):
        params=list(expert.parameters())
        before=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        losses=[]
        for kwargs,labels,mask,logp in teachers:
            hook.set_teacher_routing(labels)
            loss=full_vocab_kl(runtime.model(**kwargs).logits[mask],logp)
            (.01*loss/len(teachers)).backward();losses.append(float(loss.detach()))
        norm=float(torch.sqrt(sum(((p.grad if p.grad is not None else torch.zeros_like(p))-b).float().square().sum() for p,b in zip(params,before))))
        trace.append(dict(stage=stage,step=step,U_mean_kl=sum(losses)/len(losses),U_weight=.01,U_count=len(teachers),
            U_gradient_norm=norm,preceding_objective_gradient_norm=float(torch.sqrt(sum(b.float().square().sum() for b in before))),
            token_counts=[int(x[2].sum()) for x in teachers]))
        if step%20==0:write(run/'U_TRACE.json',trace)
        return sum(losses)/len(losses)
    return add


def train(runtime,t,arm,seed):
    import torch
    from scripts.medtrace import stage15
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace.run_selective_write import save
    run=ROOT/'runs'/f's{seed}'/arm/f'e{t["order"]:03d}'
    point=run/'FINAL.pt'
    if point.exists():
        saved=torch.load(point,map_location=runtime.device,weights_only=True)
        if saved['arm']!=arm or saved['seed']!=seed or saved['edit']!=t['canonical_edit_id']:raise ValueError('Foreign checkpoint')
        expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(runtime.device),t['seed'],rank=4).to(runtime.device)
        expert.load_state_dict(saved['expert']);expert.requires_grad_(False);return expert
    if (run/'FAILURE.json').exists():raise RuntimeError('Prior training failure: no silent retry')
    task=copy.deepcopy(t)
    task['seed']+=seed-20260924
    if arm in ['P','PEU']:task['fit_questions']=task['semantic_fit_questions']
    us=task['U_expanded'] if arm in ['U','EU','PEU'] else task['U_fit']
    protector=protection(runtime,task,run,us) if arm in ['E','U','EU','PEU'] else None
    early=protector if arm in ['E','EU','PEU'] else None
    m=read(ROOT/'RUN_MANIFEST.json');cfg=dict(campaign_epoch=rt.epoch(m['first_started_at']),train_seconds=rt.epoch(m['no_new_training_after'])-rt.epoch(m['first_started_at']))
    before=time.time();adapted=dict(task,probes=[task['native']],U=us)
    try:
        cp=stage15.initialize(runtime,run,cfg,adapted,record=record(task),seed_base=seed,layer_path=rt.LAYER,protection=early)
        expert=LowRankExpert(cp,task['seed'],rank=4).to(runtime.device)
        with torch.no_grad():
            x=torch.linspace(-1,1,14336,device=runtime.device).reshape(1,-1)
            if not torch.allclose(cp.residual(x),expert.residual(x),rtol=2e-4,atol=2e-5):raise ValueError('Conversion mismatch')
        stage15.train_steps(runtime,run,cfg,adapted,expert,'C_NO_H',record=record(task),layer_path=rt.LAYER,protection=protector)
        expert.requires_grad_(False)
        save(point,dict(expert=expert.state_dict(),arm=arm,seed=seed,edit=task['canonical_edit_id'],layer=rt.LAYER))
        restored=copy.deepcopy(expert);restored.load_state_dict(torch.load(point,map_location=runtime.device,weights_only=True)['expert'])
        if any(not torch.equal(a,b) for a,b in zip(expert.state_dict().values(),restored.state_dict().values())):raise ValueError('Reload mismatch')
        del restored,cp
        initial=run/'private/edits'/f"e{task['order']:03d}"/'initial'
        curves={}
        native=read(initial/'NATIVE_INITIALIZATION.json')
        curves['native_CP']=[dict(step=x['step'],native_ce=x['score']['nll'],grad_norm=x['gradient_norm']) for x in native['trajectory']]
        a2=read(initial/'A2_TRAINING.json')
        curves['A2']=[dict(step=x['step'],native_ce=x.get('micro_losses',[None,None])[0],fit_ce=x.get('micro_losses',[None,None])[-1],grad_norm=x.get('gradient_norm')) for x in a2['trajectory']]
        for stage in ['CP_W0','C_NO_H']:
            saved=torch.load(initial.parent/stage/'latest.pt',map_location='cpu',weights_only=True)
            curves[stage]=saved['curve']
        write(run/'TRAINING_CURVES.json',curves)
        write(run/'SEED_LOCK.json',dict(native_A2_base=seed,CP_continuation=task['seed'],native_initial_state=native['initial_expert_sha256'],A2_start_state=a2['start_state_sha256']))
        write(run/'TRAINING_COMPLETE.json',dict(status='PASS',seconds=time.time()-before,actual_U=len(us),early=early is not None,seed=seed))
        if not runtime.base_guard.verify()['unchanged']:raise RuntimeError('Base changed')
        return expert
    except Exception as e:
        write(run/'FAILURE.json',dict(error=str(e),time=time.time(),checkpoint_preserved=True));raise


def evaluate(runtime,tasks,bank,router,arm,seed,mode,prefix,folder):
    from scripts.medtrace import stage15
    from methods.medtrace import MedTraceLayerHook
    # Cache only within this freshly evaluated bank prefix; never import single outputs.
    local={};consumers=[]
    for t in tasks:
        for row in rows(t):
            rt.check_budget()
            ident=input_id(row);rec=record(t)
            if ident not in local:
                decision=asdict(router.route(key(runtime,row,rec)))
                selected=decision['logical_edit_id'];expert=bank.get(selected)
                if selected is not None and expert is None:raise ValueError('Route outside current bank')
                raw,_,binding=stage15.prepared(runtime,row,rec)
                hook=MedTraceLayerHook(runtime.get_module(rt.LAYER),expert) if expert is not None else None
                if hook:hook.attach()
                try:output=stage15.generate(runtime,raw,binding,hook)
                finally:
                    if hook:hook.detach()
                local[ident]=(decision,output)
            decision,output=local[ident]
            original,base_judge=base(runtime,row,rec)
            consumer=dict(arm=arm,seed=seed,mode=mode,prefix=prefix,edit=t['canonical_edit_id'],order=t['order'],
                task=row['task'],query_id=row['query_id'],source_group=row['source_group'],judge_key=request(row,output),
                base_judge_key=base_judge,route=decision,output=output,exact_Base_token_consistency=output['raw_token_ids']==original['raw_token_ids'])
            consumers.append(consumer)
    write(folder/'CONSUMERS.json',consumers)
    return consumers


def diagnose(runtime,tasks,job,folder):
    import torch
    from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace import stage15
    scores={p.stem:read(p)['is_correct'] for p in (ROOT/'private/judge/scores').glob('*.json')}
    for t in tasks:
        rec=record(t);selected=[]
        for kind in ['T0','T1G','T1L','T2G','T2L']:
            rr=[r for r in rows(t) if r['task']==kind]
            if not rr:continue
            eligible=[r for r in rr if scores.get(base(runtime,r,rec)[1]) is kind.endswith('L')]
            selected.append((eligible or rr)[0])
        run=ROOT/'runs'/f"s{job['seed']}"/'B0'/f"e{t['order']:03d}"
        initial=run/'private/edits'/f"e{t['order']:03d}"/'initial'
        cp=lambda:AsymmetricCPExpert(14336,4096,4).to(runtime.device)
        states=[('Base_OFF',None)]
        for label,path in [('native_CP',initial/'native/expert.pt'),('A2',initial/'A2.pt'),('CP_W0',initial/'W0_COMPLETE.pt')]:
            expert=cp();expert.load_state_dict(torch.load(path,map_location=runtime.device,weights_only=True)['expert']);expert.requires_grad_(False);states.append((label,expert))
        states.append(('low_rank_conversion',LowRankExpert(states[-1][1],t['seed'],rank=4).to(runtime.device)))
        final=LowRankExpert(cp(),t['seed'],rank=4).to(runtime.device)
        final.load_state_dict(torch.load(run/'FINAL.pt',map_location=runtime.device,weights_only=True)['expert']);final.requires_grad_(False);states.append(('continuation',final))
        router=MemoryRouter('euclidean');k,radius=router_entry(runtime,t);router.add(t['canonical_edit_id'],k,radius)
        outputs=[]
        for stage,expert in states:
            for row in selected:
                bo,bj=base(runtime,row,rec);route=asdict(router.route(key(runtime,row,rec)))
                for mode in (['OFF'] if expert is None else ['FORCED_ON','ROUTED']):
                    active=expert if mode=='FORCED_ON' or mode=='ROUTED' and route['logical_edit_id'] else None
                    raw,_,binding=stage15.prepared(runtime,row,rec)
                    hook=MedTraceLayerHook(runtime.get_module(rt.LAYER),active) if active is not None else None
                    if hook:hook.attach()
                    try:out=stage15.generate(runtime,raw,binding,hook)
                    finally:
                        if hook:hook.detach()
                    outputs.append(dict(stage=stage,mode=mode,task=row['task'],edit=t['canonical_edit_id'],query_id=row['query_id'],
                        source_group=row['source_group'],base_judge_key=bj,judge_key=request(row,out),route=route,output=out,
                        base_correct=scores.get(bj),exact_Base_token_consistency=bo['raw_token_ids']==out['raw_token_ids']))
        write(folder/f"e{t['order']:03d}.json",outputs)
        write(folder/'PROGRESS.json',dict(order=t['order'],phase='STAGE_DIAGNOSTICS',epoch=time.time()))
        del states,final,expert


def baseline(runtime,tasks,job,folder):
    from scripts.medtrace import stage15
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    editor=BalanceEditPaperSpecEditor(runtime,inactive_store_dir=folder/'weights')
    original,target=editor.wrapper.base,editor.target
    try:
        for prefix,t in enumerate(tasks,1):
            rt.check_budget(training=True)
            if job['mode']=='single':editor.reset_editor_state()
            outcome=editor.apply_edit(record(t))
            if not outcome['finite_losses'] or not outcome['finite_gradients'] or outcome['steps']!=50:
                raise FloatingPointError('Baseline numerical failure')
            output_dir=folder/'BalancEdit'/f'p{prefix:03d}'
            write(output_dir/'TRAINING.json',dict(outcome,classification='PAPER_SPEC_ADAPTATION',layer=target))
            consumers=[];cache={}
            chosen=[t] if job['mode']=='single' else tasks[:prefix]
            for old in chosen:
                for row in rows(old):
                    rec=replace(record(old),record_id='query',question=row['question'],target='',official_rephrase='',image_path=Path(row['image_path']))
                    ident=input_id(row)
                    if ident not in cache:
                        raw,_,binding=stage15.prepared(runtime,row,rec)
                        with editor.route_generation(rec) as route:out=stage15.generate(runtime,raw,binding)
                        cache[ident]=(asdict(route),out)
                    route,out=cache[ident];bo,bj=base(runtime,row,record(old))
                    consumers.append(dict(arm='BalancEdit',seed=job['seed'],mode=job['mode'],prefix=1 if job['mode']=='single' else prefix,
                        edit=old['canonical_edit_id'],order=old['order'],task=row['task'],query_id=row['query_id'],source_group=row['source_group'],
                        judge_key=request(row,out),base_judge_key=bj,route=route,output=out,
                        exact_Base_token_consistency=out['raw_token_ids']==bo['raw_token_ids']))
            write(output_dir/'CONSUMERS.json',consumers)
            editor.save_editor_state(folder/'ACTIVE_BE.tmp.pt');(folder/'ACTIVE_BE.tmp.pt').replace(folder/'ACTIVE_BE.pt')
            write(folder/'PROGRESS.json',dict(arm='BalancEdit',prefix=prefix,N=len(tasks),epoch=time.time()))
        editor.load_editor_state(folder/'ACTIVE_BE.pt')
        write(folder/'BE_RELOAD.json',dict(status='PASS',layer=target,native_steps=50,classification='PAPER_SPEC_ADAPTATION'))
    finally:
        editor.reset_editor_state();runtime.replace_module(target,original)


def main():
    import torch
    from m3bench_repro.editors.routing import MemoryRouter
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace.run_selective_write import save
    job=read(Path(os.environ.get('RUN_JOB_JSON',ROOT/'JOB.json')));jid=job['id'];folder=ROOT/'jobs'/jid;start=time.time()
    write(folder/'STATUS.json',dict(status='RUNNING',job=job,started_epoch=start))
    tasks=read(ROOT/'private/TASKS_MATRIX.json')['tasks'][:job['N']]
    if job.get('reverse'):tasks=list(reversed(tasks))
    if job.get('order_seed') is not None:
        import random
        random.Random(job['order_seed']).shuffle(tasks)
    seed=job.get('seed',20260924);training=False;work_started=None
    try:
        with rt.gpu_session(jid):
            runtime=rt.load_runtime(folder/'runtime',seed)
            work_started=time.time()
            def guard(_module,_args):
                rt.check_budget(training=training)
                if (ROOT/'JUDGE_FAILURE.json').exists():raise RuntimeError('Scoring blocked; preserve state')
            handle=runtime.model.register_forward_pre_hook(guard)
            if job['mode']=='base':
                for t in tasks:
                    for row in rows(t):base(runtime,row,record(t))
                    write(folder/'PROGRESS.json',dict(order=t['order'],phase='BASE'))
            elif job['mode']=='diagnostic':
                diagnose(runtime,tasks,job,folder)
            elif job['arms']==['BalancEdit']:
                training=True;baseline(runtime,tasks,job,folder);training=False
            else:
                for arm in job['arms']:
                    bank={};router=MemoryRouter('euclidean')
                    for prefix,t in enumerate(tasks,1):
                        write(folder/'PROGRESS.json',dict(arm=arm,order=t['order'],prefix=prefix,phase='TRAIN_OR_RELOAD',epoch=time.time()))
                        training=True;expert=train(runtime,t,arm,seed);training=False
                        if job['mode']=='single':bank={};router=MemoryRouter('euclidean')
                        bank[t['canonical_edit_id']]=expert
                        k,radius=router_entry(runtime,t);router.add(t['canonical_edit_id'],k,radius)
                        chosen=[t] if job['mode']=='single' else tasks[:prefix]
                        evaluate(runtime,chosen,bank,router,arm,seed,job['mode'],1 if job['mode']=='single' else prefix,
                            folder/arm/f'p{prefix:03d}')
                        if job['mode']=='sequential':
                            save(folder/arm/'ACTIVE_BANK.pt',dict(router=router.export_state(),experts={k:v.state_dict() for k,v in bank.items()},prefix=prefix))
                        if any(p.requires_grad for ex in bank.values() for p in ex.parameters()):raise RuntimeError('Old bank expert unfrozen')
                        write(folder/'PROGRESS.json',dict(arm=arm,order=t['order'],prefix=prefix,phase='GENERATED',epoch=time.time()))
                    if job['mode']=='sequential':
                        restored=torch.load(folder/arm/'ACTIVE_BANK.pt',map_location=runtime.device,weights_only=True)
                        restored_router=MemoryRouter.from_state(restored['router'],device=runtime.device)
                        assert set(restored['experts'])==set(bank)
                        for edit,expert in bank.items():
                            assert all(torch.equal(value,restored['experts'][edit][name]) for name,value in expert.state_dict().items())
                        probe=key(runtime,t['native'],record(t))
                        assert asdict(restored_router.route(probe))==asdict(router.route(probe))
                        write(folder/arm/'BANK_RELOAD.json',dict(status='PASS',experts=len(bank),prefix=len(tasks),exact_tensors=True,routing_equivalent=True))
                        del restored,restored_router
                    del bank,router
            handle.remove()
            if not runtime.base_guard.verify()['unchanged']:raise RuntimeError('Base guard failed')
        write(folder/'STATUS.json',dict(status='GPU_COMPLETE',job=job,seconds=time.time()-start,effective_work_seconds=time.time()-work_started if work_started else 0))
    except Exception as e:
        import traceback
        write(folder/'STATUS.json',dict(status='FAILED',job=job,error=str(e),traceback=traceback.format_exc(),seconds=time.time()-start,effective_work_seconds=time.time()-work_started if work_started else 0));raise


if __name__=='__main__':main()
