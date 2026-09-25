"""R2 B0/P/P+S continuation with shared P-W0 and frozen sparse evaluation."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(os.environ['RUN_ROOT'])
sys.path.insert(0, str(ROOT))
import worker_v3 as old
from freshstart import runtime as rt
from freshstart.runtime import read, write


def _norm(parts):
    import torch
    return torch.sqrt(sum(p.float().square().sum() for p in parts))


def _cos(a, b):
    import torch
    na = _norm(a); nb = _norm(b)
    return float(sum((x.float()*y.float()).sum() for x, y in zip(a,b)) / (na*nb)) if na > 0 and nb > 0 else None


def make_protection(runtime, task, run, expert):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from methods.medtrace.selective_write import full_vocab_kl
    from scripts.medtrace import stage15
    from scripts.medtrace.run_selective_write import teacher_batch, save
    from scripts.medtrace.stage18_cfact import assert_base_off
    assert_base_off(runtime)
    rec = old.record(task)
    teacher_audit=[]
    if any(p.requires_grad for p in runtime.model.parameters()):raise ValueError('Teacher backbone is trainable')
    def teacher(row):
        base, _, _, key = stage15.base_output(runtime, run, row, rec)
        kwargs, labels, mask, binding = teacher_batch(runtime, dict(row, eqkey=key), base['raw_token_ids'])
        binding.update(teacher='ALL_EXPERTS_OFF', direction='Base||student', layer=rt.LAYER)
        path = ROOT/'private/teachers'/(old.digest(binding)+'.pt')
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['binding'] != binding:raise ValueError('Teacher binding changed')
        else:
            with torch.no_grad():
                logp = runtime.model(**kwargs).logits[mask].float().log_softmax(-1).cpu()
            saved = dict(binding=binding, logp=logp)
            save(path, saved)
        if tuple(saved['logp'].shape)!=(int(mask.sum()),runtime.model.config.vocab_size):raise ValueError('Teacher vocab/mask shape mismatch')
        teacher_audit.append(dict(tokens=len(base['raw_token_ids']),predictor_tokens=int(mask.sum()),
            EOS_in_prefix=runtime.adapter.tokenizer.eos_token_id in base['raw_token_ids'],cap_hit=base['cap_hit'],
            modality='image+text',teacher='all experts off',direction='Base||student',reduction='full vocabulary; mean over tokens'))
        return kwargs, labels, mask, saved['logp']
    old_u = [teacher(r) for r in task['U_fit']]
    new_u = [teacher(r) for r in task['U_new']]
    write(run/'TEACHER_MASK_EOS_AUDIT.json',teacher_audit)
    if not old_u or not new_u:raise ValueError('P+S requires old and new U')
    schedule = [random.Random(task['seed']*1000003+s).randrange(len(new_u)) for s in range(1,321)]
    write(run/'SAMPLER.json', dict(rule='one uniformly sampled new U per step, stateless frozen seed',
                                   seed=task['seed'],indices=schedule,old_U=len(old_u),new_U=len(new_u)))
    def loss(teacher_row, hook):
        kwargs, labels, mask, logp = teacher_row
        hook.set_teacher_routing(labels)
        if not torch.equal(hook.token_mask, mask):raise ValueError('KL hook mask differs from teacher predictor mask')
        return full_vocab_kl(runtime.model(**kwargs).logits[mask], logp)
    if task['order'] <= 2:
        hook = MedTraceLayerHook(runtime.get_module(rt.LAYER), expert)
        hook.attach()
        try:
            first = old_u[0]
            kwargs, labels, mask, logp = first
            hook.set_teacher_routing(labels)
            logits=runtime.model(**kwargs).logits[mask]
            q=logp.to(logits.device).float()
            # Independent expression of the original full-vocabulary token-mean KL.
            a=.01*(q.exp()*(q-logits.float().log_softmax(-1))).sum(-1).mean()
            ga = torch.autograd.grad(a, tuple(expert.parameters()))
            b = 2*(.005*loss(first,hook))
            gb = torch.autograd.grad(b, tuple(expert.parameters()))
            max_abs = max(float((x-y).abs().max()) for x,y in zip(ga,gb))
            same = torch.allclose(a,b,rtol=1e-5,atol=1e-7) and all(torch.allclose(x,y,rtol=1e-4,atol=1e-7) for x,y in zip(ga,gb))
            write(run/'OLD_CALLBACK_EQUIVALENCE.json',dict(status='PASS' if same else 'FAIL',
                direction='Base||student',mask_tokens=int(first[2].sum()),
                old_loss=float(a.detach()),callback_old_component_doubled=float(b.detach()),max_abs_grad_delta=max_abs))
            if not same:raise ValueError('Old U KL and callback old component differ')
        finally:hook.detach()
    trace=[]
    def callback(_runtime, hook, selected_expert, step, stage):
        params=list(selected_expert.parameters())
        before=[p.grad.detach().clone() for p in params]
        old_losses=[]
        for sample in old_u:
            value=loss(sample,hook)
            (.005*value/len(old_u)).backward()
            old_losses.append(float(value.detach()))
        after_old=[p.grad.detach().clone() for p in params]
        old_grad=[x-y for x,y in zip(after_old,before)]
        chosen=schedule[step-1]
        new_loss=loss(new_u[chosen],hook)
        (.005*new_loss).backward()
        new_grad=[p.grad.detach()-a for p,a in zip(params,after_old)]
        total_grad=[p.grad.detach() for p in params]
        trace.append(dict(step=step,stage=stage,old_U_kl=sum(old_losses)/len(old_losses),new_U_kl_sample=float(new_loss.detach()),
                          new_U_index=chosen,old_U_count=len(old_u),new_U_count=len(new_u),
                          CE_grad_norm=float(_norm(before)),old_U_weighted_grad_norm=float(_norm(old_grad)),
                          new_U_weighted_grad_norm=float(_norm(new_grad)),total_preclip_grad_norm=float(_norm(total_grad)),
                          CE_old_cosine=_cos(before,old_grad),CE_new_cosine=_cos(before,new_grad),
                          old_new_cosine=_cos(old_grad,new_grad)))
        if step%20==0:write(run/'U_TRACE.json',trace)
        return .5*sum(old_losses)/len(old_losses)+.5*float(new_loss.detach())
    return callback


def train_ps(runtime,t,seed):
    import torch
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace import stage15
    from scripts.medtrace.run_selective_write import save
    run=ROOT/'runs'/f's{seed}'/'P+S'/f'e{t["order"]:03d}'
    point=run/'FINAL.pt'
    if point.exists():
        return load_expert(runtime,t,seed,'P+S',320)
    if (run/'FAILURE.json').exists():raise RuntimeError('Prior P+S failure; no silent retry')
    task=copy.deepcopy(t)
    task['seed']+=seed-20260924
    task['fit_questions']=task['semantic_fit_questions']
    task['probes']=[task['native']]
    task['U']=task['U_fit']
    manifest=read(ROOT/'RUN_MANIFEST.json')
    cfg=dict(campaign_epoch=rt.epoch(manifest['first_started_at']),
             train_seconds=rt.epoch(manifest['no_new_training_after'])-rt.epoch(manifest['first_started_at']))
    p_run=ROOT/'runs'/f's{seed}'/'P'/f'e{t["order"]:03d}'
    if not (p_run/'private/edits'/f'e{t["order"]:03d}'/'initial/W0_COMPLETE.pt').exists():
        raise RuntimeError('P-W0 must finish before P+S')
    began=time.time()
    try:
        cp=stage15.initialize(runtime,p_run,cfg,task,record=old.record(task),seed_base=seed,layer_path=rt.LAYER)
        expert=LowRankExpert(cp,task['seed'],rank=4).to(runtime.device)
        with torch.no_grad():
            x=torch.linspace(-1,1,14336,device=runtime.device).reshape(1,-1)
            if not torch.allclose(cp.residual(x),expert.residual(x),rtol=2e-4,atol=2e-5):
                raise ValueError('CP to low rank conversion mismatch')
        save(run/'STEP0.pt',dict(expert=expert.state_dict(),arm='P+S',seed=seed,edit=task['canonical_edit_id']))
        p0=torch.load(p_run/'STEP0.pt',map_location=runtime.device,weights_only=True)
        if any(not torch.equal(value,p0['expert'][name]) for name,value in expert.state_dict().items()):
            raise ValueError('P/P+S do not share identical step-0 state')
        write(run/'SHARED_P_W0.json',dict(status='PASS',source=str(p_run/'STEP0.pt'),same_step0=True))
        callback=make_protection(runtime,task,run,expert)
        stage15.train_steps(runtime,run,cfg,task,expert,'C_NO_H',record=old.record(task),layer_path=rt.LAYER,protection=callback)
        expert.requires_grad_(False)
        save(point,dict(expert=expert.state_dict(),arm='P+S',seed=seed,edit=task['canonical_edit_id'],layer=rt.LAYER))
        restored=load_expert(runtime,t,seed,'P+S',320)
        if any(not torch.equal(a,b) for a,b in zip(expert.state_dict().values(),restored.state_dict().values())):
            raise ValueError('P+S reload mismatch')
        write(run/'TRAINING_COMPLETE.json',dict(status='PASS',seconds=time.time()-began,old_U=len(task['U_fit']),
                                               new_U=len(task['U_new']),steps=task.get('continuation_steps',320),early_protection=False,seed=seed))
        if not runtime.base_guard.verify()['unchanged']:raise RuntimeError('Base changed')
        return expert
    except Exception as exc:
        write(run/'FAILURE.json',dict(error=str(exc),epoch=time.time(),checkpoint_preserved=True))
        raise


def load_expert(runtime,t,seed,arm,step=320):
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    run=ROOT/'runs'/f's{seed}'/arm/f'e{t["order"]:03d}'
    task_seed=t['seed']+seed-20260924
    expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(runtime.device),task_seed,rank=4).to(runtime.device)
    path=(run/'FINAL.pt' if step==320 else run/'private/edits'/f'e{t["order"]:03d}'/'C_NO_H'/f'step-{step}.pt')
    state=torch.load(path,map_location=runtime.device,weights_only=True)
    expected_seed=seed if step==320 else task_seed
    if state.get('seed')!=expected_seed or state.get('edit',state.get('canonical_edit_id'))!=t['canonical_edit_id']:
        raise ValueError('Checkpoint provenance mismatch')
    expert.load_state_dict(state['expert']);expert.requires_grad_(False)
    return expert


def ensure_p_step0(runtime,t,seed):
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace.run_selective_write import save
    run=ROOT/'runs'/f's{seed}'/'P'/f'e{t["order"]:03d}'
    path=run/'STEP0.pt'
    if path.exists():return
    w0=run/'private/edits'/f'e{t["order"]:03d}'/'initial/W0_COMPLETE.pt'
    state=torch.load(w0,map_location=runtime.device,weights_only=True)
    if state['canonical_edit_id']!=t['canonical_edit_id']:raise ValueError('P W0 provenance mismatch')
    cp=AsymmetricCPExpert(14336,4096,4).to(runtime.device)
    cp.load_state_dict(state['expert'])
    low=LowRankExpert(cp,t['seed']+seed-20260924,rank=4).to(runtime.device)
    save(path,dict(expert=low.state_dict(),arm='P',seed=seed,edit=t['canonical_edit_id']))


def main():
    import torch
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace.run_selective_write import save
    job=read(Path(os.environ['RUN_JOB_JSON']))
    folder=ROOT/'jobs'/job['id']
    if (folder/'STATUS.json').exists():
        state=read(folder/'STATUS.json')
        if state['job']!=job:raise ValueError('Job receipt binding changed')
        if state['status']=='GPU_COMPLETE':return
        raise RuntimeError('Existing unfinished job requires explicit recovery')
    start=time.time()
    write(folder/'STATUS.json',dict(status='RUNNING',job=job,started_epoch=start))
    tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks']
    orders=set(job['orders'])
    tasks=[t for t in tasks if t['order'] in orders]
    assert len(tasks)==len(orders)
    seed=job['seed'];training=False
    try:
        with rt.gpu_session(job['id']):
            runtime=rt.load_runtime(folder/'runtime',seed)
            import importlib,hashlib
            modules={}
            for name in ['freshstart.runtime','scripts.medtrace.stage15','methods.medtrace.selective_write','methods.medtrace','scripts.medtrace.run_selective_write','m3bench_repro.editors.routing','m3bench_repro.editors.llava_runtime']:
                path=Path(importlib.import_module(name).__file__).resolve()
                modules[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            write(folder/'MODULE_PATHS.json',modules)
            def guard(_module,_args):
                rt.check_budget(training=training)
                if any((ROOT/n).exists() for n in ['JUDGE_FAILURE.json','JUDGE_FAILURE_RECOVERY.json']):raise RuntimeError('Judge stopped')
            handle=runtime.model.register_forward_pre_hook(guard)
            if job['mode']=='crosscheck':
                from diagnostics_r2 import crosscheck
                crosscheck(runtime,tasks,job,folder);tasks=[]
            if job['mode']=='sequential':
                for arm in job['arms']:
                    bank={};router=MemoryRouter('euclidean')
                    step=320 if arm=='B0' else job['selected_step']
                    label='B0' if arm=='B0' else f'{arm}@{step}'
                    for prefix,t in enumerate(tasks,1):
                        bank[t['canonical_edit_id']]=load_expert(runtime,t,seed,arm,step)
                        key,radius=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],key,radius)
                        if prefix in job['prefixes']:
                            old.evaluate(runtime,tasks[:prefix],bank,router,label,seed,'sequential',prefix,folder/label/f'p{prefix:03d}')
                tasks=[]
            if job['mode']=='baseline':
                training=True
                old.baseline(runtime,tasks,dict(job,mode='single'),folder)
                training=False;tasks=[]
            for t in tasks:
                for arm in job['arms']:
                    t=dict(t,continuation_steps=320 if arm=='B0' else job.get('train_steps',320))
                    write(folder/'PROGRESS.json',dict(order=t['order'],arm=arm,phase='TRAIN_OR_RELOAD',epoch=time.time()))
                    training=True
                    if arm=='P+S' and not (ROOT/'runs'/f's{seed}'/'P'/f'e{t["order"]:03d}'/'FINAL.pt').exists():
                        old.train(runtime,t,'P',seed);ensure_p_step0(runtime,t,seed)
                    expert=train_ps(runtime,t,seed) if arm=='P+S' else old.train(runtime,t,arm,seed)
                    training=False
                    if arm=='P':ensure_p_step0(runtime,t,seed)
                    router=MemoryRouter('euclidean')
                    key,radius=old.router_entry(runtime,t)
                    router.add(t['canonical_edit_id'],key,radius)
                    for step in ([320] if arm=='B0' else job.get('steps',[80,160,320])):
                        label=arm if arm=='B0' else f'{arm}@{step}'
                        receipt=folder/label/f'p{t["order"]:03d}'/'CONSUMERS.json'
                        if receipt.exists():
                            saved=read(receipt)
                            expected={(r['task'],r['query_id']) for r in t['evaluation']}
                            if ({(r['task'],r['query_id']) for r in saved}!=expected or len(saved)!=len(t['evaluation'])
                                or any(r['arm']!=label or r['seed']!=seed or r['edit']!=t['canonical_edit_id'] or r['mode']!='single' for r in saved)):
                                raise ValueError('Saved generation binding changed')
                            continue
                        chosen=expert if step==320 else load_expert(runtime,t,seed,arm,step)
                        old.evaluate(runtime,[t],{t['canonical_edit_id']:chosen},router,label,seed,'single',1,
                                     folder/label/f'p{t["order"]:03d}')
                    write(folder/'PROGRESS.json',dict(order=t['order'],arm=arm,phase='GENERATED',epoch=time.time()))
            handle.remove()
            if not runtime.base_guard.verify()['unchanged']:raise RuntimeError('Base changed')
        write(folder/'STATUS.json',dict(status='GPU_COMPLETE',job=job,seconds=time.time()-start))
    except Exception as exc:
        import traceback
        write(folder/'STATUS.json',dict(status='FAILED',job=job,error=str(exc),traceback=traceback.format_exc(),seconds=time.time()-start))
        raise


if __name__=='__main__':main()
