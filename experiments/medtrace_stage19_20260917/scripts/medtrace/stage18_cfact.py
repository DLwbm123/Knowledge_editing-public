"""Full locked Stage18 continuation. Training API accepts no evaluation payload."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import hashlib
import random
import time
import torch
from methods.medtrace import MedTraceLayerHook
from methods.medtrace.selective_write import optimizer_for, full_vocab_kl, balanced_schedule
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_support import validate_task
from scripts.medtrace.run_selective_write import save, teacher_batch
from scripts.medtrace import stage15

LAYER='model.layers.21.mlp.down_proj'
BRANCHES=('C_NO_H','C_EXTRA','C_FACT')


def state_hash(expert):
    h=hashlib.sha256()
    for k,v in sorted(expert.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_base_off(runtime):
    if any(p.requires_grad for p in runtime.model.parameters()): raise ValueError('Base must be frozen')
    for module in runtime.model.modules():
        for fn in module._forward_hooks.values():
            owner=getattr(fn,'__self__',None)
            if isinstance(owner,MedTraceLayerHook) and owner.enabled: raise ValueError('Base feature/teacher requested with editing ON')
        if type(module).__name__ in ('RoutedFullLinear','RoutedLoRALinear','GraceReplacementLayer'):
            raise ValueError('Base path must have no editing wrappers')


def source_batch(runtime,record,row):
    return runtime.build_edit_batch(replace(record,question=row['question'],image_path=Path(row['image_path']),
        target=row['reference'],official_rephrase=''))


def check_cache(cache,binding):
    if cache['binding']!=binding: raise ValueError('Teacher cache binding mismatch')
    logp=cache['logp']
    if logp.ndim!=2 or logp.dtype!=torch.float32 or logp.requires_grad or not torch.isfinite(logp).all(): raise ValueError('Invalid teacher distribution')
    if not torch.allclose(logp.logsumexp(-1),torch.zeros(len(logp)),atol=2e-5): raise ValueError('Teacher is not normalized log probability')
    return logp


def teachers_for(runtime,root,cfg,task,record):
    assert_base_off(runtime)
    teachers=[]
    for row in task['U_fit']:
        base,_,_,key=stage15.base_output(runtime,root,row,record)
        kwargs,labels,mask,inputs=teacher_batch(runtime,dict(row,eqkey=key),base['raw_token_ids'])
        binding=dict(runtime=cfg['runtime_lock'],generation=runtime.generation_config,code=cfg['code_commit'],
            source=row,input=inputs,teacher_tokens=base['raw_token_ids'],predictors=mask.nonzero().tolist(),
            teacher='FROZEN_BASE_ALL_EDITING_OFF',temperature=1.,direction='Base||student',vocabulary='FULL',
            vocab_size=runtime.model.config.vocab_size,dtype='float32',answer_cap=128,truncation=False)
        path=root/'private/teacher'/f'{key}.pt'
        if path.exists(): cache=torch.load(path,map_location='cpu',weights_only=True)
        else:
            with torch.no_grad(): logp=runtime.model(**kwargs).logits[mask].float().log_softmax(-1).cpu()
            cache=dict(binding=binding,logp=logp); save(path,cache)
        logp=check_cache(cache,binding)
        if logp.shape!=(int(mask.sum()),runtime.model.config.vocab_size): raise ValueError('Teacher shape mismatch')
        teachers.append((kwargs,labels,mask,logp,dict(row,teacher_binding=digest(binding))))
    return teachers


def backward_term(expert,loss,weight,tokens,sample):
    """Measure the actual term's added gradients without another model forward."""
    params=list(expert.parameters())
    before=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
    (weight*loss).backward()
    contribution=sum((p.grad-old).float().square().sum() for p,old in zip(params,before) if p.grad is not None).sqrt()
    if not torch.isfinite(contribution): raise FloatingPointError('Nonfinite gradient')
    return dict(unweighted=float(loss.detach()),weight=weight,weighted=float(loss.detach())*weight,
        tokens=tokens,weighted_gradient_norm=float(contribution),sample=sample)


def update(runtime,hook,expert,optimizer,native,fit,teacher,extra=None,extra_weight=1.):
    """One common L0 update, plus the matched H/G slot if supplied."""
    optimizer.zero_grad(set_to_none=True); terms={}
    for name,batch,weight in (('native',native,.5),('fit',fit,.5)):
        hook.set_teacher_routing(batch.labels)
        terms[name]=backward_term(expert,runtime.compute_loss(batch),weight,len(batch.target_token_ids),name)
    kwargs,labels,mask,logp,row=teacher
    hook.set_teacher_routing(labels)
    terms['U']=backward_term(expert,full_vocab_kl(runtime.model(**kwargs).logits[mask],logp),.01,int(mask.sum()),digest(row))
    if extra is not None:
        batch,row=extra; hook.set_teacher_routing(batch.labels)
        terms['extra']=backward_term(expert,runtime.compute_loss(batch),extra_weight,len(batch.target_token_ids),digest(row))
    grad=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
    if not torch.isfinite(grad) or any(p.grad is None for p in expert.parameters()): raise FloatingPointError('Invalid writer gradient')
    optimizer.step(); expert.normalize_factors_(verify_dense=False)
    if any(not torch.isfinite(p).all() for p in expert.parameters()): raise FloatingPointError('Invalid writer state')
    return dict(terms=terms,gradient_norm=float(grad))


def rng_state():
    return dict(torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def resume(path,binding,expert,optimizer):
    state=torch.load(path,map_location=next(expert.parameters()).device,weights_only=True)
    if state['binding']!=binding: raise ValueError('Resume binding mismatch')
    expert.load_state_dict(state['expert']); optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu()); random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None: torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    return state['step'],state['curve']


def extra_schedule(task):
    # A source-balanced H index chooses its paired G, giving identical extra slots.
    groups=defaultdict(list)
    for i,h in enumerate(task['H_fit']): groups[h['source_group']].append(dict(index=i))
    return [x['index'] for x in balanced_schedule(dict(sorted(groups.items())),320,task['seed'])]


def train(runtime,root,cfg,task,expert,branch,record,teachers,w0_hash,*,layer_id=None):
    validate_task(task)
    from methods.medtrace.hsic import layer_path
    writer_path=LAYER if layer_id is None else layer_path(layer_id)
    if branch not in BRANCHES or state_hash(expert)!=w0_hash: raise ValueError('Branch must start from identical W0')
    from m3bench_repro.editors.llava_runtime import seed_everything
    from m3bench_repro.editors.llava_runtime import write_json_atomic
    seed_everything(task['seed']); expert.requires_grad_(True)
    optimizer=optimizer_for(expert,runtime.model)
    directory=root/'private/edits'/f"e{task['order']:03d}"/branch; point=directory/'latest.pt'
    batches=[source_batch(runtime,record,task['native'])]+[runtime.build_edit_batch(replace(record,question=q)) for q in task['fit_questions']]
    hg={k:[source_batch(runtime,record,r) for r in task[k]] for k in ('H_fit','G_fit')}
    if not all(runtime.adapter.tokenizer.eos_token_id in b.target_token_ids for b in batches+hg['H_fit']+hg['G_fit']): raise ValueError('Answer EOS missing')
    # Same coarse answer-length stratum, not an assertion of equal compute.
    bin_for=lambda n:0 if n<=4 else 1 if n<=8 else 2 if n<=16 else 3
    if any(bin_for(len(h.target_token_ids))!=bin_for(len(g.target_token_ids)) for h,g in zip(hg['H_fit'],hg['G_fit'])): raise ValueError('H/G answer length stratum mismatch')
    fit_order=list(range(1,5)); random.Random(task['seed']).shuffle(fit_order); extra_order=extra_schedule(task)
    binding=dict(task=task,branch=branch,W0=w0_hash,code=cfg['code_commit'],runtime=cfg['runtime_lock'],
        generation=runtime.generation_config,fit_order=fit_order,extra_order=extra_order,steps=320,
        U_teacher_bindings=[digest([x[3].shape[0],x[4],cfg['runtime_lock'],cfg['code_commit']]) for x in teachers])
    if layer_id is not None: binding['layer_id']=layer_id
    curve=[]; step0=0
    if point.exists(): step0,curve=resume(point,binding,expert,optimizer)
    hook=MedTraceLayerHook(runtime.get_module(writer_path),expert); hook.attach(); began=time.time()
    try:
        for step in range(step0+1,321):
            stage15.budget(root,cfg)
            role={'C_FACT':'H_fit','C_EXTRA':'G_fit'}.get(branch); i=extra_order[step-1]
            extra=(hg[role][i],task[role][i]) if role else None
            item=update(runtime,hook,expert,optimizer,batches[0],batches[fit_order[(step-1)%4]],teachers[(step-1)%len(teachers)],extra)
            item['terms']['native']['sample']=digest(task['native'])
            item['terms']['fit']['sample']=digest([task['native'],task['fit_questions'][fit_order[(step-1)%4]-1]])
            item.update(step=step,fit_index=fit_order[(step-1)%4],extra_index=i if role else None,extra_role=role)
            curve.append(item)
            if step%20==0:
                save(point,dict(binding=binding,expert=expert.state_dict(),optimizer=optimizer.state_dict(),step=step,curve=curve,**rng_state()))
                write_json_atomic(root/'public/PROGRESS.json',dict(status='RUNNING',order=task['order'],branch=branch,phase='CONTINUATION',step=step,total_steps=320))
                print('UPDATE',task['order'],branch,step,flush=True)
        loaded=torch.load(point,map_location=runtime.device,weights_only=True)
        if any(not torch.equal(v,loaded['expert'][k]) for k,v in expert.state_dict().items()): raise ValueError('Saved writer mismatch')
        extra_grad=[c['terms']['extra']['weighted_gradient_norm'] for c in curve if 'extra' in c['terms']]
        if role and not any(g>0 for g in extra_grad): raise ValueError('H/G did not contribute any gradient')
        write_json_atomic(directory/'TRAINING.json',dict(status='COMPLETE',steps=320,branch=branch,W0=w0_hash,
            resumed_from=step0,session_seconds=time.time()-began,extra_gradient_nonzero_steps=sum(g>0 for g in extra_grad),
            tokens={k:sum(c['terms'][k]['tokens'] for c in curve) for k in curve[0]['terms']},
            parameters=sum(p.numel() for p in expert.parameters()),curve=curve))
    finally: hook.detach()
    expert.requires_grad_(False)
    return expert
