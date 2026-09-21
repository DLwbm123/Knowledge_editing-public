# Public sanitized copy of the fixed runtime source. Private data paths and bindings are placeholders.
import sys,time,random
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from methods.medtrace import MedTraceLayerHook
from methods.medtrace.selective_write import optimizer_for,full_vocab_kl
from scripts.medtrace.stage18_cfact import (validate_task,state_hash,source_batch,extra_roles,extra_schedule,resume,rng_state,backward_term,BRANCHES)
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.run_selective_write import save
from scripts.medtrace import stage15
LAYER='model.layers.30.mlp.down_proj'


def update(runtime,hook,expert,optimizer,native,fit,teacher,extra=None,extra_weight=1.,U_weight=.01,diagnostic=False):
    optimizer.zero_grad(set_to_none=True);terms={};vectors={};params=list(expert.parameters())
    def term(name,loss,weight,tokens,sample):
        before=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params] if diagnostic else None
        terms[name]=backward_term(expert,loss,weight,tokens,sample)
        if diagnostic:vectors[name]=torch.cat([(p.grad-before[i]).detach().flatten() for i,p in enumerate(params)])
    for name,batch,weight in (('native',native,.5),('fit',fit,.5)):
        hook.set_teacher_routing(batch.labels);term(name,runtime.compute_loss(batch),weight,len(batch.target_token_ids),name)
    if teacher is not None and U_weight != 0:
        kwargs,labels,mask,logp,row=teacher; hook.set_teacher_routing(labels)
        term('U',full_vocab_kl(runtime.model(**kwargs).logits[mask],logp),U_weight,int(mask.sum()),digest(row))
    if extra is not None:
        batch,row=extra;hook.set_teacher_routing(batch.labels);term('extra',runtime.compute_loss(batch),extra_weight,len(batch.target_token_ids),digest(row))
    cosine=None
    if diagnostic and 'extra' in vectors:
        def cos(a,b):
            denominator=a.norm()*b.norm()
            return float((a@b)/denominator) if float(denominator)>0 else None
        cosine=dict(H_U=cos(vectors['extra'],vectors['U']),H_edit=cos(vectors['extra'],vectors['native']+vectors['fit']),definition='cosine of actual weighted incremental gradients before clipping',zero_norm='NA')
    grad=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
    if not torch.isfinite(grad) or any(p.grad is None for p in params):raise FloatingPointError('Invalid writer gradient')
    optimizer.step();expert.normalize_factors_(verify_dense=False)
    if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('Invalid writer state')
    return dict(terms=terms,gradient_norm=float(grad),gradient_cosines=cosine)


def train(runtime,root,cfg,task,expert,branch,record,teachers,w0_hash,*,layer_id=30):
    weights=cfg["weights"];assert layer_id==30 and weights in [dict(H=h,U=u) for h,u in ((0,.01),(1,.01),(.25,.01),(0,.05),(1,.05),(.25,.05))]
    assert branch==("C_FACT" if weights["H"] else "C_NO_H")
    fasttrack = cfg.get('mode') == 'STAGE19_FASTTRACK_TWO_ARMS'
    if fasttrack and branch not in ('C_FACT','C_NO_H'): raise ValueError('Unauthorized FASTTRACK branch')
    validate_task(task, fasttrack_branch=branch if fasttrack else None)
    from methods.medtrace.hsic import layer_path
    writer_path=LAYER if layer_id is None else layer_path(layer_id)
    if branch not in BRANCHES or state_hash(expert)!=w0_hash: raise ValueError('Branch must start from identical W0')
    from m3bench_repro.editors.llava_runtime import seed_everything
    from m3bench_repro.editors.llava_runtime import write_json_atomic
    seed_everything(task['seed']); expert.requires_grad_(True)
    optimizer=optimizer_for(expert,runtime.model)
    directory=root/'private/edits'/f"e{task['order']:03d}"/branch; point=directory/'latest.pt'
    batches=[source_batch(runtime,record,task['native'])]+[runtime.build_edit_batch(replace(record,question=q)) for q in task['fit_questions']]
    hg={k:[source_batch(runtime,record,r) for r in task[k]] if k in extra_roles(fasttrack) else [] for k in ('H_fit','G_fit')}
    if not all(runtime.adapter.tokenizer.eos_token_id in b.target_token_ids for b in batches+hg['H_fit']+hg['G_fit']): raise ValueError('Answer EOS missing')
    # Same coarse answer-length stratum, not an assertion of equal compute.
    bin_for=lambda n:0 if n<=4 else 1 if n<=8 else 2 if n<=16 else 3
    if any(bin_for(len(h.target_token_ids))!=bin_for(len(g.target_token_ids)) for h,g in zip(hg['H_fit'],hg['G_fit'])): raise ValueError('H/G answer length stratum mismatch')
    fit_order=list(range(1,5)); random.Random(task['seed']).shuffle(fit_order)
    extra_order=extra_schedule(task) if task['H_fit'] else [0]*320
    binding=dict(task=task,branch=branch,W0=w0_hash,code=cfg['code_commit'],runtime=cfg['runtime_lock'],
        generation=runtime.generation_config,fit_order=fit_order,extra_order=extra_order,steps=320,weights=weights,diagnostic_steps=[1,80,160,320],
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
            item=update(runtime,hook,expert,optimizer,batches[0],batches[fit_order[(step-1)%4]],teachers[(step-1)%len(teachers)],extra,extra_weight=weights['H'],U_weight=weights['U'],diagnostic=step in (1,80,160,320))
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
