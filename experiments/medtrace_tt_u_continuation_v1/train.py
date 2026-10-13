"""TT88 native/fit and optional full-vocabulary Base||student KL only."""
import copy
import math
from dataclasses import replace
import fcntl
from pathlib import Path
import random
import torch
from common import RUN,LAYER,read,write,save,digest,state_hash,rng,restore_rng,diagnostic_scope,budget,local_path
from structures import TT4,optimizer_for
from methods.medtrace.core import MedTraceLayerHook
from methods.medtrace.selective_write import full_vocab_kl,predictor_mask

def clone(state,seed,device):
    assert set(state)=={'G1','G2','G3','G4'} and sum(v.numel() for v in state.values())==7168
    e=TT4(seed,8,8).to(device);e.load_state_dict(state);return e

def assert_base_off(runtime):
    assert not any(p.requires_grad for p in runtime.model.parameters())
    for m in runtime.model.modules():
        for fn in m._forward_hooks.values():
            owner=getattr(fn,'__self__',None)
            assert not isinstance(owner,MedTraceLayerHook) or not owner.enabled,'Teacher requires all edits OFF'
        assert type(m).__name__ not in ('RoutedFullLinear','RoutedLoRALinear','GraceReplacementLayer')

def teacher_batch(runtime,row,tokens):
    # Exact shared teacher-prefix causal expansion; no answer re-tokenization.
    raw=runtime.adapter.prepare_inputs(Path(local_path(row['image_path'])),row['question'],None)
    answer=torch.tensor([tokens],device=runtime.device,dtype=raw['input_ids'].dtype)
    ids=torch.cat([raw['input_ids'],answer],1);attention=torch.cat([raw['attention_mask'],torch.ones_like(answer)],1)
    labels=torch.cat([torch.full_like(raw['input_ids'],-100),answer],1)
    embeds,attention,positions,labels=runtime._expand_multimodal(raw_input_ids=ids,attention_mask=attention,labels=labels,images=raw['images'])
    mask=predictor_mask(labels,attention)
    assert int(mask.sum())==len(tokens) and labels[0,labels[0]!=-100].tolist()==tokens
    kwargs=dict(inputs_embeds=embeds.detach(),attention_mask=attention,position_ids=positions,labels=None,use_cache=False,return_dict=True)
    binding=dict(prompt_tokens=raw['input_ids'][0].tolist(),image=raw['image_sha256'],image_shape=list(embeds.shape),positions=positions.tolist() if positions is not None else None,attention=attention.tolist() if attention is not None else None,labels=labels.tolist(),tokens=tokens,predictor_positions=mask.nonzero().tolist(),dtype=str(embeds.dtype))
    return kwargs,labels,mask,binding

def teachers_for(runtime,rows):
    assert_base_off(runtime);teachers=[]
    for row in rows:
        budget();key=digest(row)
        with (RUN/'private/teacher'/('lock.'+key)).open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);path=RUN/'private/teacher'/(key+'.pt')
            if path.exists():cache=torch.load(path,map_location='cpu',weights_only=True);tokens=cache['tokens']
            else:
                raw=runtime.adapter.prepare_inputs(Path(local_path(row['image_path'])),row['question'],None)
                with torch.inference_mode():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                tokens=list(g.raw_token_ids);cache=None
            assert 0<len(tokens)<=128 and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,'Teacher must end with actual EOS; no truncation'
            kwargs,labels,mask,inputs=teacher_batch(runtime,row,tokens)
            binding=dict(row=row,input=inputs,runtime=read(RUN/'private/RUNTIME_BINDING.json'),generation=runtime.generation_config,teacher='FROZEN_BASE_ALL_EDITING_OFF',temperature=1.,direction='Base||student',vocabulary='FULL',answer_cap=128,truncation=False)
            if cache is None:
                with torch.no_grad():logp=runtime.model(**kwargs).logits[mask].float().log_softmax(-1).cpu()
                cache=dict(binding=binding,tokens=tokens,logp=logp,raw_answer=g.decoded_text);save(path,cache)
            assert cache['binding']==binding
            logp=cache['logp'];assert not logp.requires_grad and logp.shape==(int(mask.sum()),runtime.model.config.vocab_size) and torch.isfinite(logp).all()
            assert torch.allclose(logp.logsumexp(-1),torch.zeros(len(logp)),atol=2e-5)
            teachers.append((kwargs,labels,mask,logp,row,cache))
    return teachers

def grad(e):return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten().clone() for p in e.parameters()])
def update(runtime,hook,e,opt,native,fit,teacher=None,u_weight=.01):
    assert math.isfinite(u_weight) and u_weight>0
    opt.zero_grad(set_to_none=True);terms={};vectors=[];before={k:v.detach().clone() for k,v in e.state_dict().items()}
    for name,batch in [('native',native),('fit',fit)]:
        old=grad(e);hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CE')
        (.5*loss).backward();vectors.append(grad(e)-old)
        terms[name]=dict(unweighted=float(loss.detach()),weighted=.5*float(loss.detach()),tokens=len(batch.target_token_ids),weighted_gradient_norm=float(vectors[-1].norm()))
    ce=vectors[0]+vectors[1];uv=torch.zeros_like(ce)
    if teacher is not None:
        kwargs,labels,mask,logp,row,_=teacher;old=grad(e);hook.set_teacher_routing(labels)
        loss=full_vocab_kl(runtime.model(**kwargs).logits[mask],logp);(u_weight*loss).backward();uv=grad(e)-old
        terms['U']=dict(unweighted=float(loss.detach()),weighted=u_weight*float(loss.detach()),tokens=int(mask.sum()),source=digest(row['source_group']),weighted_gradient_norm=float(uv.norm()),unweighted_gradient_norm=float(uv.norm())/u_weight,zero_reason='EXACT_ZERO_OR_NUMERICAL_ZERO' if uv.norm()==0 else None)
    norm=torch.nn.utils.clip_grad_norm_(e.parameters(),1.)
    assert torch.isfinite(norm) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters())
    post=float(grad(e).norm());opt.step();assert all(torch.isfinite(p).all() for p in e.parameters())
    return dict(terms=terms,CE_gradient_norm=float(ce.norm()),U_gradient_norm=float(uv.norm())/u_weight,U_weighted_fraction=float(uv.norm()/(ce.norm()+uv.norm())) if ce.norm()+uv.norm()>0 else None,CE_U_cosine=float(torch.nn.functional.cosine_similarity(ce[None],uv[None]).item()) if ce.norm()>0 and uv.norm()>0 else None,preclip_norm=float(norm),postclip_norm=post,core_updates={k:float((v-before[k]).norm()) for k,v in e.state_dict().items()},learning_rates=[g['lr'] for g in opt.param_groups],Adam_steps=[int(s['step']) for s in opt.state.values()],forwards=2+int(teacher is not None),backwards=2+int(teacher is not None),train_tokens=sum(x['tokens'] for x in terms.values()))

def activations(runtime,hook,batches,teachers):
    result=[]
    def capture(_,args):result.append(args[0][hook.token_mask].detach().float()[:16].clone())
    with diagnostic_scope(hook),torch.no_grad():
        handle=runtime.get_module(LAYER).register_forward_pre_hook(capture)
        try:
            for b in batches:hook.set_teacher_routing(b.labels);runtime.compute_loss(b)
            for kwargs,labels,*_ in teachers:hook.set_teacher_routing(labels);runtime.model(**kwargs)
        finally:handle.remove()
    return torch.cat(result)
def functional(e,x):
    with torch.no_grad():return e.residual(x).float().detach()

def continuation(runtime,t,record,init,directory,teachers,steps,nodes,fixed_activations=None):
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert set(t)=={'edit_id','order','native','fit_questions','seed','U_fit','bindings'},'Training API accepts explicit native/fit/U only'
    assert bool(t['U_fit'])==bool(teachers)
    if teachers:assert all(x[4]['role']=='FIT' for x in teachers)
    e=clone(init,t['seed'],runtime.device);opt=optimizer_for(e,runtime.model);assert not opt.state
    batches=[runtime.build_edit_batch(record)]+[runtime.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
    assert len(batches)==5 and all(runtime.adapter.tokenizer.eos_token_id==b.target_token_ids[-1] for b in batches)
    fit=list(range(1,5));random.Random(t['seed']).shuffle(fit)
    binding=dict(task=t,W0=state_hash(e),steps=steps,nodes=nodes,fit_order=fit,teacher_bindings=[digest(x[5]['binding']) for x in teachers],execution=read(RUN/'private/GPU_SOURCE_VERSION.json'),runtime=read(RUN/'private/RUNTIME_BINDING.json'))
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);latest=directory/'latest.pt'
    if (directory/'TRAINING.json').exists():assert read(directory/'TRAINING.json')['binding']==binding;return
    hook=MedTraceLayerHook(runtime.get_module(LAYER),e);hook.attach();seed_everything(t['seed']);curve=[];start=0;began=__import__('time').time()
    try:
        x=activations(runtime,hook,batches,teachers) if fixed_activations is None else fixed_activations;f0=functional(e,x);previous=f0;path=0.
        if latest.exists():
            s=torch.load(latest,map_location='cpu',weights_only=True);assert s['binding']==binding
            e.load_state_dict(s['expert']);opt.load_state_dict(s['optimizer']);restore_rng(s);start=s['step'];curve=s['curve'];path=s['function_path'];previous=functional(e,x)
        for step in range(start+1,steps+1):
            budget();teacher=teachers[(step-1)%len(teachers)] if teachers else None
            item=update(runtime,hook,e,opt,batches[0],batches[fit[(step-1)%4]],teacher);current=functional(e,x)
            distance=float((current-previous).norm());path+=distance;previous=current
            item.update(step=step,fit_index=fit[(step-1)%4],net_function_change=float((current-f0).norm()),cumulative_function_path=path,function_step=distance);curve.append(item)
            if step%20==0 or step in nodes:
                save(latest,dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),step=step,curve=curve,function_path=path,**rng()))
                print('UPDATE',t['order'],directory.name,step,flush=True)
            if step in nodes:save(directory/('step'+str(step)+'.pt'),dict(binding=binding,expert=e.state_dict(),step=step,state_hash=state_hash(e)))
        write(directory/'TRAINING.json',dict(status='COMPLETE',binding=binding,curve=curve,steps=steps,seconds=__import__('time').time()-began,actual_updates=steps,parameter_count=7168,forwards=sum(x['forwards'] for x in curve)+5+len(teachers),backwards=sum(x['backwards'] for x in curve),tokens=sum(x['train_tokens'] for x in curve),final_state_hash=state_hash(e),U_nonzero_updates=sum(x.get('terms',{}).get('U',{}).get('weighted_gradient_norm',0)>0 for x in curve)))
        latest.unlink()
    finally:hook.detach()
