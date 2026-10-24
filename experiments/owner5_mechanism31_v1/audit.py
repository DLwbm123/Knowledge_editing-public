"""Fixed trajectory reconstruction with post-freeze intermediate semantic diagnostics."""
import os,sys,json,time,traceback,subprocess,copy
from pathlib import Path
from types import SimpleNamespace
P=Path(os.environ['PARENT_RUN']);OUT=Path(os.environ['OUTPUT_RUN'])
sys.path[:0]=[str(OUT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
import torch
import module_run as r
import trajectory,training_math
import owner5_semantic_loss as semantic
from prefix_cache import PrefixCache
from scripts.medtrace.run_realmodel_core import load_real_runtime
from m3bench_repro.editors.llava_runtime import seed_everything

def read(p):return json.loads(Path(p).read_text())
def write(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
def save(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def budget():
    assert not (OUT/'STOP').exists(),'Explicit stop'
    import shutil
    assert shutil.disk_usage(OUT).free>8*1024**3
    assert time.time()<float(os.environ['DEADLINE_EPOCH']),'Finite chain deadline'
    assert counts['updates']<=62 and counts['backwards']<=400 and counts['operator_backwards']<=100 and counts['generations']==0
    assert counts['forwards']<=12000

def prepare(runtime,bindings,task,rows,training):
    runtime.model.half()
    items=r.s.prepare(runtime,bindings,task,rows,training)
    for row,batch,raw,expanded,binding in items:
        name=row['query_id']
        write(('training/' if training else 'evaluation/')+'bindings/'+name+'.json',dict(binding,prompt=raw['prompt'],preprocessing_seed=raw['preprocessing_seed']))
        save(OUT/(('train_inputs_' if training else 'eval_inputs_')+name+'.pt'),dict(expanded=expanded,prompt_ids=raw['input_ids'].cpu(),images=raw['images'].cpu() if isinstance(raw['images'],torch.Tensor) else [x.cpu() for x in raw['images']]))
    r.old.fp32(runtime)
    return items

def separated_step(before,edit,retention,cache,coefficient,scalar,target):
    params={k:v.detach().clone().requires_grad_(True) for k,v in before.items()}
    penalty=r.mm.operator_penalty(params,cache)
    extra=dict(zip(params,torch.autograd.grad(coefficient*penalty,tuple(params.values()))))
    from balanced_fisher import cap_gradient,remove_opposing
    extra,balance=cap_gradient(edit,extra,r.mm.GROUPS)
    extra,direction=remove_opposing(edit,extra,r.mm.GROUPS)
    for k,v in params.items():v.grad=edit[k].detach()+extra[k].detach()+retention[k].detach()
    norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
    assert torch.isfinite(norm) and all(torch.isfinite(v.grad).all() for v in params.values())
    proposed,updated=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in params.items()},copy.deepcopy(scalar))
    if r.fm.map_norm(before,proposed)<=1e-14:selected={k:v.clone() for k,v in before.items()};matching=dict(status='ZERO_DIRECTION')
    else:proposed,matching=r.match_proposal(before,proposed,target);selected=r.mm.canonicalize(proposed)[0]
    return selected,updated,dict(gradient_norm=float(norm),balance=balance,anti_parallel=direction,matching=matching)

def stats(a,b):
    x=torch.cat([v.detach().double().flatten() for v in a.values()]);y=torch.cat([b[k].detach().double().flatten() for k in a])
    return dict(first_norm=float(x.norm()),second_norm=float(y.norm()),cosine=float(x.dot(y)/(x.norm()*y.norm())) if x.norm()>0 and y.norm()>0 else None)

def main():
    with (OUT/'ATTEMPT').open('x') as f:f.write(str(os.getpid()))
    write('SELF_CHECK.json',semantic.selfcheck())
    config=read(OUT/'CONFIG.json');assert config['owner']==5 and config['steps']==31
    torch.use_deterministic_algorithms(True)
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval()
    assert not any(p.requires_grad for p in runtime.model.parameters())
    assert not runtime.get_module(r.c.LAYER)._forward_hooks
    bindings=read(P/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
    write('RUNTIME.json',dict(generation=runtime.generation_config,inherited_runtime=next(iter(bindings.values()))['runtime'],image_processor=runtime.adapter.image_processor.to_dict(),torch=torch.__version__,precision='MATH_FP32_FROZEN_FP16_PREFILL'))
    commands=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True)
    assert commands.strip().endswith(sys.executable+' -u -')
    write('PROCESS.json',dict(commands=commands,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)))
    packet=read(P/'private/M13_SOURCE_PACKET.json');task=packet['original_tasks'][4]
    source=trajectory.source_rows(packet,5)
    train_rows=read(OUT/'CONSTRUCT.json');assert len(train_rows)==2
    assert train_rows[0]['query_id']=='positive_construct' and train_rows[1]['query_id']=='negative_construct'
    items=prepare(runtime,bindings,task,source+train_rows,True)
    batches=[r.old.batch_on(runtime,item[1]) for item in items]
    candidate_rows=[dict(train_rows[1],query_id='semantic_'+str(i),reference=text)
                    for i,text in enumerate(semantic.ANSWERS)]
    candidate_items=prepare(runtime,bindings,task,candidate_rows,True)
    candidate_batches=[]
    for row,value,raw,_,binding in candidate_items:
        kwargs,labels,tokens=value
        prefix=raw['input_ids'][0].tolist()
        full=runtime._tokenize_prompt(raw['prompt']+' '+row['reference'])[0].tolist()
        assert full[:len(prefix)]==prefix
        content=tuple(full[len(prefix):]);assert content and tuple(tokens[:len(content)])==content
        assert tuple(tokens[len(content):])==(28705,2), 'Unexpected template tail'
        positions=torch.where(labels[0]!=-100)[0]
        labels=labels.clone();labels[0,positions[len(content):]]=-100
        kwargs=dict(kwargs,labels=labels)
        candidate_batches.append(r.old.batch_on(runtime,(kwargs,labels,content)))
        write('semantic_bindings/'+row['query_id']+'.json',dict(binding,content_token_ids=list(content),
              labels=labels.tolist(),prediction_positions=(positions[:len(content)]-1).tolist()))
    initial=r.p.tr.TT4(task['seed'],8,8).state_dict()
    saved=r.p.load_state(P/'private/candidates/C/5.pt')['states']
    assert all(torch.equal(initial[k],saved['BASE'][k]) for k in initial)
    targets=[v['operator_step_norm'] for v in read(P/'private/results/C/5.json')['curve']];assert len(targets)==160
    coefficient=read(P/'private/calibration146/5.json')['coefficient']
    cache={k:v.to(runtime.device) if isinstance(v,torch.Tensor) else v for k,v in r.p.load_state(P/'private/OPERATOR_FISHER.pt').items()}
    expert=r.p.expert(initial,task['seed'],runtime.device)
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
    handle=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
    try:
        def score():
            values=[]
            with torch.no_grad():
                for batch in candidate_batches:
                    hook.set_teacher_routing(batch.labels)
                    values.append(float(runtime.compute_loss(batch)))
            return semantic.summary(values)
        def evidence(label):
            records=[]
            for index,batch in enumerate(candidate_batches):
                positions=torch.where(batch.labels[0]!=-100)[0]
                hook.set_teacher_routing(batch.labels)
                captured={}
                def capture(_m,_a,output):
                    logits=output.logits[0,positions-1].detach().float()
                    losses=torch.nn.functional.cross_entropy(logits,batch.labels[0,positions],reduction='none')
                    captured.update(logits=logits.cpu(),labels=batch.labels[0,positions].cpu(),losses=losses.cpu())
                h=runtime.model.register_forward_hook(capture)
                try:
                    with torch.no_grad():mean=float(runtime.compute_loss(batch))
                finally:h.remove()
                save(OUT/(label+'_candidate_'+str(index)+'.pt'),dict(captured,model_mean_nll=mean))
                assert abs(float(captured['losses'].mean())-mean)<2e-6
                records.append(dict(mean_nll=mean,token_losses=captured['losses'].tolist()))
            result=dict(semantic.summary([v['mean_nll'] for v in records]),records=records)
            write(label+'_SEMANTIC.json',result);return result
        base=r.mm.canonicalize(initial)[0]
        expert.load_state_dict(base)
        with PrefixCache(runtime,batches+candidate_batches):
            baseline=evidence('BASE');s0=baseline['score']
            write('ADMISSION.json',dict(score=s0,positive=s0>0,lambda_protection=1.0))
            if s0<=0:
                write('COMPLETE.json',dict(status='PROXY_ADMISSION_FAILED',counts=counts,automatic_next_run=False))
                return
            for arm in ('CONTROL','PROTECTED'):
                seed_everything(task['seed']);expert.load_state_dict(base)
                scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
                with (OUT/('ONCE_'+arm)).open('x'):pass
                for step in range(1,32):
                    budget();before=r.old.snapshot(expert);scalar_before=copy.deepcopy(scalar);expert.zero_grad(set_to_none=True);losses=[]
                    for index,weight in ((0,.25),(1+(step-1)%4,.25),(5,.5)):
                        hook.set_teacher_routing(batches[index].labels)
                        loss=runtime.compute_loss(batches[index]);assert torch.isfinite(loss)
                        (weight*loss).backward();counts['backwards']+=1;losses.append(float(loss.detach()))
                    edit={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                    contrast=score();hinge=max(0.,s0-contrast['score'])
                    applied=arm=='PROTECTED' and hinge>0
                    if applied:
                        # Sequential backwards is exactly the active hinge gradient, without retaining six graphs.
                        for index,batch in enumerate(candidate_batches):
                            hook.set_teacher_routing(batch.labels)
                            loss=runtime.compute_loss(batch)
                            (semantic.GRADIENT_WEIGHTS[index]*loss).backward();counts['backwards']+=1
                    gradients={k:v.grad.detach().clone() for k,v in expert.named_parameters()}
                    proposal,scalar,audit=training_math.step(r,before,gradients,cache,coefficient,scalar,targets[step-1],'M13P')
                    counts['operator_backwards']+=1
                    if step==1:
                        if arm=='CONTROL':first={k:v.cpu().clone() for k,v in proposal.items()}
                        else:assert all(torch.equal(first[k],v.cpu()) for k,v in proposal.items()),'Matched first proposals differ'
                    if step in (2,12,19,31):
                        retention={k:torch.zeros_like(v) for k,v in edit.items()}
                        if hinge>0:
                            for index,batch in enumerate(candidate_batches):
                                hook.set_teacher_routing(batch.labels)
                                value=runtime.compute_loss(batch)
                                gs=torch.autograd.grad(semantic.GRADIENT_WEIGHTS[index]*value,tuple(expert.parameters()))
                                counts['backwards']+=1
                                for (key,_),g in zip(expert.named_parameters(),gs):retention[key].add_(g.detach())
                        if applied:assert all(torch.allclose(gradients[k],edit[k]+retention[k],atol=1e-6,rtol=1e-4) for k in edit)
                        else:assert all(torch.equal(gradients[k],edit[k]) for k in edit)
                        mixed={k:edit[k]+retention[k] for k in edit}
                        variant={}
                        for label,grads in (('EDIT_ONLY',edit),('LEGACY_MIXED',mixed)):
                            variant[label]=training_math.step(r,before,grads,cache,coefficient,scalar_before,targets[step-1],'M13P');counts['operator_backwards']+=1
                        variant['SEPARATED']=separated_step(before,edit,retention,cache,coefficient,scalar_before,targets[step-1]);counts['operator_backwards']+=1
                        zero=separated_step(before,edit,{k:torch.zeros_like(v) for k,v in edit.items()},cache,coefficient,scalar_before,targets[step-1]);counts['operator_backwards']+=1
                        assert all(torch.equal(zero[0][k],variant['EDIT_ONLY'][0][k]) for k in edit),'Zero-retention interface parity failed'
                        measures={}
                        for label,(state,_,details) in variant.items():
                            expert.load_state_dict(state);sc=score();ls=[]
                            with torch.no_grad():
                                for index,weight in ((0,.25),(1+(step-1)%4,.25),(5,.5)):
                                    hook.set_teacher_routing(batches[index].labels);ls.append(weight*float(runtime.compute_loss(batches[index])))
                            measures[label]=dict(score=sc,weighted_edit_loss=sum(ls),operator_norm=r.fm.map_norm(before,state),details=details)
                        delta=r.fm.map_norm(variant['EDIT_ONLY'][0],variant['SEPARATED'][0])
                        write('diagnostics/'+arm+'_'+str(step)+'.json',dict(step=step,hinge=hinge,gradient_stats=stats(edit,retention),variants=measures,nonzero_proposal_effect=delta>1e-12,proposal_distance=delta,zero_retention_parity=True,gradient_addition_checked=True))
                        save(OUT/('gradients_'+arm+'_'+str(step)+'.pt'),dict(before={k:v.cpu() for k,v in before.items()},edit={k:v.cpu() for k,v in edit.items()},semantic={k:v.cpu() for k,v in retention.items()},scalar=scalar_before))
                    selected=proposal
                    guard=dict(score=contrast['score'],pair_margins=contrast['pair_margins'],
                               candidate_mean_nll=contrast['candidate_mean_nll'],hinge=hinge,applied=applied)
                    expert.load_state_dict(selected);counts['updates']+=1
                    curve.append(dict(step=step,losses=losses,guard=guard,proposal=audit,operator_step_norm=r.fm.map_norm(before,selected)))
                    if step%10==0 or step==31:write('PROGRESS.json',dict(arm=arm,step=step,counts=counts));print(arm,step,flush=True)
                final=r.old.snapshot(expert);save(OUT/(arm+'.pt'),dict(state={k:v.cpu() for k,v in final.items()}))
                disk=r.p.load_state(OUT/(arm+'.pt'))['state'];assert all(torch.equal(disk[k],v.cpu()) for k,v in final.items())
                write(arm+'_TRAIN.json',dict(status='COMPLETE',steps=31,curve=curve,save_load_exact=True,semantic=evidence(arm),base_semantic_score=s0,scalar=scalar))
        write('FROZEN.json',dict(arms=['CONTROL','PROTECTED'],steps=31,check_data_accessed=False,counts=counts))
        hook.detach()
        parent=Path(os.environ['REFERENCE_RUN'])
        for arm in ('CONTROL','PROTECTED'):
            previous=read(parent/(arm+'_TRAIN.json'))['curve'][:31]
            current=read(OUT/(arm+'_TRAIN.json'))['curve'];assert previous==current,'Prefix trajectory mismatch'
        diagnostics=[read(p) for p in sorted((OUT/'diagnostics').glob('*.json'))];assert len(diagnostics)==8
        assert counts['updates']==62 and counts['generations']==0
        influence=any(v['nonzero_proposal_effect'] for v in diagnostics)
        write('ADMISSION.json',dict(status='PASS' if influence else 'NO_MEASURABLE_UPDATE_EFFECT',trajectory_exact=True,zero_retention_parity=True,gradient_addition_checked=True,nonzero_semantic_update=influence,diagnostic_cells=8))
        write('COMPLETE.json',dict(status='COMPLETE',counts=counts,seconds=time.time()-START,automatic_retry=False))
        return runtime
    finally:
        hook.detach();handle.remove()

START=time.time();counts=dict(updates=0,backwards=0,operator_backwards=0,generations=0,forwards=0)
# Reuse pure computation helpers without invoking or charging a historical run.
r.c.budget=budget
code=0
try:
    runtime=main()
    while time.time()<float(os.environ['DEADLINE_EPOCH']) and not (OUT/'RELEASE_RESIDENCY').exists():
        write('RESIDENCY.json',dict(time=time.time(),status='AUDIT_COMPLETE_MODEL_RESIDENT'))
        time.sleep(10)
except BaseException:
    code=1;write('FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
finally:write('EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
