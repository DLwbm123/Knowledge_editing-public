"""Owner-five matched semantic contrast diagnostic; fixed 160 steps and lambda one."""
import os,sys,json,time,traceback,subprocess,copy
from pathlib import Path
from types import SimpleNamespace
P=Path(os.environ['PARENT_RUN']);OUT=Path(os.environ['OUTPUT_RUN'])
sys.path[:0]=[str(OUT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
import torch
import module_run as r
import trajectory,training_math,semantic
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
    assert time.time()-START<6*3600,'Six-hour finite run bound; no automatic restart'
    assert counts['updates']<=320 and counts['backwards']<=1920 and counts['generations']<=27
    assert counts['forwards']<=60000

def prepare(runtime,bindings,task,rows,training):
    runtime.model.half()
    items=r.s.prepare(runtime,bindings,task,rows,training)
    for row,batch,raw,expanded,binding in items:
        name=row['query_id']
        write(('training/' if training else 'evaluation/')+'bindings/'+name+'.json',dict(binding,prompt=raw['prompt'],preprocessing_seed=raw['preprocessing_seed']))
        save(OUT/(('train_inputs_' if training else 'eval_inputs_')+name+'.pt'),dict(expanded=expanded,prompt_ids=raw['input_ids'].cpu(),images=raw['images'].cpu() if isinstance(raw['images'],torch.Tensor) else [x.cpu() for x in raw['images']]))
    r.old.fp32(runtime)
    return items

def main():
    with (OUT/'ATTEMPT').open('x') as f:f.write(str(os.getpid()))
    write('SELF_CHECK.json',semantic.selfcheck())
    config=read(OUT/'CONFIG.json');assert config['owner']==5 and config['steps']==160
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
                for step in range(1,161):
                    budget();before=r.old.snapshot(expert);expert.zero_grad(set_to_none=True);losses=[]
                    for index,weight in ((0,.25),(1+(step-1)%4,.25),(5,.5)):
                        hook.set_teacher_routing(batches[index].labels)
                        loss=runtime.compute_loss(batches[index]);assert torch.isfinite(loss)
                        (weight*loss).backward();counts['backwards']+=1;losses.append(float(loss.detach()))
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
                    selected=proposal
                    guard=dict(score=contrast['score'],pair_margins=contrast['pair_margins'],
                               candidate_mean_nll=contrast['candidate_mean_nll'],hinge=hinge,applied=applied)
                    expert.load_state_dict(selected);counts['updates']+=1
                    curve.append(dict(step=step,losses=losses,guard=guard,proposal=audit,operator_step_norm=r.fm.map_norm(before,selected)))
                    if step%20==0:
                        save(OUT/('active_'+arm+'.pt'),dict(state=r.old.snapshot(expert),scalar=copy.deepcopy(scalar),step=step,rng=r.c.rng(),curve=curve))
                        write('PROGRESS.json',dict(arm=arm,step=step,counts=counts));print(arm,step,flush=True)
                final=r.old.snapshot(expert);save(OUT/(arm+'.pt'),dict(state={k:v.cpu() for k,v in final.items()}))
                disk=r.p.load_state(OUT/(arm+'.pt'))['state'];assert all(torch.equal(disk[k],v.cpu()) for k,v in final.items())
                write(arm+'_TRAIN.json',dict(status='COMPLETE',steps=160,curve=curve,save_load_exact=True,semantic=evidence(arm),base_semantic_score=s0,scalar=scalar))
        write('FROZEN.json',dict(arms=['CONTROL','PROTECTED'],steps=160,check_data_accessed=False,counts=counts))
        hook.detach()
        # Check images and labels become available only after both endpoints freeze.
        checks=read(OUT/'CHECK.json');assert [v['query_id'] for v in checks]==['positive_check','negative_check']
        evaluation=prepare(runtime,bindings,task,source+train_rows+checks,False)
        def emit(label,item,active):
            row,_,raw,expanded,binding=item;dest='outputs/'+label+'/'+row['query_id']+'.json'
            assert not (OUT/dest).exists(),'No regeneration'
            write('attempts/'+label+'_'+row['query_id']+'.json',dict(attempt=1,time=time.time()))
            counts['generations']+=1;budget()
            answer,trace=r.s.fit.short.generate(runtime,active,raw,expanded)
            write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),generation_trace=trace,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024))
        for item in evaluation:
            emit('BASE',item,None)
        for arm in ('CONTROL','PROTECTED'):
            expert.load_state_dict(r.p.load_state(OUT/(arm+'.pt'))['state']);hook.attach()
            for item in evaluation:emit(arm,item,hook)
            hook.detach()
        assert counts['updates']==320 and counts['backwards']==960+6*sum(v['guard']['applied'] for v in read(OUT/'PROTECTED_TRAIN.json')['curve']) and counts['operator_backwards']==320 and counts['generations']==27
        write('COMPLETE.json',dict(status='COMPLETE',counts=counts,seconds=time.time()-START,training_owners=[5],automatic_retry=False,check_used_for_selection=False))
        write('RETENTION.json',dict(final_states=['CONTROL.pt','PROTECTED.pt'],
              reason='Two small terminal states retained for semantic audit and reproducibility; no model copy or trajectory archive',
              active_states_pending_analysis=['active_CONTROL.pt','active_PROTECTED.pt']))
    finally:
        hook.detach();handle.remove()

START=time.time();counts=dict(updates=0,backwards=0,operator_backwards=0,generations=0,forwards=0)
# Reuse pure computation helpers without invoking or charging a historical run.
r.c.budget=budget
code=0
try:main()
except BaseException:
    code=1;write('FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
finally:write('EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
