"""Fixed64 greedy answers from the two frozen V11 finals, zero training."""
import os,json,time,shutil
from pathlib import Path
import torch
from answer_check import check,selfcheck
from prefix_cache import PrefixCache
from source_path_v13 import proposal,construct,same
from qp import project

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);source=Path(os.environ['V11_ROOT'])/str(partition);read=lambda p:json.loads(Path(p).read_text())
    def write(p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
    def budget():
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (shared/'STOP').exists()
        assert counts['forwards']<=33100 and counts['generations']<=32 and counts['updates']<=31 and counts['backwards']<=30
        assert shutil.disk_usage(shared).free>8*1024**3
    def save(path,value):
        path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    r.c.budget=budget;write(out/'CPU_ADMISSION.json',selfcheck());assert read(source/'STUDY_COMPLETE.json')['trajectory_qualified'] and read(source/'EXIT.json')['code']==0
    config=read(shared/'DATA_ADMISSION.json');task=config['tasks'][partition];rows=read(shared/'EVAL_ROWS.json')+[task['rephrase']];assert len(rows)==len({x['query_id'] for x in rows})==32
    template=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(parent/'private/EVAL_BINDINGS.json');runtime.model.half();training=r.s.prepare(runtime,bindings,template,[task['source']],True);r.old.fp32(runtime)
    native=task['source']['query_id'];_,value,raw,expanded,binding=training[0];previous=Path(os.environ['V7_ROOT'])/str(partition);oldbinding=read(previous/'bindings'/(native+'.json'));actual=dict(binding,preprocessing_seed=raw['preprocessing_seed']);assert all(actual[k]==oldbinding[k] for k in oldbinding if k!='image_path')
    batch=r.old.batch_on(runtime,value);initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0];expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base);hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
    inputs=torch.load(previous/'QP_INPUTS.pt',map_location='cpu',weights_only=True);keys=list(base);params=dict(expert.named_parameters());reference=read(previous/'candidates/SOURCE_1.0.json');initial_loss=reference['private_metrics']['source_loss']+reference['summary']['source_gain'];scalar=dict(step=0,second_moments=[0.,0.]);curve=[];counts['reused_gradient_sequences']=1
    hook.attach()
    try:
        with PrefixCache(runtime,[batch]):
            for step in range(1,32):
                budget();before=r.old.snapshot(expert)
                if step==1:gradient=inputs['source_gradient'];loss_before=initial_loss
                else:
                    hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch);gs=torch.autograd.grad(loss,tuple(params[k] for k in keys),allow_unused=True);counts['backwards']+=1;loss_before=float(loss.detach());gradient=torch.cat([(v if v is not None else torch.zeros_like(params[k])).detach().cpu().double().reshape(-1) for k,v in zip(keys,gs)])
                raw,scalar,direction=proposal(r,before,gradient,scalar,keys)
                if step==1:assert torch.equal(direction,inputs['source_direction'])
                zero,_=project(direction,torch.empty((0,direction.numel())),torch.empty(0));candidate=construct(r,before,raw,direction,direction,1.,keys);assert same(candidate,construct(r,before,raw,direction,zero,1.,keys));norm=r.fm.map_norm(before,candidate);assert norm>0
                expert.load_state_dict(candidate);counts['updates']+=1;counts['candidate_checks']+=1;curve.append(dict(step=step,source_loss_before=loss_before,actual_operator_norm=norm));save(out/'active.pt',dict(state={k:v.cpu() for k,v in candidate.items()},scalar=scalar,curve=curve,counts=dict(counts)));write(out/'PROGRESS.json',dict(phase='TRAIN',step=step,source_loss_before=loss_before,counts=counts))
            capture={}
            def tap(_m,_a,output):
                mask=batch.labels[:,1:]!=-100;target=batch.labels[:,1:][mask];logits=output.logits[:,:-1][mask].float();selected=logits.gather(1,target[:,None])[:,0];capture.update(target_ids=target.tolist(),NLL=(-logits.log_softmax(-1).gather(1,target[:,None])[:,0]).tolist(),probability=(selected-logits.logsumexp(-1)).exp().tolist(),rank=((logits>selected[:,None]).sum(-1)+1).tolist())
            handle=runtime.model.register_forward_hook(tap)
            try:
                with torch.no_grad():hook.set_teacher_routing(batch.labels);final_loss=float(runtime.compute_loss(batch))
            finally:handle.remove()
            write(out/'SOURCE_TOKEN_DIAGNOSTIC.json',capture)
    finally:hook.detach()
    state=r.old.snapshot(expert);save(out/'final.pt',dict(state={k:v.cpu() for k,v in state.items()}));saved=torch.load(out/'final.pt',map_location='cpu',weights_only=True)['state'];assert same(saved,{k:v.cpu() for k,v in state.items()})
    for i,x in enumerate(curve):x['source_loss_after']=curve[i+1]['source_loss_before'] if i+1<len(curve) else final_loss
    write(out/'TRAIN_COMPLETE.json',dict(status='COMPLETE',updates=31,curve=curve,source_gain=initial_loss-final_loss,final_source_loss=final_loss,save_load_exact=True,zero_constraint_path_exact=True,initial_gradient_reused=True,counts=counts));write(shared/('FROZEN_'+str(partition)+'.json'),dict(task=task['key'],updates=31))
    while not all((shared/('FROZEN_'+str(i)+'.json')).exists() for i in (0,1)):
        budget();assert not any((shared/str(i)/'FAILURE.json').exists() for i in (0,1));time.sleep(2)
    runtime.model.half();evaluation=r.s.prepare(runtime,bindings,template,rows,False);r.old.fp32(runtime)
    for row,_,raw,expanded,binding in evaluation:
        old=read(read(shared/'BASE_REUSE.json')[row['query_id']])['binding'];assert all(binding[k]==old[k] for k in ('image_sha256','question','reference','prompt_ids','generation'));write(out/'bindings'/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
    assert all(torch.equal(v.cpu(),saved[k]) for k,v in expert.state_dict().items());write(out/'ENGINEERING_ADMISSION.json',dict(status='PASS',input_bindings=32,source_final_state_exact=True,zero_constraint_path_exact=True,updates=31,greedy_original_cap=1024,new_Base_answers=0));hook.attach()
    try:
        for index,(row,_,raw,expanded,binding) in enumerate(evaluation):
            budget();dest=out/'outputs'/(row['query_id']+'.json');attempt=out/'attempts'/dest.name;assert not dest.exists() and not attempt.exists();write(attempt,dict(attempt=1,time=time.time()));counts['generations']+=1
            answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded);write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),generation_trace=trace,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024));write(out/'PROGRESS.json',dict(completed=index+1,expected=32,counts=counts));print(task['key'],'ANSWER',index+1,flush=True)
        coverage=check(out/'outputs',[x['query_id'] for x in rows]);write(out/'STUDY_COMPLETE.json',dict(status='COMPLETE',task=task['key'],answers=coverage,source_final_state_exact=True,counts=counts))
    finally:hook.detach()
