"""Fixed64 greedy answers from the two frozen V11 finals, zero training."""
import os,json,time,shutil
from pathlib import Path
import torch
from check import check,selfcheck

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN']);source=Path(os.environ['V11_ROOT'])/str(partition);read=lambda p:json.loads(Path(p).read_text())
    def write(p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
    def budget():
        assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (shared/'STOP').exists()
        assert counts['forwards']<=33000 and counts['generations']<=32 and counts['updates']==counts['backwards']==0
        assert shutil.disk_usage(shared).free>8*1024**3
    r.c.budget=budget;write(out/'CPU_ADMISSION.json',selfcheck());completed=read(source/'STUDY_COMPLETE.json');assert completed['trajectory_qualified'] and read(source/'EXIT.json')['code']==0
    config=read(shared/'DATA_ADMISSION.json');task=config['tasks'][partition];rows=read(shared/'EVAL_ROWS.json')+[task['rephrase']];assert len(rows)==len({x['query_id'] for x in rows})==32
    template=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(parent/'private/EVAL_BINDINGS.json');runtime.model.half();evaluation=r.s.prepare(runtime,bindings,template,rows,False);r.old.fp32(runtime)
    for row,_,raw,expanded,binding in evaluation:
        previous=read(read(shared/'BASE_REUSE.json')[row['query_id']])['binding'];assert all(binding[k]==previous[k] for k in ('image_sha256','question','reference','prompt_ids','generation'))
        write(out/'bindings'/(row['query_id']+'.json'),dict(binding,preprocessing_seed=raw['preprocessing_seed']))
    state=torch.load(source/'final.pt',map_location='cpu',weights_only=True)['state'];expert=r.p.expert(state,template['seed'],runtime.device);expert.load_state_dict(state);assert all(torch.equal(v.cpu(),state[k]) for k,v in expert.state_dict().items())
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);write(out/'ENGINEERING_ADMISSION.json',dict(status='PASS',input_bindings=32,source_final_state_exact=True,greedy_original_cap=1024,training_updates=0,new_Base_answers=0));hook.attach()
    try:
        for index,(row,_,raw,expanded,binding) in enumerate(evaluation):
            budget();dest=out/'outputs'/(row['query_id']+'.json');attempt=out/'attempts'/dest.name;assert not dest.exists() and not attempt.exists();write(attempt,dict(attempt=1,time=time.time()));counts['generations']+=1
            answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded);write(dest,dict(binding=binding,raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids),generation_trace=trace,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(answer.raw_token_ids)>=1024));write(out/'PROGRESS.json',dict(completed=index+1,expected=32,counts=counts));print(task['key'],'ANSWER',index+1,flush=True)
        coverage=check(out/'outputs',[x['query_id'] for x in rows]);write(out/'STUDY_COMPLETE.json',dict(status='COMPLETE',task=task['key'],answers=coverage,source_final_state_exact=True,counts=counts))
    finally:hook.detach()
