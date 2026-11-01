"""One guarded nine-owner matched-span run, then all fixed event predictions."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT'])
PARENT=Path(os.environ['PARENT_RUN'])
FIXED=Path(os.environ['FIXED_DEPTH_ROOT'])
MED=Path(os.environ['RAW_MED_ROOT'])
LOCK=Path(os.environ['SPAN_LOCK'])
DEADLINE=float(os.environ['EXPERIMENT_DEADLINE'])
COUNTS=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,operator_backwards=0)


def write(name, value, once=False):
    p=ROOT/name; p.parent.mkdir(parents=True,exist_ok=True)
    if once:
        with p.open('x') as f:json.dump(value,f,ensure_ascii=False)
        p.chmod(0o600)
    else:
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(p)


def budget():
    if time.time()>=DEADLINE-60 or (ROOT/'STOP.json').exists():raise TimeoutError('Deadline or guard stop')
    h=json.loads((ROOT/'GUARD_HEALTH.json').read_text())
    if time.time()-h['time']>45 or h['hard_end']!=DEADLINE:raise RuntimeError('Guard stale')
    if COUNTS['updates']>4320 or COUNTS['backwards']>8640 or COUNTS['generations']>1113:
        raise RuntimeError('Frozen scientific count exceeded')


def main():
    with (ROOT/'WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Deadline')))
    signal.alarm(max(1,int(DEADLINE-time.time()-60)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
    import torch
    import module_run as r
    import runtime_binding
    import span_train
    import fixed_generate
    import stream_extract as stream
    import prior_binding
    from first_eval import rng_digest
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    r.c.budget,r.COUNTS=budget,COUNTS
    admission=json.loads((FIXED/'ROLE_ADMISSION.json').read_text())
    source=json.loads((FIXED/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    tasks=json.loads((FIXED/'TASK_IDENTITIES_PRIVATE.json').read_text())
    lk=json.loads(LOCK.read_text())
    if lk['status']!='WEAK_TOKEN_PROXY_ONLY' or len(lk['records'])!=16:raise ValueError('Span lock incomplete')
    spans={x['order']:x for x in lk['records'] if x['status']=='WEAK_TOKEN_MATCH'}
    owners=admission['owners']
    if (admission['status']!='SOURCE_AND_EVALUATION_ROLE_ADMITTED' or len(spans)!=9 or len(owners)!=16
        or [x['task']['order'] for x in source]!=owners or [x['order'] for x in tasks]!=owners):
        raise ValueError('Fixed panel identity mismatch')
    def select_rows(path):
        result=[]; total=0
        with path.open('rb') as f:
            if stream.nonspace(f)!=b'[':raise ValueError('Expected input array')
            while True:
                token=stream.nonspace(f)
                if token==b']':break
                start=f.tell()-1
                ident=prior_binding.select(f,token,{'owner':True})
                end=f.tell();total+=1
                if ident.get('owner') in spans:
                    f.seek(start);entry=json.loads(f.read(end-start));f.seek(end);result.append(entry)
                token=stream.nonspace(f)
                if token==b']':break
                if token!=b',':raise ValueError('Invalid input array')
        if total!=508:raise ValueError('Original input coverage mismatch')
        return result
    rows=select_rows(FIXED/'EVAL_INPUTS_PRIVATE.json')
    prior_rows=select_rows(FIXED/'PRIOR_INPUT_BINDINGS_PRIVATE.json')
    prior={(x['owner'],x['query_id']):x for x in prior_rows}
    source=[x for x in source if x['task']['order'] in spans]
    if len(source)!=9 or len(rows)!=371 or len(prior)!=371:raise ValueError('Matched panel coverage mismatch')
    taskmap={x['order']:x for x in tasks}
    with (PARENT/'private/EVAL_BINDINGS.json').open('rb') as f:
        config=runtime_binding.extract(f)['configuration_only']
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'))
    runtime.model.eval();r.old.fp32(runtime)
    if runtime.generation_config!=config['generation']:raise ValueError('Generation config differs')
    rng=torch.load(MED/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    rng_id=rng_digest(rng)
    model_source=json.loads((MED/'PROTOCOL_LOCK.json').read_text())['source_reference']
    expected_runtime=dict(inherited_runtime=config['runtime'],actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',frozen_FP16_prefill=True)
    def count(_module,_args,_kwargs):budget();COUNTS['forwards']+=1
    h=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    try:
        complete=[]
        for entry in source:
            owner=entry['task']['order']
            complete.extend(span_train.train(runtime,r,entry['task'],entry['source_rows'],{'configuration_only':config},admission,spans[owner],ROOT/'weights',budget))
            write('TRAIN_PROGRESS.json',dict(completed_owners=len(complete)//3,counts=COUNTS))
        if len(complete)!=27 or COUNTS['updates']!=4320 or COUNTS['backwards']!=8640:
            raise ValueError('Training counts differ')
        write('TRAIN_FROZEN.json',dict(status='FROZEN',results=complete,counts=COUNTS))
        for row in rows:
            owner,q=row['owner'],row['query'];old=prior[(owner,q['query_id'])]
            if (old['input']['runtime']!=expected_runtime or old['prediction_binding']['model_source']!=model_source
                or old['prediction_binding']['generation_rng_binding']!=rng_id):raise ValueError('Original runtime binding differs')
            orig=old['route'];route=dict(activated=orig['activated'],logical_edit_id=orig['logical_edit_id'] if orig['activated'] else None)
            binding={k:old['input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation')}
            for arm in ('B','C','D'):
                key=hashlib.sha256(json.dumps([owner,q['query_id'],arm],separators=(',',':')).encode()).hexdigest()
                write('attempts/'+key+'.json',dict(time=time.time(),arm=arm),once=True)
                COUNTS['generations']+=1
                val=fixed_generate.generate_one(runtime,r,taskmap[owner],q,route,binding,route,ROOT/'weights'/f'{owner}_{arm}.pt','C21',rng,budget)
                val.update(owner=owner,arm=arm,original_consumer_id=old['consumer_id'],original_execution_binding=old['prediction_binding'],original_route=orig,prediction_reused_from=None)
                write('predictions/'+key+'.json',val,once=True)
            write('EVAL_PROGRESS.json',dict(predictions=COUNTS['generations'],expected=1113))
        if COUNTS['generations']!=1113:raise ValueError('Prediction coverage mismatch')
        write('EVAL_COMPLETE.json',dict(status='GENERATION_COMPLETE',events=371,predictions=1113,counts=COUNTS))
    finally:h.remove()


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('WORKER_FAILURE.json',dict(error=traceback.format_exc(),counts=COUNTS));raise
    finally:write('WORKER_EXIT.json',dict(code=code,counts=COUNTS))
