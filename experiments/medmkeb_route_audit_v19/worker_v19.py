"""Frozen two-expert R0: one key per query, generate only absent selected outputs."""
import os,sys,time,json,traceback,subprocess,shutil
from pathlib import Path
from types import SimpleNamespace
OUT=Path(os.environ['OUTPUT_RUN']);ROOT=OUT.parent;V18=Path(os.environ['V18_RUN']);P=Path(os.environ['PARENT_RUN']);began=time.time()
counts=dict(forwards=0,vision_forwards=0,generations=0,backwards=0,updates=0,route_keys=0)
last_assets=[0.]
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def budget():
    if time.time()-last_assets[0]>60:
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<128*1024**2
        last_assets[0]=time.time()
    assert time.time()<float(os.environ['DEADLINE_EPOCH']) and not (ROOT/'STOP').exists()
    assert counts['forwards']<=20000 and counts['generations']<=15 and counts['route_keys']<=14 and counts['vision_forwards']<=50 and counts['backwards']==counts['updates']==0
    assert shutil.disk_usage(ROOT).free>8*1024**3

def main():
    with (OUT/'ATTEMPT').open('x'):pass
    sys.path[:0]=[str(OUT/'tools'),str(V18/os.environ['GPU']/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
    import torch
    import module_run as r
    import route_math_v19 as math
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    assert read(ROOT/'CPU_ADMISSION.json')['status']=='PASS'
    tasks=read(V18/'DATA_ADMISSION.json')['tasks'];base=read(V18/'BASE_REUSE.json');rows=read(V18/'EVAL_ROWS.json')+[t[k] for t in tasks for k in ('source','cross','rephrase')]
    assert len(rows)==28 and len({x['query_id'] for x in rows})==28
    for i,t in enumerate(tasks):
        assert read(V18/str(i)/'FORMAL_EXIT.json')['code']==0
        assert read(V18/str(i)/t['key']/'MODEL_CONSUMERS_COMPLETE.json')['status']=='PASS'
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.matmul.allow_tf32=False
    r.c.budget=budget;runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime);assert not any(p.requires_grad for p in runtime.model.parameters())
    def count(*_):budget();assert counts['forwards']<20000;counts['forwards']+=1
    handle=runtime.model.register_forward_pre_hook(count);vision=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    write(OUT/'MODEL_READY.json',dict(time=time.time(),gpu=os.environ['GPU'],commands=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True)))
    try:
        entries=[]
        for i,t in enumerate(tasks):
            state=torch.load(V18/str(i)/t['key']/'BALANCEDIT.pt',map_location='cpu',weights_only=True);es=state['router']['entries'];assert len(es)==1 and es[0]['logical_edit_id']==t['key'];entries+=es
        router=MemoryRouter.from_state(dict(distance='euclidean',entries=entries),device=runtime.device)
        template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(P/'private/EVAL_BINDINGS.json');assigned=rows[int(os.environ['GPU'])::2];assert len(assigned)==14
        runtime.model.half();items=r.s.prepare(runtime,bindings,template,assigned,False);r.old.fp32(runtime)
        forced={}
        for i,t in enumerate(tasks):
            for q in read(V18/'EVAL_ROWS.json')+[t[k] for k in ('source','cross','rephrase')]:forced[t['key'],q['query_id']]=str(V18/str(i)/t['key']/'outputs'/'ARM_PLACEHOLDER'/(q['query_id']+'.json'))
        records=[];saved_keys={};owner={t[k]['query_id']:t['key'] for t in tasks for k in ('source','cross','rephrase')}
        for row,_,raw,expanded,binding in items:
            budget();qid=row['query_id'];prior=read(base[qid]);assert math.same_input(binding,prior['binding']);assert raw['preprocessing_seed']==prior['binding']['preprocessing_seed']
            batch=runtime.build_question_batch(r.s.record(template,row));assert batch.image_sha256==binding['image_sha256'] and batch.raw_input_ids[0].tolist()==binding['prompt_ids'] and batch.preprocessing_seed==raw['preprocessing_seed']
            with torch.inference_mode():key=runtime.extract_layer_input_key(batch,module_path=runtime.target_lock['balancedit']['targets'][0],pooling='mean')
            counts['route_keys']+=1;saved_keys[qid]=key.detach().cpu();decision=router.route(key);route=decision_as_json(decision);selected=decision.logical_edit_id if decision.activated else None
            if qid.endswith('_SOURCE'):
                entry=next(x for x in entries if x['logical_edit_id']==owner[qid]);difference=float((key.detach().cpu()-torch.as_tensor(entry['key'])).abs().max());assert difference<=1e-6,'Saved native router key mismatch';write(OUT/'NATIVE_KEY_ADMISSION.json',dict(status='PASS',max_difference=difference,reused_saved_anchor=True))
            write(OUT/'routes'/(qid+'.json'),dict(route=route,preprocessing_seed=raw['preprocessing_seed'],binding=binding,counts=counts))
            for arm in math.ARMS:
                arm_forced={k:v.replace('/ARM_PLACEHOLDER/','/'+arm+'/') for k,v in forced.items()};choice=math.resolve(selected,qid,base,arm_forced)
                if choice['kind']!='MISSING':
                    output=Path(choice['path']);answer=read(output);assert math.same_input(binding,answer['binding'])
                else:
                    output=OUT/'outputs'/arm/(qid+'.json');assert not output.exists();write(OUT/'attempts'/arm/(qid+'.json'),dict(attempt=1,time=time.time(),selected=selected));counts['generations']+=1;budget()
                    i=next(i for i,t in enumerate(tasks) if t['key']==selected);state=torch.load(V18/str(i)/selected/arm/'final.pt',map_location=runtime.device,weights_only=True)['state'];expert=r.p.expert(state,template['seed'],runtime.device).requires_grad_(False);expert.load_state_dict(state);hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
                    try:answer,trace=r.s.fit.short.generate(runtime,hook,raw,expanded)
                    finally:hook.detach()
                    tokens=list(answer.raw_token_ids);write(output,dict(binding=dict(binding,preprocessing_seed=raw['preprocessing_seed']),raw_answer=answer.decoded_text,raw_token_ids=tokens,generation_trace=trace,EOS=bool(tokens) and tokens[-1]==runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,selected_expert=selected));del expert
                records.append(dict(arm=arm,query_id=qid,route_source='FROZEN_R0_TWO_EXPERT',route=route,expected_expert=owner.get(qid),expected_action='OWN_EXPERT' if qid in owner else 'OFF',output_path=str(output),reuse=choice['kind']!='MISSING',reuse_kind=choice['kind']))
            write(OUT/'PROGRESS.json',dict(completed_queries=counts['route_keys'],counts=counts,time=time.time()))
        budget();assert math.validate(records,[(arm,x['query_id']) for arm in math.ARMS for x in assigned])==70
        torch.save(saved_keys,OUT/'QUERY_KEYS.pt');write(OUT/'CONSUMERS.json',records);write(OUT/'COMPLETE.json',dict(status='COMPLETE',queries=14,consumers=70,counts=counts,seconds=time.time()-began))
    finally:handle.remove();vision.remove()
if __name__=='__main__':
    code=0
    try:main()
    except BaseException:code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
    finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-began))
