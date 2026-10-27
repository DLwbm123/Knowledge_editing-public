"""Freeze all answer-blind text-dominance routes, then reuse exact past outputs."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT']);PARENT=Path(os.environ['PARENT_RUN']);MED=Path(os.environ['MED150_ROOT']);OLD=Path(os.environ['RAW_MED_ROOT']);PREV=Path(os.environ['PREVIOUS_MODAL_ROOT'])
CLOCK=json.loads((ROOT/'CLOCK.json').read_text());LOCK=json.loads((ROOT/'EVAL_LOCK.json').read_text())
COUNTS=dict(forwards=0,generations=0,backwards=0,updates=0,vision_forwards=0)


def write(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def budget():
    assert time.time()<CLOCK['hard_end']-30 and not (ROOT/'EVAL_STOP.json').exists()
    assert time.time()-json.loads((ROOT/'EVAL_GUARD_HEALTH.json').read_text())['time']<45
    for k,v in LOCK['budgets'].items():
        if k in COUNTS:assert COUNTS[k]<=v,(k,COUNTS[k],v)


def main():
    with (ROOT/'EVAL_WORKER_ONCE').open('x'):pass
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original experiment clock')))
    signal.alarm(max(1,int(CLOCK['hard_end']-time.time()-30)))
    sys.path[:0]=[str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),str(PARENT/'private/official_llava')]
    import numpy as np
    import torch
    import module_run as r
    import modal_select as choose
    import text_select
    import structure
    from llava.constants import IMAGE_TOKEN_INDEX
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json,euclidean_distances
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget=budget;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(LOCK['physical_GPU'])
    torch.cuda.set_per_process_memory_fraction(.78)
    pkg=torch.load(MED/'retained/MED150.pt',map_location='cpu',weights_only=True)
    tasks=pkg['tasks'];entries=pkg['R0_entries'];states=pkg['states'];hashes=pkg['state_hashes'];byid={t['edit_id']:t for t in tasks}
    assert len(tasks)==len(entries)==len(states)==len(byid)==150
    import med133_evaluate
    assert all(med133_evaluate.state_id(v)==hashes[k] for k,v in states.items())
    plan=json.loads((MED/'EVAL_PLAN_FROZEN.json').read_text());bank=[x for x in plan['plan'] if x['mode']=='bank' and x['prefix']==150]
    assert len(bank)==2148 and len({x['query']['query_id'] for x in bank})==2148 and r.c.digest(plan)==LOCK['source_plan_binding']
    assert json.loads((MED/'EVAL_GENERATION_COMPLETE.json').read_text())['consumers']==2207
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'));runtime.model.eval();r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings=r.c.read(PARENT/'private/EVAL_BINDINGS.json');assert runtime.generation_config==next(iter(bindings.values()))['generation']
    router=MemoryRouter.from_state(dict(distance='euclidean',entries=entries),device=runtime.device)
    assert router.logical_ids==[t['edit_id'] for t in tasks]
    layer=runtime.target_lock['balancedit']['targets'][0];rng=torch.load(OLD/'GENERATION_RNG.pt',map_location='cpu',weights_only=False)
    fh=runtime.model.register_forward_pre_hook(lambda *_a,**_k:(budget(),COUNTS.__setitem__('forwards',COUNTS['forwards']+1)) and None,with_kwargs=True)
    vh=runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_:COUNTS.__setitem__('vision_forwards',COUNTS['vision_forwards']+1))
    def feature(task,row):
        budget();raw=runtime.adapter.prepare_inputs(Path(r.c.local_path(row['image_path'])),row['question'],None)
        assert raw['image_sha256']==row['image_sha256']
        with torch.inference_mode():
            emb,att,pos,_=runtime._expand_multimodal(raw_input_ids=raw['input_ids'],attention_mask=raw['attention_mask'],labels=torch.full_like(raw['input_ids'],-100),images=raw['images'])
            masks=structure.token_masks(raw['input_ids'][0],emb.shape[1],None if att is None else att[0],IMAGE_TOKEN_INDEX)
            captured=[];hook=runtime.get_module(layer).register_forward_pre_hook(lambda _,args:captured.append(structure.pool(args[0][0],masks)))
            try:runtime.model(inputs_embeds=emb,attention_mask=att,position_ids=pos,labels=None,use_cache=False,return_dict=True)
            finally:hook.remove()
        assert len(captured)==1 and captured[0][0].shape==(3,4096)
        return captured[0][0].detach(),raw
    runtime.model.half()
    # One actual GPU preflight, before the full source or query pass.
    first=bank[0]['query'];vectors,raw=feature(tasks[0],first)
    cid=json.dumps(['MED','bank',150,0,first['query_id']],separators=(',',':'))
    v=json.loads((MED/'eval_outputs'/(r.c.digest(cid)+'.json')).read_text());oldroute=v['route'];oldinput=v['binding']['judge_input'];del v
    assert raw['input_ids'][0].tolist()==oldinput['prompt_ids'] and raw['attention_mask'][0].tolist()==oldinput['attention_mask']
    assert raw['image_sha256']==oldinput['image_sha256'] and runtime.generation_config==oldinput['generation']
    decision=decision_as_json(router.route(vectors[0]));assert decision['activated']==oldroute['activated'] and decision['logical_edit_id']==oldroute['logical_edit_id']
    ref=runtime.extract_layer_input_key(runtime.build_question_batch(r.s.record(tasks[0],first)),module_path=layer,pooling='mean')
    torch.testing.assert_close(vectors[0],ref,rtol=1e-4,atol=1e-5)
    write('GPU_ADMISSION.json',dict(status='PASS',time=time.time(),all_modal_mean_matches_R0=True,first_R0_and_prompt_match=True))
    sources=[]
    for t in tasks:
        values=pkg['source_rows'][str(t['order'])];assert len(values)==5
        sources.append(torch.stack([feature(t,row)[0] for row in values]))
        if t['order']%10==0:write('EVAL_PROGRESS.json',dict(time=time.time(),phase='SOURCE_KEYS',done=t['order'],total=150,counts=COUNTS))
    keys=torch.stack(sources);assert keys.shape==(150,5,3,4096)
    visual=torch.nn.functional.normalize(keys[:,:,1].float(),dim=-1,eps=1e-12);textkeys=torch.nn.functional.normalize(keys[:,:,2].float(),dim=-1,eps=1e-12)
    for i,row in enumerate(bank):
        budget();query=row['query'];qid=query['query_id'];vectors,raw=feature(tasks[0],query)
        cid=json.dumps(['MED','bank',150,0,qid],separators=(',',':'));oldpath=MED/'eval_outputs'/(r.c.digest(cid)+'.json')
        prior=json.loads(oldpath.read_text());oldroute=prior['route'];oldinput=prior['binding']['judge_input'];del prior
        assert raw['input_ids'][0].tolist()==oldinput['prompt_ids'] and raw['attention_mask'][0].tolist()==oldinput['attention_mask']
        assert raw['image_sha256']==oldinput['image_sha256'] and runtime.generation_config==oldinput['generation']
        decision=decision_as_json(router.route(vectors[0]));assert decision['activated']==oldroute['activated'] and decision['logical_edit_id']==oldroute['logical_edit_id']
        assert decision['nearest_logical_edit_id']==oldroute['nearest_logical_edit_id']
        assert abs(decision['nearest_distance']-oldroute['nearest_distance'])<=1e-5*max(1.,abs(oldroute['nearest_distance']))
        scores=((visual*torch.nn.functional.normalize(vectors[1].float(),dim=-1,eps=1e-12)).sum(-1)*.5+(textkeys*torch.nn.functional.normalize(vectors[2].float(),dim=-1,eps=1e-12)).sum(-1)*.5).max(dim=1).values
        distances=euclidean_distances(router._key_matrix(),vectors[0]);radii=torch.as_tensor(router.radii,device=distances.device,dtype=distances.dtype)
        nearest=router.logical_ids.index(decision['nearest_logical_edit_id']);chosen=choose.select(decision['activated'],nearest,distances,radii,scores)
        previous=router.logical_ids[chosen] if chosen is not None else None
        oldmodal=json.loads((PREV/'routes'/(r.c.digest(qid)+'.json')).read_text())
        assert oldmodal['selected']==previous and oldmodal['original_route']['logical_edit_id']==decision['logical_edit_id']
        text_scores=(textkeys*torch.nn.functional.normalize(vectors[2].float(),dim=-1,eps=1e-12)).sum(-1).max(dim=1).values
        selected=text_select.select(decision['logical_edit_id'],previous,float(text_scores[nearest]),float(text_scores[chosen]) if chosen is not None else 0.)
        dest=ROOT/'routes'/(r.c.digest(qid)+'.json');assert not dest.exists()
        write(str(dest.relative_to(ROOT)),dict(query_id=qid,original_route=decision,selected=selected,changed=selected!=decision['logical_edit_id'],
            candidate_count=int((distances<=radii).sum()),margin=0.02,previous_modal_selected=previous,text_rule_fallback=selected!=previous,answer_used=False,score_used=False))
        if (i+1)%32==0:write('EVAL_PROGRESS.json',dict(time=time.time(),phase='ROUTE_FREEZE',done=i+1,total=len(bank),counts=COUNTS))
    assert len(list((ROOT/'routes').glob('*.json')))==2148
    write('ROUTES_FROZEN.json',dict(time=time.time(),queries=2148,selection_used_answer=False,selection_used_judge=False,margin=.02))
    r.old.fp32(runtime);original=0;modal=0;fallback=0
    for i,row in enumerate(bank):
        budget();q=row['query'];qid=q['query_id'];route=json.loads((ROOT/'routes'/(r.c.digest(qid)+'.json')).read_text());selected=route['selected']
        if selected==route['original_route']['logical_edit_id']:
            arm='MED';source=MED;original+=1
        else:
            assert selected==route['previous_modal_selected'];arm='MED_MODAL';source=PREV;modal+=1
        fallback+=route['text_rule_fallback']
        oldcid=json.dumps([arm,'bank',150,0,qid],separators=(',',':'));old=json.loads((source/'eval_outputs'/(r.c.digest(oldcid)+'.json')).read_text())
        assert old['effective_expert']==selected and old['effective_state_hash']==(hashes[selected] if selected else 'UNEDITED_BASE')
        basecid=json.dumps(['MED','bank',150,0,qid],separators=(',',':'));baseline=json.loads((MED/'eval_outputs'/(r.c.digest(basecid)+'.json')).read_text())
        assert old['binding']['judge_input']==baseline['binding']['judge_input']
        assert old['binding']['input']==q and old['binding']['judge_input']['question']==q['question'] and old['binding']['judge_input']['reference']==q['reference'] and old['binding']['judge_input']['image_sha256']==q['image_sha256']
        assert old['R0']['raw_token_ids']
        cid=json.dumps(['MED_TEXT','bank',150,0,qid],separators=(',',':'));dest=ROOT/'eval_outputs'/(r.c.digest(cid)+'.json');assert not dest.exists()
        write(str(dest.relative_to(ROOT)),dict(method='MED_TEXT',mode='bank',prefix=150,owner=0,consumer_id=cid,binding=old['binding'],R0=old['R0'],route=route,
            effective_expert=selected,effective_state_hash=old['effective_state_hash'],reuse_old=True,reused_arm=arm,
            generation_trace=old['generation_trace'],EOS=old['EOS'],at_cap=old['at_cap']))
        if (i+1)%64==0:write('EVAL_PROGRESS.json',dict(time=time.time(),phase='EXACT_OUTPUT_REUSE',done=i+1,total=len(bank),counts=COUNTS,original_reused=original,modal_reused=modal,text_fallback=fallback))
    fh.remove();vh.remove();assert len(list((ROOT/'eval_outputs').glob('*.json')))==2148 and original+modal==2148 and COUNTS['generations']==0
    write('EVAL_GENERATION_COMPLETE.json',dict(time=time.time(),consumers=2148,original_reused=original,modal_reused=modal,text_fallback=fallback,counts=COUNTS,all_routes_frozen_first=True,cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('EVAL_FAILURE.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS));raise
    finally:write('EVAL_WORKER_EXIT.json',dict(time=time.time(),code=code,counts=COUNTS))
