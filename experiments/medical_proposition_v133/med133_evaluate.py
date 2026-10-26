"""Actual unchanged-Base R0 selection, with state-bound deterministic answer reuse."""
import hashlib
import json
from pathlib import Path
import time
import torch


def state_id(state):
    value=hashlib.sha256()
    for key in sorted(state):
        value.update(key.encode());value.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def membership(entries,row):
    assert len(entries)==146 and len({e['logical_edit_id'] for e in entries})==146
    if row['mode']=='single':return [entries[row['owner']-1]]
    assert row['mode'] in ('bank','insertion') and 1<=row['prefix']<=146
    return entries[:row['prefix']]


def run(runtime,r,inputs,counts,budget,write,root,parent,lock):
    import contract
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    tasks=inputs['tasks'];byid={t['edit_id']:t for t in tasks}
    entries=[r.p.load_state(Path(inputs['router_bindings'][str(t['order'])]))['entries'][0] for t in tasks]
    states={t['edit_id']:r.p.load_state(root/'candidates'/f"MED_{t['order']}.pt") for t in tasks}
    hashes={key:state_id(v) for key,v in states.items()}
    assert len(states)==146
    package=root/'retained/MED146.pt'
    if not package.exists():
        package.parent.mkdir(exist_ok=True)
        torch.save(dict(states=states,R0_entries=entries,tasks=tasks,source_rows=inputs['source_rows'],
            mapping=inputs['mapping'],unmapped=inputs['unmapped'],state_hashes=hashes,protocol=lock,
            calibration={str(t['order']):r.c.read(root/'calibration'/f"{t['order']}.json") for t in tasks}),package.with_suffix('.tmp'))
        package.with_suffix('.tmp').replace(package)
    check=torch.load(package,map_location='cpu',weights_only=True)
    assert check['state_hashes']==hashes and len(check['states'])==len(check['R0_entries'])==146
    write('EXPERT_LIBRARY_FROZEN.json',dict(experts=146,states=hashes,input_binding=lock['input_binding'],
        canonical=str(package),bytes=package.stat().st_size,save_load_exact=True,time=time.time()))
    # Canonical finals, original R0 entries/config and hashes remain after scoring.
    import numpy as np
    rngpath=root/'GENERATION_RNG.pt'
    if rngpath.exists():generation_rng=torch.load(rngpath,map_location='cpu',weights_only=False)
    else:
        generation_rng=dict(rng=r.c.rng(),numpy_rng=np.random.get_state());torch.save(generation_rng,rngpath)
    h=hashlib.sha256()
    for key in sorted(generation_rng['rng']):
        value=generation_rng['rng'][key]
        h.update(key.encode());h.update(value.cpu().numpy().tobytes() if isinstance(value,torch.Tensor) else repr(value).encode())
    nr=generation_rng['numpy_rng'];h.update(nr[0].encode());h.update(nr[1].tobytes());h.update(repr(nr[2:]).encode());rng_binding=h.hexdigest()
    routers,features,answers={},{},{}
    for path in (root/'outputs').glob('*.json'):
        v=r.c.read(path);selected=v['effective_expert']
        assert v['effective_state_hash']==(hashes[selected] if selected else 'UNEDITED_BASE')
        answers[v['prediction_cache_key']]=dict(v['R0'],generation_trace=v['generation_trace'])
    bindings=r.c.read(parent/'private/EVAL_BINDINGS.json')
    route_layer=runtime.target_lock['balancedit']['targets'][0]
    assert not runtime.generation_config.get('do_sample',False)
    active_sets={n:{(t['native']['image_sha256'],t['native']['question']) for t in tasks[:n]} for n in range(1,147)}
    for stage in ('A','B'):
        if stage=='A' and (root/'STAGE_A_GENERATION_COMPLETE.json').exists():continue
        if stage=='B':
            while not (root/'STAGE_A_REPORT_COMPLETE.json').exists():
                budget();time.sleep(5)
            report=r.c.read(root/'STAGE_A_REPORT_COMPLETE.json')
            assert report['engineering_complete'] and report['single_consumers']==2956
            # There is intentionally no scientific-performance gate here.
            write('STAGE_B_STARTED.json',dict(time=time.time(),reused_experts=146,additional_training=0))
        rows=[row for row in inputs['plan'] if (row['mode']=='single')==(stage=='A')]
        for begin in range(0,len(rows),32):
            budget();runtime.model.half();prepared=[]
            try:
                for row in rows[begin:begin+32]:
                    task=tasks[row['owner']-1] if row['owner'] else tasks[0]
                    query=row['query'];qkey=r.c.digest(query)
                    if qkey not in features:
                        batch=runtime.build_question_batch(r.s.record(task,query))
                        with torch.inference_mode():
                            features[qkey]=runtime.extract_layer_input_key(batch,module_path=route_layer,pooling='mean').detach().cpu()
                    key=(row['mode'],row['prefix'],row['owner'] if row['mode']=='single' else 0)
                    if key not in routers:
                        routers[key]=MemoryRouter.from_state(dict(distance='euclidean',entries=membership(entries,row)),device=runtime.device)
                    decision=routers[key].route(features[qkey].to(runtime.device))
                    selected=decision.logical_edit_id if decision.activated else None
                    allowed=membership(entries,row)
                    assert selected is None or selected in {e['logical_edit_id'] for e in allowed}
                    item=r.s.prepare(runtime,bindings,task,[query])[0]
                    prepared.append((row,query,item,decision_as_json(decision),selected))
            finally:r.old.fp32(runtime)
            for row,query,item,route,selected in prepared:
                budget();cid=contract.consumer_id(row);dest=root/'outputs'/(r.c.digest(cid)+'.json')
                prediction_binding=dict(input=item[4],effective_state=hashes[selected] if selected else 'UNEDITED_BASE',
                    model_source=lock['source_reference'],deterministic_greedy=True,generation_rng_binding=rng_binding)
                cachekey=r.c.digest(prediction_binding)
                if dest.exists():
                    old=r.c.read(dest)
                    assert old['consumer_id']==cid and old['route']==route and old['prediction_binding']==prediction_binding
                    continue
                reused=cachekey in answers
                if reused:answer=answers[cachekey]
                else:
                    hook=None
                    try:
                        if selected is not None:
                            expert=r.p.expert(states[selected],byid[selected]['seed'],runtime.device).requires_grad_(False)
                            hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert);hook.attach()
                        r.c.restore_rng(generation_rng['rng']);np.random.set_state(generation_rng['numpy_rng'])
                        counts['generations']+=1
                        value,trace=r.s.fit.short.generate(runtime,hook,item[2],item[3])
                        observed=r.c.rng()
                        assert observed['python_rng']==generation_rng['rng']['python_rng']
                        for key in ('torch_rng','cuda_rng'):
                            assert torch.equal(observed[key],generation_rng['rng'][key])
                        answer=dict(raw_answer=value.decoded_text,raw_token_ids=list(value.raw_token_ids),generation_trace=trace)
                        assert answer['raw_token_ids']
                        answers[cachekey]=answer
                    finally:
                        if hook is not None:hook.detach()
                value=dict(method='MED',mode=row['mode'],prefix=row['prefix'],owner=row['owner'],consumer_id=cid,
                    binding=dict(input=query,judge_input=item[4]),R0={k:answer[k] for k in ('raw_answer','raw_token_ids')},
                    route=route,effective_expert=selected,effective_state_hash=prediction_binding['effective_state'],
                    prediction_binding=prediction_binding,prediction_cache_key=cachekey,prediction_reused=reused,
                    new_GPU_generation=not reused,original_router=True,actual_Base_R0_evaluated=True,
                    active_target=row['mode']!='single' and (query.get('image_sha256'),query['question']) in active_sets[row['prefix']],
                    generation_trace=answer['generation_trace'],EOS=answer['raw_token_ids'][-1]==runtime.adapter.tokenizer.eos_token_id,
                    at_cap=len(answer['raw_token_ids'])>=1024)
                write(str(dest.relative_to(root)),value)
            write('PROGRESS.json',dict(phase='SINGLE' if stage=='A' else 'SEQUENTIAL',consumers=min(begin+32,len(rows)),
                total=len(rows),counts=counts,time=time.time(),unique_Base_features=len(features),unique_predictions=len(answers)))
            print('EVAL',stage,min(begin+32,len(rows)),len(rows),flush=True)
        write('STAGE_'+stage+'_GENERATION_COMPLETE.json',dict(time=time.time(),consumers=len(rows),counts=counts,
            all_candidates_before_evaluation=True,actual_R0=True,cuda_max_memory_allocated=torch.cuda.max_memory_allocated()))
    assert len(list((root/'outputs').glob('*.json')))==7494
    write('GENERATION_COMPLETE.json',dict(time=time.time(),consumers=7494,counts=counts,
        single=2956,sequential=4538,all_candidates_retained=True,functional_R0_with_strict_reuse=True))
