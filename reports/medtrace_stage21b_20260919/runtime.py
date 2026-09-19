"""Stage21B freshly initialized L31 control of a frozen 45-edit bank; no Base edits."""
import os,json,sys,time,gc,shutil
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage19_fasttrack_budget import write,check
from scripts.medtrace.stage19_fasttrack import load,record_for,query_id


def campaign(cfg):
    from scripts.medtrace.stage17_single import setup
    setup(cfg)
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from methods.medtrace.hsic import bound_hook,layer_path
    from m3bench_repro.editors.routing import MemoryRouter,balanced_radius
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import assert_base_off,state_hash,teachers_for,train
    from scripts.medtrace.run_selective_write import save
    root=Path(cfg['run']);old=Path(cfg['stage20_run']);stream=read(root/'private/STREAM.json')
    assert digest(stream)==cfg['stream_binding'] and cfg['arm']=='FACT_FIXED_L31_DOWN'
    assert len(stream['tasks'])==45 and stream['track']=='P'
    runtime=load(cfg);target=runtime.target_lock['balancedit']['targets'][0]
    assert target=='model.layers.31.mlp.up_proj'
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    prior=torch.load(old/'private/BANKS.pt',map_location='cpu',weights_only=False)
    statepath=root/'private/BANKS.pt';bank=torch.load(statepath,map_location='cpu',weights_only=False) if statepath.exists() else dict(experts={},routes={},inserted=[],stream=cfg['stream_binding'],code=cfg['code_commit'])
    assert bank['stream']==cfg['stream_binding'] and bank['code']==cfg['code_commit']
    tasks=stream['tasks'];ids=[t['canonical_edit_id'] for t in tasks]
    assert bank['inserted']==ids[:len(bank['inserted'])]
    baseline={r['query_id']:r for r in read(root/'private/FRESH_BASE_OUTPUTS.json')['records']}
    outputpath=root/'private/OUTPUTS.jsonl';existing=[]
    if outputpath.exists():
        text=outputpath.read_text();assert text.endswith('\n'),'Incomplete output append requires audited recovery'
        existing=[json.loads(x) for x in text.splitlines()]
    done={(r['mode'],r['prefix'],r['query_id']):r for r in existing}
    assert len(done)==len(existing),'Duplicate consumers'
    expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(runtime.device).requires_grad_(False)
    record0=record_for(tasks[0]);keycache={}
    def guard():
        check(cfg);assert_base_off(runtime)
        assert not (root/'STOP').exists()
        assert all(p._version==v and p.data_ptr()==ptr and not p.requires_grad for p,v,ptr in frozen)
        if shutil.disk_usage(root).free<8*1024**3:raise OSError('Eight GiB reserve reached')
    def key(record):
        assert_base_off(runtime)
        with torch.inference_mode():return runtime.extract_layer_input_key(runtime.build_question_batch(record),module_path=target,pooling='mean')
    def generated(n,row,mode):
        identity=(mode,n,query_id(row))
        if identity in done:return done[identity]
        guard();raw,_,binding=stage15.prepared(runtime,row,record0);base=baseline[query_id(row)]['output'];assert binding==base['binding']
        began=time.perf_counter()
        if query_id(row) not in keycache:keycache[query_id(row)]=key(replace(record0,question=row['question'],target='',image_path=Path(row['image_path']))).cpu()
        router=MemoryRouter.from_state(dict(distance='euclidean',entries=[bank['routes'][e] for e in ids[:n]]),device=runtime.device)
        decision=router.route(keycache[query_id(row)].to(runtime.device));torch.cuda.synchronize();routing_seconds=time.perf_counter()-began
        selected=decision.logical_edit_id;out=base;weight=None;load_seconds=None;s=None
        if selected is not None:
            s=bank['experts'][selected];began=time.perf_counter();saved=torch.load(s['path'],map_location='cpu',weights_only=True)
            assert saved['step']==320 and digest(saved['binding']['task'])==s['task'] and saved['binding']['code']==s['code']
            expert.load_state_dict(saved['expert']);torch.cuda.synchronize();load_seconds=time.perf_counter()-began;weight=state_hash(expert)
            with bound_hook(runtime,expert,dict(layer_id=s['layer_id'],expert_id=selected,W0_id=s['W0'])) as hook:out=stage15.generate(runtime,raw,binding,hook)
            del saved
        guard();r=dict(arm=cfg['arm'],mode=mode,prefix=n,inserted=ids[:n],query_id=query_id(row),source=row,route=asdict(decision),selected_kind='C_FACT' if selected else 'BASE',writer_layer=s['layer_id'] if s else None,weight_binding=weight,Base=base,output=out,code=cfg['code_commit'],expert_producer_code=s['code'] if s else None,semantic_status='UNJUDGED',routing_seconds=routing_seconds,checkpoint_load_seconds=load_seconds)
        with outputpath.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
        done[identity]=r;return r
    replay=[]
    for position in (1,19,45):
        guard();row=tasks[position-1]['native'];raw,_,binding=stage15.prepared(runtime,row,record0)
        oldbase=baseline[query_id(row)]['output'];assert binding==oldbase['binding']
        current=stage15.generate(runtime,raw,binding)
        assert current['raw_token_ids']==oldbase['raw_token_ids'],'ENVIRONMENT_DRIFT: stop; never mix incompatible Base lanes'
        replay.append(dict(position=position,exact_tokens=True,seconds=current['seconds']))
    write(root/'public/ENVIRONMENT_ACCEPTANCE.json',dict(status='PASS',device_lane=cfg['device_lane'],gpu_uuid=cfg['gpu_uuid'],Base_OFF_replay=replay,runtime=cfg['runtime_lock'],generation=cfg['generation_lock'],tokenizer=type(runtime.adapter.tokenizer).__name__,EOS=runtime.adapter.tokenizer.eos_token_id,backbone_dtype=str(next(runtime.model.parameters()).dtype),writer_dtype=str(next(expert.parameters()).dtype),model_and_vision='Pinned local snapshots through unchanged runtime loader'))
    costs=[];finished=len(bank['inserted']);prefixes=[]
    for task in tasks:
        guard();n=task['order'];edit=task['canonical_edit_id'];record=record_for(task);directory=root/'private/edits'/f'e{n:03d}';directory.mkdir(parents=True,exist_ok=True)
        if edit not in bank['experts']:
            if finished and cfg['gpu_deadline_epoch']-time.time()<max(costs,default=240)*1.25+360:
                print('RESOURCE_BOUNDARY',finished,flush=True);break
            edit_began=time.time()
            # Fresh, layer-bound CP initialization, never transfer an L30 trained state.
            began=time.time()
            adapted=dict(canonical_edit_id=edit,order=n,seed=task['seed'],probes=[task['native']],U=task['U_fit'],fit_questions=task['fit_questions'])
            cp=stage15.initialize(runtime,root,cfg,adapted,record=record,seed_base=20260912,layer_path=layer_path(31))
            initial=LowRankExpert(cp,task['seed'],rank=4).to(runtime.device)
            x=torch.linspace(-1,1,cp.d_in,device=runtime.device).reshape(1,-1)
            assert torch.allclose(cp.residual(x),initial.residual(x),rtol=2e-4,atol=2e-5)
            expert.load_state_dict(initial.state_dict());w0=state_hash(expert);del initial,cp,x
            save(directory/'W0_R4.pt',dict(expert=expert.state_dict(),W0=w0,layer_id=31,task=digest(task),code=cfg['code_commit']))
            write(directory/'INITIALIZATION_RECEIPT.json',dict(writer_layer=31,W0=w0,seconds=time.time()-began,reused_L30_weights=False))
            k=key(record);pos=key(replace(record,question=task['fit_questions'][0]));black=runtime.make_black_image(record,root/'private/black');neg=key(replace(record,image_path=black))
            router=MemoryRouter('euclidean');router.add(edit,k,balanced_radius(k,pos,neg,alpha=.2,distance='euclidean'));entry=router.export_state()['entries'][0];entry['label']=[]
            assert torch.equal(prior['routes'][edit]['key'],entry['key'].cpu()) and prior['routes'][edit]['radius']==entry['radius'],'Frozen router changed'
            bank['routes'][edit]=entry
            teachers=teachers_for(runtime,root,cfg,task,record);began=time.time()
            train(runtime,root,cfg,task,expert,'C_FACT',record,teachers,w0,layer_id=31);del teachers
            s=dict(kind='C_FACT',path=str(directory/'C_FACT/latest.pt'),layer_id=31,W0=w0,task=digest(task),code=cfg['code_commit'],Stage21_training_seconds=time.time()-began)
            bank['experts'][edit]=s;bank['inserted']=ids[:n];save(statepath,bank);costs.append(time.time()-edit_began)
        finished=n
        # A saved bank may precede unfinished generation: fulfill every consumer once.
        first=generated(n,task['native'],'insertion')
        if n in (1,12):
            again=generated(n,task['native'],'integration_replay');assert again['output']['raw_token_ids']==first['output']['raw_token_ids']
            raw,_,binding=stage15.prepared(runtime,task['native'],record);off=stage15.generate(runtime,raw,binding)
            assert off['raw_token_ids']==first['Base']['raw_token_ids']
            write(root/'public'/f'INTEGRATION_{n:03d}.json',dict(status='PASS',Base_OFF=True,save_load=True,Base_inference_seconds=off['seconds']))
        if n in (11,19,32,45):
            from scripts.medtrace.stage20_closeout import expected
            rows=expected(stream,n,final=n==45)
            for row in rows.values():generated(n,row,'endpoint')
            write(root/'public'/f'PREFIX_{n:03d}.json',dict(N=n,arm=cfg['arm'],queries=len(rows),status='GENERATED_NOT_SCORED'));prefixes.append(n)
        write(root/'public/PROGRESS.json',dict(phase='TRAIN_GENERATE',completed=n,total=45,arm=cfg['arm']))
        gc.collect();torch.cuda.empty_cache()
    if not finished:raise RuntimeError('No completed edit before resource boundary')
    if finished!=45:
        from scripts.medtrace.stage20_closeout import expected
        rows=expected(stream,finished,final=True)
        for row in rows.values():generated(finished,row,'endpoint')
        write(root/'public'/f'PREFIX_{finished:03d}.json',dict(N=finished,arm=cfg['arm'],queries=len(rows),status='GENERATED_NOT_SCORED',resource_limited_final=True))
    write(root/'public/GENERATED.json',dict(N=finished,planned_N=45,arm=cfg['arm'],status='GENERATED_NOT_SCORED',prefixes=sorted(set(prefixes+[finished])),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),reused_experts=0,new_experts=finished,full45=finished==45))


if __name__=='__main__':
    campaign(read(os.environ['JOB_CONFIG']))
