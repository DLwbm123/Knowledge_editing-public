"""Stage21 single-arm continuation of a frozen 45-edit bank; no Base edits."""
import os,json,sys,time,gc,shutil
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage19_fasttrack_budget import write,check
from scripts.medtrace.stage19_fasttrack import load,record_for,query_id


def prepare(cfg):
    """CPU-only materialization: never initialize or write into Stage20."""
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace.stage18_cfact import state_hash
    from scripts.medtrace.run_selective_write import save
    root=Path(cfg['run']);old=Path(cfg['stage20_run']);stream=read(old/'private/STREAM.json')
    assert digest(stream)==cfg['stream_binding'] and len(stream['tasks'])==45
    for name in ('STREAM.json','FRESH_BASE_OUTPUTS.json'):
        path=root/'private'/name
        if path.exists():assert read(path)==read(old/'private'/name)
        else:shutil.copy2(old/'private'/name,path)
    for name in ('base','teacher'):
        if not (root/'private'/name).exists():shutil.copytree(old/'private'/name,root/'private'/name)
    inventory=[]
    for t in stream['tasks']:
        d=old/'private/edits'/f"e{t['order']:03d}";cp=AsymmetricCPExpert(14336,4096,4)
        saved=torch.load(d/'initial/W0_COMPLETE.pt',map_location='cpu',weights_only=True)
        assert saved['canonical_edit_id']==t['canonical_edit_id']
        cp.load_state_dict(saved['expert']);expert=LowRankExpert(cp,t['seed'],rank=4)
        w0=state_hash(expert);assert w0==read(d/'C_FACT/TRAINING.json')['W0']
        path=root/'private/W0'/f"e{t['order']:03d}.pt"
        if not path.exists():save(path,dict(expert=expert.state_dict(),W0=w0,task=digest(t),origin_layer=read(d/'SELECTION.json')['layer_id']))
        inventory.append(dict(position=t['order'],W0_exact=True,bytes=path.stat().st_size))
    write(root/'public/W0_REUSE_AUDIT.json',dict(rows=inventory,reconstructed_on_CPU=True,GPU_initialization_repeated=False))


def campaign(cfg):
    from scripts.medtrace.stage17_single import setup
    setup(cfg)
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from methods.medtrace.hsic import collect_features,select_layer,bound_hook
    from m3bench_repro.editors.routing import MemoryRouter,balanced_radius
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import assert_base_off,state_hash,teachers_for,train
    from scripts.medtrace.stage20_runtime import hsic_diagnostics
    from scripts.medtrace.run_selective_write import save
    root=Path(cfg['run']);old=Path(cfg['stage20_run']);stream=read(root/'private/STREAM.json')
    assert digest(stream)==cfg['stream_binding'] and cfg['arm']=='NO_H_HSIC'
    assert len(stream['tasks'])==45 and stream['track']=='P'
    runtime=load(cfg);target=runtime.target_lock['balancedit']['targets'][0]
    assert target=='model.layers.31.mlp.up_proj'
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    prior=torch.load(old/'private/BANKS.pt',map_location='cpu',weights_only=False)
    old_noh=torch.load(old/'private/ablation/private/BANKS.pt',map_location='cpu',weights_only=False)
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
        guard();r=dict(arm=cfg['arm'],mode=mode,prefix=n,inserted=ids[:n],query_id=query_id(row),source=row,route=asdict(decision),selected_kind='C_NO_H' if selected else 'BASE',writer_layer=s['layer_id'] if s else None,weight_binding=weight,Base=base,output=out,code=cfg['code_commit'],expert_producer_code=s['code'] if s else None,semantic_status='UNJUDGED',routing_seconds=routing_seconds,checkpoint_load_seconds=load_seconds)
        with outputpath.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
        done[identity]=r;return r
    for task in tasks:
        guard();n=task['order'];edit=task['canonical_edit_id'];record=record_for(task);directory=root/'private/edits'/f'e{n:03d}';directory.mkdir(parents=True,exist_ok=True)
        if edit not in bank['experts']:
            selectionpath=directory/'SELECTION.json'
            if selectionpath.exists():selection=read(selectionpath);assert selection['task']==digest(task)
            else:
                features,counts=collect_features(runtime,record,task['fit_questions']);selection=select_layer(features)
                selection.update(task=digest(task),token_counts=counts,diagnostics=hsic_diagnostics(features,selection));write_new(selectionpath,selection);del features
            w0=torch.load(root/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True)
            assert w0['task']==digest(task) and selection['layer_id']==w0['origin_layer'],'Selector changed: do not silently reuse differently located W0'
            k=key(record);pos=key(replace(record,question=task['fit_questions'][0]));black=runtime.make_black_image(record,root/'private/black');neg=key(replace(record,image_path=black))
            router=MemoryRouter('euclidean');router.add(edit,k,balanced_radius(k,pos,neg,alpha=.2,distance='euclidean'));entry=router.export_state()['entries'][0];entry['label']=[]
            assert torch.equal(prior['routes'][edit]['key'],entry['key'].cpu()) and prior['routes'][edit]['radius']==entry['radius'],'Frozen router changed'
            bank['routes'][edit]=entry
            if n<=11:
                s=dict(old_noh['banks']['A'][edit]);saved=torch.load(s['path'],map_location='cpu',weights_only=True)
                assert s['kind']=='C_NO_H' and s['task']==digest(task) and s['layer_id']==selection['layer_id'] and saved['step']==320 and saved['binding']['W0']==w0['W0']
                assert saved['binding']['runtime']==cfg['runtime_lock'] and saved['binding']['generation']==cfg['generation_lock']
                assert s['code'] in cfg['approved_predecessor_commits']
                s['reuse']='Stage20 completed NO_H pure11';s['Stage21_training_seconds']=0
            else:
                expert.load_state_dict(w0['expert']);assert state_hash(expert)==w0['W0']
                teachers=teachers_for(runtime,root,cfg,task,record);began=time.time()
                train(runtime,root,cfg,task,expert,'C_NO_H',record,teachers,w0['W0'],layer_id=selection['layer_id']);del teachers
                s=dict(kind='C_NO_H',path=str(directory/'C_NO_H/latest.pt'),layer_id=selection['layer_id'],W0=w0['W0'],task=digest(task),code=cfg['code_commit'],Stage21_training_seconds=time.time()-began)
            bank['experts'][edit]=s;bank['inserted']=ids[:n];save(statepath,bank)
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
            write(root/'public'/f'PREFIX_{n:03d}.json',dict(N=n,arm=cfg['arm'],queries=len(rows),status='GENERATED_NOT_SCORED'))
        write(root/'public/PROGRESS.json',dict(phase='TRAIN_GENERATE',completed=n,total=45,arm=cfg['arm']))
        gc.collect();torch.cuda.empty_cache()
    write(root/'public/GENERATED.json',dict(N=45,arm=cfg['arm'],status='GENERATED_NOT_SCORED',prefixes=[11,19,32,45],peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),reused_experts=11,new_experts=34))


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    if cfg['phase']=='prepare':prepare(cfg)
    elif cfg['phase']=='NO_H_HSIC':campaign(cfg)
    else:raise ValueError('Unregistered phase')
