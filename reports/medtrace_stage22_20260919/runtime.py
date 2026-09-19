"""Fixed-L30 DEV execution. All inference uses the unchanged natural router."""
import os,sys,json,time,gc,shutil
from pathlib import Path
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import write,check
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage19_fasttrack import load,record_for,query_id
ARMS={'E0':(0.,.01),'E1':(1.,.01),'E3':(0.,.05),'E2':(.25,.01),'E4':(1.,.05),'E5':(.25,.05)}


def campaign(cfg):
    from scripts.medtrace.stage17_single import setup
    setup(cfg)
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from methods.medtrace.hsic import bound_hook
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import assert_base_off,state_hash,teachers_for
    from scripts.medtrace.run_selective_write import save
    from training import train
    root=Path(cfg['run']);p=root/'private';pub=root/'public';stream=read(p/'STREAM.json');dev=read(p/'DEV_PANEL.json')
    assert digest(stream)==cfg['stream_binding'] and digest({k:v for k,v in dev.items() if k!='binding'})==dev['binding']==cfg['DEV_binding']
    assert cfg['phase'] in ('22A','22B') and read(p/'REUSE_AUDIT.json')['status']=='CPU_COMPATIBLE_PENDING_CURRENT_GPU_MECHANICAL_REPLAY'
    tasks=stream['tasks'][:19];ids=[t['canonical_edit_id'] for t in tasks];rows=dev['natives']+dev['rows'];assert len(rows)==66
    runtime=load(cfg);target=runtime.target_lock['balancedit']['targets'][0];assert target=='model.layers.31.mlp.up_proj'
    frozen=[(x,x._version,x.data_ptr()) for x in runtime.model.parameters()]
    def guard():
        check(cfg);assert_base_off(runtime);assert not (root/'STOP').exists()
        assert all(x._version==v and x.data_ptr()==ptr and not x.requires_grad for x,v,ptr in frozen)
        if shutil.disk_usage(root).free<8*1024**3:raise OSError('Eight GiB reserve reached')
    prior=torch.load(Path(cfg['stage20_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False)
    noh=torch.load(Path(cfg['stage21A_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False)
    record0=record_for(tasks[0]);baseline={r['query_id']:r for r in read(p/'HISTORICAL_BASE.json')['records']}
    if cfg['phase']=='22A':
        targetfile=p/'DEV_BASE.json';existing=read(targetfile)['records'] if targetfile.exists() else [];done={r['query_id'] for r in existing}
        for row in rows:
            guard();q=query_id(row)
            if q in done:continue
            raw,_,binding=stage15.prepared(runtime,row,record0);out=stage15.generate(runtime,raw,binding);old=baseline[q]['output']
            assert binding==old['binding'] and out['raw_token_ids']==old['raw_token_ids'],'Current lane changed: stop before method comparisons'
            existing.append(dict(query_id=q,source=row,output=out,Base=out,mode='Base',arm='Base',prefix=19));write(targetfile,dict(records=existing,device_lane=cfg['device_lane'],DEV_binding=dev['binding']))
            write(pub/'PROGRESS.json',dict(phase='22A_BASE',completed=len(existing),total=66))
        write(pub/'GENERATED_22A.json',dict(queries=66,status='BASE_GENERATED_EXACT_HISTORICAL_TOKEN_PARITY',DEV_binding=dev['binding'],gpu_uuid=cfg['gpu_uuid']));return
    assert read(pub/'BASE_SCORE_ACCEPTANCE.json')['DEV_binding']==dev['binding']
    base={r['query_id']:r['output'] for r in read(p/'DEV_BASE.json')['records']};assert set(base)=={query_id(r) for r in rows}
    # Old route states are reused only after checking every native key in the current lane.
    keycache={}
    for t in tasks:
        guard()
        with torch.inference_mode():key=runtime.extract_layer_input_key(runtime.build_question_batch(record_for(t)),module_path=target,pooling='mean').cpu()
        assert torch.equal(key,prior['routes'][t['canonical_edit_id']]['key']);keycache[query_id(t['native'])]=key
    router=MemoryRouter.from_state(dict(distance='euclidean',entries=[prior['routes'][e] for e in ids]),device=runtime.device)
    expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(runtime.device).requires_grad_(False)
    outputfile=p/'OUTPUTS.jsonl';existing=[]
    if outputfile.exists():
        text=outputfile.read_text();assert text.endswith('\n');existing=[json.loads(x) for x in text.splitlines()]
    done={(r['arm'],r['query_id']):r for r in existing};assert len(done)==len(existing)
    for arm,(H,U) in ARMS.items():
        ar=p/'arms'/arm;(ar/'private').mkdir(parents=True,exist_ok=True);(ar/'public').mkdir(exist_ok=True)
        for name in ('base','teacher'):
            if not (ar/'private'/name).exists():(ar/'private'/name).symlink_to(p/name,target_is_directory=True)
        bankfile=p/f'BANKS_{arm}.pt';bank=torch.load(bankfile,map_location='cpu',weights_only=False) if bankfile.exists() else dict(arm=arm,weights=dict(H=H,U=U),experts={},routes={e:prior['routes'][e] for e in ids},inserted=[],code=cfg['code_commit'],DEV_binding=dev['binding'])
        assert bank['code']==cfg['code_commit'] and bank['weights']==dict(H=H,U=U) and bank['inserted']==ids[:len(bank['inserted'])]
        for t in tasks:
            guard();edit=t['canonical_edit_id'];n=t['order']
            if edit in bank['experts']:continue
            w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True);assert w0['origin_layer']==30 and w0['task']==digest(t)
            if arm in ('E0','E1'):
                s=dict(noh['experts'][edit] if arm=='E0' else prior['banks']['A'][edit]);d=torch.load(s['path'],map_location='cpu',weights_only=True)
                assert d['step']==320 and d['binding']['W0']==w0['W0'] and digest(d['binding']['task'])==digest(t) and s['layer_id']==30
                assert d['binding']['runtime']==cfg['runtime_lock'] and d['binding']['generation']==cfg['generation_lock'];s['reused']=True;del d
            else:
                expert.load_state_dict(w0['expert']);assert state_hash(expert)==w0['W0'];record=record_for(t)
                teachers=teachers_for(runtime,root,cfg,t,record);branch='C_FACT' if H else 'C_NO_H';began=time.time()
                train(runtime,ar,dict(cfg,weights=dict(H=H,U=U)),t,expert,branch,record,teachers,w0['W0'],layer_id=30);del teachers
                s=dict(kind=branch,path=str(ar/'private/edits'/f'e{n:03d}'/branch/'latest.pt'),layer_id=30,W0=w0['W0'],task=digest(t),code=cfg['code_commit'],training_seconds=time.time()-began,reused=False)
            bank['experts'][edit]=s;bank['inserted']=ids[:n];save(bankfile,bank);write(pub/'PROGRESS.json',dict(phase='22B_TRAIN',arm=arm,completed=n,total=19))
        for row in rows:
            guard();q=query_id(row)
            if (arm,q) in done:continue
            raw,_,binding=stage15.prepared(runtime,row,record0);assert binding==base[q]['binding']
            began=time.perf_counter()
            if q not in keycache:
                with torch.inference_mode():keycache[q]=runtime.extract_layer_input_key(runtime.build_question_batch(replace(record0,question=row['question'],target='',image_path=Path(row['image_path']))),module_path=target,pooling='mean').cpu()
            decision=router.route(keycache[q].to(runtime.device));torch.cuda.synchronize();route_seconds=time.perf_counter()-began
            selected=decision.logical_edit_id;out=base[q];s=None;weight=None;load_seconds=None
            if selected is not None:
                s=bank['experts'][selected];began=time.perf_counter();saved=torch.load(s['path'],map_location='cpu',weights_only=True)
                assert saved['step']==320 and digest(saved['binding']['task'])==s['task'] and saved['binding']['code']==s['code']
                if not s['reused']:assert saved['binding']['weights']==dict(H=H,U=U)
                expert.load_state_dict(saved['expert']);torch.cuda.synchronize();load_seconds=time.perf_counter()-began;weight=state_hash(expert)
                with bound_hook(runtime,expert,dict(layer_id=30,expert_id=selected,W0_id=s['W0'])) as hook:out=stage15.generate(runtime,raw,binding,hook)
                del saved
            guard();r=dict(arm=arm,weights=dict(H=H,U=U),prefix=19,mode='DEV',query_id=q,source=row,route=asdict(decision),output=out,Base=base[q],writer_layer=30 if selected else None,weight_binding=weight,expert_producer_code=s['code'] if s else None,code=cfg['code_commit'],device_lane=cfg['device_lane'],routing_seconds=route_seconds,checkpoint_load_seconds=load_seconds)
            with outputfile.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
            done[(arm,q)]=r
        write(pub/f'ARM_{arm}.json',dict(arm=arm,N=19,queries=66,status='GENERATED_NOT_SCORED',weights=dict(H=H,U=U)));gc.collect();torch.cuda.empty_cache()
    write(pub/'GENERATED_22B.json',dict(arms=list(ARMS),N=19,queries=396,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),status='GENERATED_NOT_SCORED'))

if __name__=='__main__':campaign(read(os.environ['JOB_CONFIG']))
