"""Registered H-coverage consumer; existing six-arm workers are unchanged."""
import os,sys,json,time,gc
from pathlib import Path
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write,check
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage19_fasttrack import load,record_for,query_id
from scripts.medtrace.astra_judge_bundle import write_new


def emit(runtime,cfg,root,expert,bank,rows,base,router,keys,arm,prefix,mode,record,guard,done):
    import torch
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import state_hash
    from methods.medtrace.hsic import bound_hook
    outputfile=Path(root)/'private/OUTPUTS.jsonl'
    for row in rows:
        guard();q=query_id(row);identity=(arm,mode,prefix,q)
        if identity in done:continue
        raw,_,binding=stage15.prepared(runtime,row,record);assert binding==base[q]['binding']
        began=time.perf_counter()
        if q not in keys:
            with torch.inference_mode():
                keys[q]=runtime.extract_layer_input_key(runtime.build_question_batch(replace(record,question=row['question'],target='',image_path=Path(row['image_path']))),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
        decision=router.route(keys[q].to(runtime.device));torch.cuda.synchronize();route_seconds=time.perf_counter()-began
        selected=decision.logical_edit_id;out=base[q];s=None;weight=None;load_seconds=None
        if selected is not None:
            s=bank['experts'][selected];began=time.perf_counter();saved=torch.load(s['path'],map_location='cpu',weights_only=True)
            assert saved['step']==320 and digest(saved['binding']['task'])==s['task'] and saved['binding']['code']==s['code']
            assert saved['binding']['W0']==s['W0'] and s['layer_id']==30
            if not s.get('reused',False):assert saved['binding']['weights']==bank['weights']
            expert.load_state_dict(saved['expert']);torch.cuda.synchronize();load_seconds=time.perf_counter()-began;weight=state_hash(expert)
            with bound_hook(runtime,expert,dict(layer_id=30,expert_id=selected,W0_id=s['W0'])) as hook:out=stage15.generate(runtime,raw,binding,hook)
            del saved
        guard()
        r=dict(arm=arm,weights=bank['weights'],prefix=prefix,mode=mode,query_id=q,source=row,route=asdict(decision),output=out,Base=base[q],writer_layer=30 if selected else None,weight_binding=weight,expert_producer_code=s['code'] if s else None,code=cfg['code_commit'],device_lane=cfg['device_lane'],routing_seconds=route_seconds,checkpoint_load_seconds=load_seconds)
        with outputfile.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
        done.add(identity)


def campaign(cfg):
    from scripts.medtrace.stage17_single import setup
    setup(cfg)
    import torch,shutil
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace.stage18_cfact import assert_base_off,state_hash,teachers_for
    from scripts.medtrace.run_selective_write import save
    from scripts.medtrace.stage20_closeout import outputs
    from coverage import build_selections,task_with_support
    from training import train
    assert cfg['phase']=='22C'
    root=Path(cfg['run']);p=root/'private';pub=root/'public'
    assert read(pub/'EXIT_22B.json')['exit_code']==0
    lock=read(pub/'COVERAGE_WEIGHT_LOCK.json');assert digest({k:v for k,v in lock.items() if k!='binding'})==lock['binding']
    stream=read(p/'STREAM.json');dev=read(p/'DEV_PANEL.json');pool=read(p/'COVERAGE_POOL_FREEZE.json')
    assert digest(stream)==cfg['stream_binding'] and dev['binding']==cfg['DEV_binding']
    assert pool['binding']==cfg['coverage_pool_binding']==lock['pool_binding']
    tasks=stream['tasks'][:19];ids=[t['canonical_edit_id'] for t in tasks];rows=dev['natives']+dev['rows']
    assert len(rows)==66
    runtime=load(cfg);frozen=[(x,x._version,x.data_ptr()) for x in runtime.model.parameters()]
    def guard():
        check(cfg);assert_base_off(runtime);assert not (root/'STOP').exists()
        assert all(x._version==v and x.data_ptr()==ptr and not x.requires_grad for x,v,ptr in frozen)
        if shutil.disk_usage(root).free<8*1024**3:raise OSError('Eight GiB reserve reached')
    prior=torch.load(Path(cfg['stage20_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False)
    if (p/'H_SELECTION_FREEZE.json').exists():
        selection=read(p/'H_SELECTION_FREEZE.json');assert selection['pool_binding']==pool['binding']
        assert digest({k:v for k,v in selection.items() if k!='freeze_id'})==selection['freeze_id']
        assert selection['binding']['runtime']==cfg['runtime_lock'] and selection['binding']['generation']==cfg['generation_lock'] and selection['binding']['gpu_uuid']==cfg['gpu_uuid']
    else:selection=build_selections(runtime,root,cfg,pool,stream,prior,guard)
    if all(r['same_selection'] for r in selection['rows'][:19]):
        write(pub/'GENERATED_22C.json',dict(status='UNSUPPORTED_NO_SELECTION_DIFFERENCE',trained_arms=[],queries=0));return
    choices={r['edit_id']:r for r in selection['rows']};base={r['query_id']:r['output'] for r in read(p/'DEV_BASE.json')['records']}
    assert set(base)=={query_id(r) for r in rows}
    record=record_for(tasks[0]);keys={}
    for t in tasks:
        guard()
        with torch.inference_mode():key=runtime.extract_layer_input_key(runtime.build_question_batch(record_for(t)),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
        assert torch.equal(key,prior['routes'][t['canonical_edit_id']]['key']);keys[query_id(t['native'])]=key
    router=MemoryRouter.from_state(dict(distance='euclidean',entries=[prior['routes'][e] for e in ids]),device=runtime.device)
    expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(runtime.device).requires_grad_(False)
    done={(r['arm'],r['mode'],r['prefix'],r['query_id']) for r in outputs(root)}
    for arm in ('S1','S2'):
        ar=p/'arms'/arm;(ar/'private').mkdir(parents=True,exist_ok=True);(ar/'public').mkdir(exist_ok=True)
        for name in ('base','teacher'):
            if not (ar/'private'/name).exists():(ar/'private'/name).symlink_to(p/name,target_is_directory=True)
        bankfile=p/f'BANKS_{arm}.pt'
        bank=torch.load(bankfile,map_location='cpu',weights_only=False) if bankfile.exists() else dict(arm=arm,weights=lock['weights'],experts={},routes={e:prior['routes'][e] for e in ids},inserted=[],code=cfg['code_commit'],selection_binding=selection['freeze_id'],weight_lock=lock['binding'])
        assert bank['code']==cfg['code_commit'] and bank['weight_lock']==lock['binding'] and bank['selection_binding']==selection['freeze_id'] and bank['weights']==lock['weights']
        for original in tasks:
            guard();n=original['order'];edit=original['canonical_edit_id']
            if edit in bank['experts']:continue
            selected=[pool['candidates'][q] for q in choices[edit][arm]];task=task_with_support(original,selected)
            w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True)
            assert w0['origin_layer']==30 and w0['task']==digest(original)
            expert.load_state_dict(w0['expert']);assert state_hash(expert)==w0['W0']
            teachers=teachers_for(runtime,root,cfg,task,record_for(task));began=time.time()
            train(runtime,ar,dict(cfg,weights=lock['weights']),task,expert,'C_FACT',record_for(task),teachers,w0['W0'],layer_id=30);del teachers
            bank['experts'][edit]=dict(kind='C_FACT',path=str(ar/'private/edits'/f'e{n:03d}'/'C_FACT/latest.pt'),layer_id=30,W0=w0['W0'],task=digest(task),original_task=digest(original),code=cfg['code_commit'],training_seconds=time.time()-began,reused=False)
            bank['inserted']=ids[:n];save(bankfile,bank);write(pub/'PROGRESS.json',dict(phase='22C_TRAIN',arm=arm,completed=n,total=19))
        emit(runtime,cfg,root,expert,bank,rows,base,router,keys,arm,19,'DEV',record,guard,done)
        write(pub/f'ARM_{arm}.json',dict(arm=arm,N=19,queries=66,status='GENERATED_NOT_SCORED',weights=lock['weights'],selection_binding=selection['freeze_id']))
        gc.collect();torch.cuda.empty_cache()
    write(pub/'GENERATED_22C.json',dict(arms=['S1','S2'],N=19,queries=132,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),status='GENERATED_NOT_SCORED'))


if __name__=='__main__':campaign(read(os.environ['JOB_CONFIG']))
