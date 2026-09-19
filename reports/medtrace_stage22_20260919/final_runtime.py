"""Stage23 frozen45 regression before one qualified, source-held-out probe pass."""
import os,sys,json,time,gc,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write,check
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage19_fasttrack import load,record_for,query_id
from scripts.medtrace.stage20_closeout import outputs,expected
from continuation_runtime import emit
from coverage import task_with_support
from report import WEIGHTS


def campaign(cfg):
    from scripts.medtrace.stage17_single import setup
    setup(cfg)
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import assert_base_off,state_hash,teachers_for
    from scripts.medtrace.run_selective_write import save
    from training import train
    assert cfg['phase']=='23'
    root=Path(cfg['run']);p=root/'private';pub=root/'public'
    lock=read(p/'FINAL_STAGE23_LOCK.json');confirm=read(p/'CONFIRM_QUALIFIED_FREEZE.json');stream=read(p/'STREAM.json')
    assert digest({k:v for k,v in lock.items() if k!='binding'})==lock['binding']==cfg['final_lock_binding']
    assert digest({k:v for k,v in confirm.items() if k!='binding'})==confirm['binding']==lock['CONFIRM_binding']
    assert digest(stream)==cfg['stream_binding'] and read(pub/'EXIT_22C.json')['exit_code']==0
    assert lock['N']==45 and lock['prefixes']==[11,19,32,45] and len(lock['banks'])==len(set(lock['banks']))
    selection=read(p/'H_SELECTION_FREEZE.json');pool=read(p/'COVERAGE_POOL_FREEZE.json')
    assert selection['pool_binding']==pool['binding']==cfg['coverage_pool_binding']
    choices={r['edit_id']:r for r in selection['rows']}
    tasks=stream['tasks'];ids=[t['canonical_edit_id'] for t in tasks];record=record_for(tasks[0])
    base={r['query_id']:r['output'] for r in read(p/'HISTORICAL_BASE.json')['records']}
    assert set(expected(stream,45,final=True))<=set(base)
    runtime=load(cfg);frozen=[(x,x._version,x.data_ptr()) for x in runtime.model.parameters()]
    def guard():
        check(cfg);assert_base_off(runtime);assert not (root/'STOP').exists()
        assert all(x._version==v and x.data_ptr()==ptr and not x.requires_grad for x,v,ptr in frozen)
        if shutil.disk_usage(root).free<8*1024**3:raise OSError('Eight GiB reserve reached; preserve recovery and registered consumers')
    prior=torch.load(Path(cfg['stage20_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False)
    noh=torch.load(Path(cfg['stage21A_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False)
    keys={};accept=[]
    for task in tasks:
        guard()
        with torch.inference_mode():key=runtime.extract_layer_input_key(runtime.build_question_batch(record_for(task)),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
        assert torch.equal(key,prior['routes'][task['canonical_edit_id']]['key'])
        keys[query_id(task['native'])]=key
        if task['order'] in (1,19,45):
            raw,_,binding=stage15.prepared(runtime,task['native'],record);out=stage15.generate(runtime,raw,binding)
            assert binding==base[query_id(task['native'])]['binding'] and out['raw_token_ids']==base[query_id(task['native'])]['raw_token_ids']
            accept.append(dict(position=task['order'],exact_tokens=True))
    write(pub/'ENVIRONMENT_23.json',dict(status='PASS',Base_OFF=accept,native_keys=45,gpu_uuid=cfg['gpu_uuid'],final_lock=lock['binding']))
    routers={n:MemoryRouter.from_state(dict(distance='euclidean',entries=[prior['routes'][e] for e in ids[:n]]),device=runtime.device) for n in range(1,46)}
    expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(runtime.device).requires_grad_(False)
    existing=outputs(root);done={(r['arm'],r['mode'],r['prefix'],r['query_id']) for r in existing};assert len(done)==len(existing)
    banks={}
    for arm in lock['banks']:
        weights=dict(zip(('H','U'),WEIGHTS[arm])) if arm in WEIGHTS else lock['weights']
        ancestor=torch.load(p/f'BANKS_{arm}.pt',map_location='cpu',weights_only=False)
        assert ancestor['weights']==weights and ancestor['inserted']==ids[:19]
        if arm.startswith('S'):assert ancestor['selection_binding']==selection['freeze_id']
        bankpath=p/f'BANKS_23_{arm}.pt'
        bank=torch.load(bankpath,map_location='cpu',weights_only=False) if bankpath.exists() else dict(arm=arm,weights=weights,experts={},inserted=[],routes=prior['routes'],code=cfg['code_commit'],final_lock=lock['binding'])
        assert bank['final_lock']==lock['binding'] and bank['weights']==weights and bank['code']==cfg['code_commit']
        ar=p/'arms'/('R_'+arm);(ar/'private').mkdir(parents=True,exist_ok=True);(ar/'public').mkdir(exist_ok=True)
        for name in ('base','teacher'):
            if not (ar/'private'/name).exists():(ar/'private'/name).symlink_to(p/name,target_is_directory=True)
        for original in tasks:
            guard();edit=original['canonical_edit_id'];n=original['order']
            task=task_with_support(original,[pool['candidates'][q] for q in choices[edit][arm]]) if arm.startswith('S') else original
            if edit not in bank['experts']:
                w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True)
                assert w0['origin_layer']==30 and w0['task']==digest(original)
                if n<=19 or arm in ('E0','E1'):
                    s=dict(ancestor['experts'][edit] if n<=19 else noh['experts'][edit] if arm=='E0' else prior['banks']['A'][edit])
                    saved=torch.load(s['path'],map_location='cpu',weights_only=True);b=saved['binding']
                    assert saved['step']==320 and digest(b['task'])==digest(task) and b['W0']==w0['W0'] and s['layer_id']==30
                    assert b['runtime']==cfg['runtime_lock'] and b['generation']==cfg['generation_lock'] and b['code']==s['code']
                    if 'weights' in b:assert b['weights']==weights
                    else:assert arm in ('E0','E1') and b['branch']==('C_NO_H' if arm=='E0' else 'C_FACT')
                    s['reused']=True;del saved
                else:
                    expert.load_state_dict(w0['expert']);assert state_hash(expert)==w0['W0']
                    teachers=teachers_for(runtime,root,cfg,task,record_for(task));began=time.time();branch='C_FACT' if weights['H'] else 'C_NO_H'
                    train(runtime,ar,dict(cfg,weights=weights),task,expert,branch,record_for(task),teachers,w0['W0'],layer_id=30);del teachers
                    s=dict(kind=branch,path=str(ar/'private/edits'/f'e{n:03d}'/branch/'latest.pt'),layer_id=30,W0=w0['W0'],task=digest(task),original_task=digest(original),code=cfg['code_commit'],training_seconds=time.time()-began,reused=False)
                bank['experts'][edit]=s;bank['inserted']=ids[:n];save(bankpath,bank)
            emit(runtime,cfg,root,expert,bank,[original['native']],base,routers[n],keys,'R_'+arm,n,'insertion',record,guard,done)
            if n in lock['prefixes']:
                rows=list(expected(stream,n,final=n==45).values())
                emit(runtime,cfg,root,expert,bank,rows,base,routers[n],keys,'R_'+arm,n,'endpoint',record,guard,done)
                write(pub/f'PREFIX_23_{arm}_{n:03d}.json',dict(arm=arm,N=n,queries=len(rows),status='GENERATED_NOT_SCORED'))
            write(pub/'PROGRESS_23.json',dict(phase='45_REGRESSION',arm=arm,completed=n,total=45));gc.collect();torch.cuda.empty_cache()
        banks[arm]=bank
        write(pub/f'REGRESSION_23_{arm}.json',dict(N=45,formal_outputs=298,reused_experts=sum(s['reused'] for s in bank['experts'].values()),new_experts=sum(not s['reused'] for s in bank['experts'].values())))
    # A single sealed probe pass is permitted only after every frozen bank reaches45.
    write(pub/'REGRESSION_23_COMPLETE.json',dict(banks=lock['banks'],N=45,final_lock=lock['binding']))
    probe_file=p/'CONFIRM_BASE_23.json';probe=read(probe_file) if probe_file.exists() else dict(records=[],binding=confirm['binding'],final_lock=lock['binding'])
    assert probe['binding']==confirm['binding'] and probe['final_lock']==lock['binding']
    completed={r['query_id'] for r in probe['records']}
    for row in confirm['rows']:
        guard();q=query_id(row)
        if q in completed:continue
        raw,_,binding=stage15.prepared(runtime,row,record);out=stage15.generate(runtime,raw,binding)
        probe['records'].append(dict(query_id=q,source=row,output=out,Base=out,mode='CONFIRM_BASE',arm='Base',prefix=45));write(probe_file,probe)
    write(probe_file,probe);probe_base={r['query_id']:r['output'] for r in probe['records']}
    write(pub/'CONFIRM_BASE_23_READY.json',dict(queries=len(probe_base),binding=confirm['binding'],method_independent=True))
    for arm in lock['banks']:
        emit(runtime,cfg,root,expert,banks[arm],confirm['rows'],probe_base,routers[45],keys,'R_'+arm,45,'CONFIRM',record,guard,done)
        write(pub/f'CONFIRM_23_{arm}.json',dict(queries=len(confirm['rows']),N=45,arm=arm,once_only=True))
    write(pub/'GENERATED_23.json',dict(banks=lock['banks'],N=45,regression_per_bank=298,CONFIRM_queries=len(confirm['rows']),status='GENERATED_NOT_SCORED',peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),final_lock=lock['binding']))


if __name__=='__main__':campaign(read(os.environ['JOB_CONFIG']))
