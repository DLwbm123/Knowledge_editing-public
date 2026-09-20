"""Stage24E bounded three-arm worker.  All scheduling comes from COMPILED_PLAN."""
import os,sys,json,time,shutil,gc
from pathlib import Path
from dataclasses import replace
ROOT=Path(os.environ.get('E_CODE_ROOT','/root/rivermind-data/job-524e/code'))
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports/medtrace_stage22_20260919'))
from scripts.medtrace.stage17_single import setup
from scripts.medtrace.stage19_fasttrack_budget import read,write,check
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_score import query_id
from scripts.medtrace.stage19_fasttrack import load,record_for

def main(cfg):
    setup(cfg)
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert,optimizer_for
    from methods.medtrace.hsic import bound_hook
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import assert_base_off,state_hash,teachers_for,source_batch
    from scripts.medtrace.run_selective_write import save
    from training import train,update
    root=Path(cfg['run']); p=root/'private'; pub=root/'public'; common=Path(cfg['common_run'])
    plan=read(p/'COMPILED_PLAN.json'); stream=read(common/'private/STREAM.json'); tasks=stream['tasks'][:19]
    ids=[t['canonical_edit_id'] for t in tasks]; prior=torch.load(Path(cfg['stage20_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False)
    dev=read(p/'DEV_UNION.json'); rows=dev['rows']
    baseline={}
    for bp in (common/'private/HISTORICAL_BASE.json', common/'private/DEV_BASE.json', p/'BASE.json'):
        if bp.exists(): baseline.update({r['query_id']:r['output'] for r in read(bp)['records']})
    rt=load(cfg); torch.use_deterministic_algorithms(True)
    frozen=[(x,x._version,x.data_ptr()) for x in rt.model.parameters()]
    def guard():
        check(cfg); assert_base_off(rt); assert not (root/'STOP').exists()
        assert all(x._version==v and x.data_ptr()==ptr and not x.requires_grad for x,v,ptr in frozen)
        if shutil.disk_usage(root).free < 8*1024**3: raise OSError('8GiB reserve reached')
    keys={}; row_by_q={query_id(r):r for t in tasks for role in ('H_fit','U_fit') for r in t[role]}; routers={n:MemoryRouter.from_state(dict(distance='euclidean',entries=[prior['routes'][e] for e in ids[:n]]),device=rt.device) for n in range(1,20)}
    ex=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(rt.device).requires_grad_(False)
    slot_by_t={s['t']:s for s in plan['slots']}
    def emit(arm,n,rec,bank,query_rows):
        outpath=p/'OUTPUTS.jsonl'; existing={(x['arm'],x['prefix'],x['query_id'],x['mode']) for x in (json.loads(l) for l in outpath.read_text().splitlines())} if outpath.exists() else set()
        for row in query_rows:
            guard(); q=query_id(row); raw,_,binding=stage15.prepared(rt,row,rec); base=baseline[q]
            if q not in keys:
                with torch.inference_mode(): keys[q]=rt.extract_layer_input_key(rt.build_question_batch(replace(rec,question=row['question'],target='',image_path=Path(row['image_path']))),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
            d=routers[n].route(keys[q].to(rt.device)); selected=d.logical_edit_id; o=base; wh=None
            if selected:
                s=bank['experts'][selected]; v=torch.load(s['path'],map_location='cpu',weights_only=True); assert v['step'] in (20,320) and v['binding']['W0']==s['W0']; ex.load_state_dict(v['expert']); wh=state_hash(ex)
                with bound_hook(rt,ex,dict(layer_id=30,expert_id=selected,W0_id=s['W0'])) as hook: o=stage15.generate(rt,raw,binding,hook)
            key=(arm,n,q,'natural')
            if key not in existing:
                with outpath.open('a') as f:f.write(json.dumps(dict(arm=arm,prefix=n,mode='natural',query_id=q,source=row,route=d.__dict__,output=o,Base=base,weight_binding=wh,code=cfg['code_commit']))+'\n')
        write(pub/'PROGRESS.json',dict(status='RUNNING',arm=arm,prefix=n,outputs=sum(1 for _ in outpath.open()) if outpath.exists() else 0))
    for arm in ('B0','B1','B2'):
        ar=p/'arms'/arm; (ar/'private').mkdir(parents=True,exist_ok=True); (ar/'public').mkdir(exist_ok=True)
        (ar/'private').mkdir(parents=True,exist_ok=True)
        bankfile=p/f'BANKS_{arm}.pt'; bank=torch.load(bankfile,map_location='cpu',weights_only=False) if bankfile.exists() else dict(arm=arm,weights=dict(H=.25,U=.01),experts={},routes={e:prior['routes'][e] for e in ids},inserted=[],code=cfg['code_commit'])
        for task in tasks:
            guard(); n=task['order']; e=task['canonical_edit_id']
            if e not in bank['experts']:
                w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True); ex.load_state_dict(w0['expert']); assert state_hash(ex)==w0['W0']
                teachers=teachers_for(rt,common,cfg,task,record_for(task)); began=time.time()
                train(rt,ar,dict(cfg,weights=dict(H=.25,U=.01)),task,ex,'C_FACT',record_for(task),teachers,w0['W0'],layer_id=30); del teachers
                bank['experts'][e]=dict(path=str(ar/'private/edits'/f'e{n:03d}'/'C_FACT/latest.pt'),layer_id=30,W0=w0['W0'],task=digest(task),code=cfg['code_commit'],training_seconds=time.time()-began); bank['inserted']=ids[:n]; save(bankfile,bank)
            emit(arm,n,record_for(task),bank,[task['native']])
            if arm in ('B1','B2') and n in slot_by_t:
                slot=slot_by_t[n]
                for j in sorted(set([slot['expert']])):
                    if j<1 or j>n: continue
                    existing_s=bank['experts'].get(ids[j-1],{})
                    if str(existing_s.get('path','')).endswith(f"t{n:03d}.pt") and Path(existing_s['path']).exists():
                        continue
                    target=tasks[j-1]; s=bank['experts'][ids[j-1]]; v=torch.load(s['path'],map_location='cpu',weights_only=True); ex.load_state_dict(v['expert']); ex.requires_grad_(True); opt=optimizer_for(ex,rt.model)
                    teacher_map={query_id(x[4]):x for x in teachers_for(rt,common,cfg,target,record_for(target))}
                    hook=__import__('methods.medtrace',fromlist=['MedTraceLayerHook']).MedTraceLayerHook(rt.get_module('model.layers.30.mlp.down_proj'),ex); hook.attach()
                    try:
                        for step in range(20):
                            role_items=[]
                            for role in ('H_fit','U_fit'):
                                qids=slot['roles'][role].get(arm,[]) if slot['roles'][role].get('active') else []
                                if qids:
                                    q=qids[step%len(qids)]; row=row_by_q[q]
                                    batch=source_batch(rt,record_for(target),row) if role=='H_fit' else None
                                    role_items.append((role,row,batch))
                            if not role_items: continue
                            h=next(((x[2],x[1]) for x in role_items if x[0]=='H_fit'),None)
                            uq=next((x[1] for x in role_items if x[0]=='U_fit'),None)
                            u=teacher_map.get(query_id(uq), next(iter(teacher_map.values()))) if teacher_map else None
                            if u is None: raise RuntimeError('missing frozen U teacher binding')
                            update(rt,hook,ex,opt,source_batch(rt,record_for(target),target['native']),rt.build_edit_batch(replace(record_for(target),question=target['fit_questions'][0])),u,h,extra_weight=.25,U_weight=.01)
                    finally: hook.detach(); ex.requires_grad_(False)
                    newpath=ar/'private'/'maintenance'/f'e{j:03d}'/f't{n:03d}.pt';newpath.parent.mkdir(parents=True,exist_ok=True);save(newpath,dict(step=20,expert=ex.state_dict(),binding=dict(task=target,arm=arm,maintenance_at=n,W0=s['W0'],code=cfg['code_commit']),optimizer=opt.state_dict())); bank['experts'][ids[j-1]]=dict(path=str(newpath),layer_id=30,W0=s['W0'],task=digest(target),code=cfg['code_commit']);save(bankfile,bank)
                emit(arm,n,record_for(task),bank,[task['native']])
        emit(arm,19,record_for(tasks[0]),bank,rows)
        write(pub/f'ARM_{arm}.json',dict(arm=arm,status='GENERATED_NOT_SCORED',N=19))
    write(pub/'GENERATED.json',dict(status='COMPLETE_NOT_SCORED',arms=['B0','B1','B2'],N=19,phase='24E'))

if __name__=='__main__': main(read(os.environ['JOB_CONFIG']))
