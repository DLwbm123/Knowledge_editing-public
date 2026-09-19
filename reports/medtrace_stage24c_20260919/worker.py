"""Pure19 R_H with frozen E0/E2 controls, real natural routing and79-query DEV."""
import os,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
from scripts.medtrace.stage19_fasttrack_budget import read,write,check
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage19_fasttrack import load,record_for,query_id
from scripts.medtrace.stage20_closeout import outputs


def run(cfg):
    from scripts.medtrace.stage17_single import setup
    setup(cfg)
    import torch,shutil,importlib.util
    torch.use_deterministic_algorithms(True)
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace.stage18_cfact import state_hash,assert_base_off,teachers_for
    from scripts.medtrace import stage15
    from scripts.medtrace.run_selective_write import save
    from training import train
    spec=importlib.util.spec_from_file_location('emit24',ROOT/'reports/medtrace_stage22_20260919/continuation_runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);emit=m.emit
    root=Path(cfg['run']);p=root/'private';pub=root/'public';common=Path(cfg['common_run']);reg=read(pub/'PREREGISTRATION.json');assert reg['binding']==cfg['registration_binding']
    accepted=read(p/'MECHANICAL_ACCEPTANCE.json');assert accepted['status']=='PASS' and accepted['scale']==16777216 and accepted['cache_20step_exact']
    stream=read(common/'private/STREAM.json');assert digest(stream)==cfg['stream_binding'];tasks=stream['tasks'][:19];ids=[t['canonical_edit_id'] for t in tasks]
    dev=read(p/'DEV_UNION.json');assert dev['binding']==reg['DEV_binding'] and len(dev['rows'])==79
    baseline={r['query_id']:r['output'] for r in read(common/'private/HISTORICAL_BASE.json')['records']}
    # DEV66 includes4 positive rows; merge the exact same frozen Base outputs.
    baseline.update({r['query_id']:r['output'] for r in read(common/'private/DEV_BASE.json')['records']})
    assert {query_id(r) for r in dev['rows']}<=set(baseline)
    rt=load(cfg);frozen=[(x,x._version,x.data_ptr()) for x in rt.model.parameters()]
    def guard():
        check(cfg);assert_base_off(rt)
        if (root/'STOP').exists():raise RuntimeError('User stop marker')
        if shutil.disk_usage(root).free<8*1024**3:raise OSError('8GiB storage reserve reached')
        assert all(x._version==v and x.data_ptr()==ptr and not x.requires_grad for x,v,ptr in frozen)
    prior=torch.load(Path(cfg['stage20_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False);keys={};env=[]
    for t in tasks:
        guard();record=record_for(t)
        with torch.inference_mode():key=rt.extract_layer_input_key(rt.build_question_batch(record),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
        assert torch.equal(key,prior['routes'][t['canonical_edit_id']]['key']);keys[query_id(t['native'])]=key
        if t['order'] in (1,8,19):
            raw,_,b=stage15.prepared(rt,t['native'],record);o=stage15.generate(rt,raw,b);assert o['binding']==baseline[query_id(t['native'])]['binding'] and o['raw_token_ids']==baseline[query_id(t['native'])]['raw_token_ids'];env.append(t['order'])
    write(pub/'ENVIRONMENT_ACCEPTANCE.json',dict(Base_OFF_positions=env,exact_raw_tokens=True,exact_route_keys=19,device=cfg['gpu_uuid'],deterministic_backend=True))
    ex=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(rt.device).requires_grad_(False)
    routers={n:MemoryRouter.from_state(dict(distance='euclidean',entries=[prior['routes'][e] for e in ids[:n]]),device=rt.device) for n in range(1,20)}
    done={(r['arm'],r['mode'],r['prefix'],r['query_id']) for r in outputs(root)};record=record_for(tasks[0])
    for arm in ('E0','E2'):
        bank=torch.load(common/'private'/f'BANKS_{arm}.pt',map_location='cpu',weights_only=False);assert bank['inserted']==ids and bank['weights']==dict(H=0. if arm=='E0' else .25,U=.01)
        for t in tasks:
            s=bank['experts'][t['canonical_edit_id']];point=torch.load(s['path'],map_location='cpu',weights_only=True);w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f"e{t['order']:03d}.pt",map_location='cpu',weights_only=True)
            assert point['step']==320 and point['binding']['W0']==w0['W0'] and digest(point['binding']['task'])==digest(t) and s['layer_id']==30;del point,w0
        emit(rt,cfg,root,ex,bank,dev['rows'],baseline,routers[19],keys,arm,19,'DEV79',record,guard,done)
        write(pub/f'ARM_{arm}.json',dict(arm=arm,queries=79,N=19,reused_experts=19,status='GENERATED_NOT_SCORED'))
    bankpath=p/'BANKS_R_H.pt';bank=torch.load(bankpath,map_location='cpu',weights_only=False) if bankpath.exists() else dict(arm='R_H',weights=dict(H=.25,U=.01),experts={},routes={e:prior['routes'][e] for e in ids},inserted=[],code=cfg['code_commit'],registration=reg['binding'])
    assert bank['code']==cfg['code_commit'] and bank['registration']==reg['binding']
    for t in tasks:
        guard();n=t['order'];e=t['canonical_edit_id']
        if e in bank['experts']:continue
        w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True);assert w0['task']==digest(t) and w0['origin_layer']==30
        ex.load_state_dict(w0['expert']);assert state_hash(ex)==w0['W0'];teachers=teachers_for(rt,common,cfg,t,record_for(t));began=time.time()
        train(rt,root,dict(cfg,weights=bank['weights']),t,ex,'C_FACT',record_for(t),teachers,w0['W0'],layer_id=30);del teachers
        bank['experts'][e]=dict(kind='C_FACT',path=str(p/'edits'/f'e{n:03d}'/'C_FACT/latest.pt'),layer_id=30,W0=w0['W0'],task=digest(t),code=cfg['code_commit'],training_seconds=time.time()-began,reused=False)
        bank['inserted']=ids[:n];save(bankpath,bank)
        emit(rt,cfg,root,ex,bank,[t['native']],baseline,routers[n],keys,'R_H',n,'insertion',record,guard,done)
        write(pub/'PROGRESS.json',dict(phase='PURE19_HSIC',completed=n,total=19));gc.collect();torch.cuda.empty_cache()
    emit(rt,cfg,root,ex,bank,dev['rows'],baseline,routers[19],keys,'R_H',19,'DEV79',record,guard,done)
    write(pub/'ARM_R_H.json',dict(arm='R_H',queries=79,insertions=19,N=19,status='GENERATED_NOT_SCORED'))
    profiles=[]
    for t in tasks:
        tr=read(p/'edits'/f"e{t['order']:03d}"/'C_FACT/TRAINING.json');sparse=[dict(step=x['step'],regularization={k:v for k,v in x['regularization'].items() if k!='masks'},gradient_norm=x['gradient_norm']) for x in tr['curve'] if x['step'] in (1,80,160,320)]
        profiles.append(dict(position=t['order'],steps=tr['steps'],session_seconds=tr['session_seconds'],tokens=tr['tokens'],extra_gradient_nonzero_steps=tr['extra_gradient_nonzero_steps'],diagnostics=sparse))
    write(pub/'RESOURCE_PROFILE.json',dict(writers=profiles,peak_allocated=torch.cuda.max_memory_allocated(),numeric_version='R2_FIXED_2POW24'))
    write(pub/'GENERATED.json',dict(status='COMPLETE_NOT_SCORED',N=19,arms=['E0','E2','R_H'],queries=256,peak_allocated=torch.cuda.max_memory_allocated(),no_auto_stage25=True))

if __name__=='__main__':run(read(os.environ['JOB_CONFIG']))
