"""Two finite, preregistered development experiments; historical results stay intact."""
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import core
import benchmark146 as retro

common, worker, train, RUN, BASE = core.common, core.worker, core.train, core.RUN, core.BASE
ARMS = ('U_PAIR_CE', 'U_PAIR_CEU')
ROUTE_ARM = 'TT88_FITKEY5'


def selected():
    return [t for t in common.read(BASE/'private/QUEUES.json')['tasks'] if t['cohort']=='P2']


def configure():
    """Record new source identity without altering old runtime/source receipts."""
    original = worker.write
    def write(path, data):
        if 'binding' in data and 'phase' in data['binding']:
            data['research_lock'] = common.digest(common.read(RUN/'private/RESEARCH_LOCK.json'))
        original(path, data)
    worker.write = write


def u_pair():
    import torch
    slot, gpu = int(os.environ['PARTITION']), int(os.environ['GPU'])
    assert gpu==slot and slot in (0,1)
    configure();ts=selected();roles=common.read(BASE/'private/U_ROLES.json')
    with common.lease(gpu):
        runtime, bindings=common.load(gpu)
        fit=train.teachers_for(runtime,roles['FIT'])
        evaluation=train.teachers_for(runtime,roles['CAL']+roles['CHECK'])
        points={a:{} for a in ARMS}
        for t in ts[:8]:
            initial=torch.load(worker.w0_path(t,0),map_location='cpu',weights_only=True)['expert']
            fixed=worker.fixed_x(runtime,t,slot,initial,fit)
            for arm in ARMS:
                teachers=fit if arm==ARMS[1] else []
                oldarm='CE_U_MULTI' if teachers else 'CE_ONLY'
                if slot==0:
                    p=BASE/'private/edits'/t['anonymous_edit']/'s0'/oldarm/'step160.pt'
                    saved=torch.load(p,map_location='cpu',weights_only=True)
                    b=saved['binding'];assert b['steps']==160 and b['task']['seed']==t['seed']
                    assert b['task']['native']==t['native'] and b['task']['fit_questions']==t['fit_questions']
                    assert b['task']['U_fit']==([x[4] for x in teachers])
                    assert b['W0']==common.state_hash(train.clone(initial,t['seed'],runtime.device))
                else:
                    directory=RUN/'private/research/u'/t['anonymous_edit']/('s'+str(slot))/arm
                    task=worker.train_task(t,slot,[x[4] for x in teachers])
                    task['bindings']['research_lock']=common.digest(common.read(RUN/'private/RESEARCH_LOCK.json'))
                    train.continuation(runtime,task,common.record(t),initial,directory,teachers,160,[160],fixed)
                    p=directory/'step160.pt'
                points[arm][t['edit_id']]=p
                worker.evaluate(runtime,bindings,t,slot,arm,160,{t['edit_id']:p},worker.router(runtime,t),[t],teacher_rows=evaluation)
                worker.evaluate(runtime,bindings,t,slot,arm,160,{t['edit_id']:p},worker.router(runtime,t),[dict(t,events=[])],forced=True,teacher_rows=evaluation)
            print('U_PAIR_SINGLE_COMPLETE',t['order'],slot,flush=True)
        for prefix in (8,24):
            bank=[]
            for t in ts[:prefix]:bank+=worker.router(runtime,t)
            for arm in ARMS:
                ps={t['edit_id']:points[arm][t['edit_id']] if i<8 else worker.w0_path(t,0) for i,t in enumerate(ts[:prefix])}
                worker.evaluate(runtime,bindings,ts[prefix-1],slot,arm,160,ps,bank,ts[:prefix],mode='bank',prefix=prefix,teacher_rows=evaluation)
        common.write(RUN/'private'/('RESEARCH_U_DONE_'+str(slot)+'.json'),dict(status='GENERATED_NOT_SCORED',epoch=time.time()))


def choose(distances, radii):
    """Keep the frozen nearest-key then radius rule, using five fit-only keys."""
    i=int(distances.argmin().item())
    return i, bool(distances[i]<=radii[i])


def route_bank():
    import torch
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter, RouteDecision, distances
    part,gpu=int(os.environ['PARTITION']),int(os.environ['GPU'])
    assert gpu==2+part and part in (0,1)
    configure();ts=retro.queue()['tasks'];folder=RUN/'private/research/keys';folder.mkdir(parents=True,exist_ok=True)
    with common.lease(gpu):
        runtime,bindings=common.load(gpu)
        editor=BalanceEditPaperSpecEditor(runtime)
        try:
            for t in ts[part::2]:
                p=folder/(t['anonymous_edit']+'.pt')
                binding=dict(edit=t['edit_id'],native=t['native'],fit=t['fit_questions'],rule='native plus four training paraphrases',lock=common.digest(common.read(RUN/'private/RESEARCH_LOCK.json')))
                if p.exists():assert torch.load(p,map_location='cpu',weights_only=True)['binding']==binding;continue
                keys=[]
                with torch.no_grad():
                    for question in t['fit_questions']:
                        query=replace(common.record(t),question=question,target='',official_rephrase='')
                        keys.append(editor._question_key(query).detach().cpu())
                common.save(p,dict(binding=binding,keys=keys))
            common.write(RUN/'private'/('RESEARCH_KEYS_DONE_'+str(part)+'.json'),dict(status='COMPLETE',epoch=time.time()))
        finally:
            target,base=editor.target,editor.wrapper.base;editor.reset_editor_state();runtime.replace_module(target,base)
        while not all((RUN/'private'/('RESEARCH_KEYS_DONE_'+str(i)+'.json')).exists() for i in (0,1)):
            common.budget()
            assert not list((RUN/'private').glob('FAILURE_research_route_*.json'))
            time.sleep(5)
        extras={t['edit_id']:torch.load(folder/(t['anonymous_edit']+'.pt'),map_location='cpu',weights_only=True)['keys'] for t in ts}
        def route(router,query):
            if not router.keys:return RouteDecision(None,None,None,None,False,router.distance)
            query=query.detach().float().reshape(-1)
            keys=torch.stack([k for eid,key in zip(router.logical_ids,router.keys) for k in [key.cpu()]+extras[eid]]).to(query.device)
            ds=distances(keys,query,router.distance).reshape(len(router.logical_ids),5).min(dim=1).values
            i,on=choose(ds,torch.tensor(router.radii,device=ds.device))
            eid=router.logical_ids[i]
            return RouteDecision(eid if on else None,eid,float(ds[i]),router.radii[i],on,router.distance)
        MemoryRouter.route=route
        bank=[];points={}
        for t in ts:bank+=worker.router(runtime,t);points[t['edit_id']]=retro.point(t)
        # Both workers retain all native identities; only query IDs are partitioned.
        worker.evaluate(runtime,bindings,ts[-1],0,ROUTE_ARM,0,points,bank,retro.shard_tasks(ts,part,count=2),mode='bank',prefix=146)
        common.write(RUN/'private'/('RESEARCH_ROUTE_DONE_'+str(part)+'.json'),dict(status='GENERATED_NOT_SCORED',epoch=time.time()))


def controller():
    import pipeline
    common.write(RUN/'public/RESEARCH_PROGRESS.json',dict(status='TWO_TRACKS_RUNNING',complete=False))
    jobs=[pipeline.launch('research.py','research_u',i,i) for i in (0,1)]
    jobs += [pipeline.launch('research.py','research_route',2+i,i) for i in (0,1)]
    pipeline.wait(jobs)
    common.write(RUN/'private/RESEARCH_GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',epoch=time.time()))
    pipeline.wait([pipeline.launch('research_queue.py','research_ingest')])
    # Only newly generated paired continuation weights; historical W0 and fits are preserved.
    removed=[]
    for p in (RUN/'private/research/u').glob('*/s1/*/step160.pt'):
        assert not p.is_symlink() and p.resolve().is_relative_to((RUN/'private/research/u').resolve())
        removed.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
    common.write(RUN/'private/RESEARCH_DELETION.json',dict(files=removed,all_registered_GPU_consumers_complete=True,bindings_durable=True,rebuild='retrain',epoch=time.time()))
    common.write(RUN/'public/RESEARCH_PROGRESS.json',dict(status='ASTRA_SCORING',complete=False))
    root=RUN/'private/judge_research_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        common.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(30)
    pipeline.wait([pipeline.launch('research_report.py','research_report')])


def selfcheck():
    import torch
    # New fit key changes the selected expert; radius still rejects outside all keys.
    assert choose(torch.tensor([2.,1.]),torch.tensor([3.,3.]))==(1,True)
    assert choose(torch.tensor([.5,1.]),torch.tensor([3.,3.]))==(0,True)
    assert choose(torch.tensor([4.,5.]),torch.tensor([3.,3.]))==(0,False)
    assert choose(torch.tensor([1.,1.]),torch.tensor([1.,1.]))==(0,True)
    retro.shard_test()
    ts=retro.queue()['tasks'];parts=[retro.query_ids(retro.shard_tasks(ts,i,count=2)) for i in (0,1)]
    assert not parts[0]&parts[1] and len(parts[0]|parts[1])==1509
    assert len(selected())==24
    common.write(RUN/'public/RESEARCH_CPU_TEST.json',dict(status='PASS',tests=['fit nearest expert','radius rejection','stable ties','exhaustive disjoint query sharding'],final_route_queries=1509,U_edit_pairs=16))


if __name__=='__main__':
    try:
        {'research_u':u_pair,'research_route':route_bank,'research_controller':controller,'research_test':selfcheck}[os.environ['ACTION']]()
    except BaseException as error:
        common.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False));raise
