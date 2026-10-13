"""Freeze actual single-expert consumers and reconstruct unchanged MARGIN_002 routing."""
import os
import sqlite3
from pathlib import Path
import torch
import scoped as exp
import structure as s
from m3bench_repro.editors.routing import MemoryRouter,decision_as_json,euclidean_distances

p,c,RUN=exp.p,exp.c,exp.RUN


def feature(row):return RUN/'private/route_features'/(c.digest([row['image_sha256'],row['question']])+'.pt')


def prepare_rows():
    db=sqlite3.connect('file:'+str(exp.SOURCE/'private/judge_replay_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
    index={arm:{} for arm in ('W0','CE192','BASE')}
    for arm,qid,path in db.execute("SELECT method,query_id,path FROM consumer WHERE method IN ('W0','CE192','BASE')"):
        index[arm][qid]=path
    c.write(RUN/'private/HISTORICAL_OUTPUTS.json',index);db.close()
    baseline=c.read(exp.SOURCE/'private/BASELINE.json');main=p.queries(146)
    check=c.read(exp.SOURCE/'private/REPLAY_CHECK.json')
    for row in check:
        old=c.read(exp.historical('W0',row['query_id']))
        row['image_sha256']=old['binding']['judge_input']['image_sha256']
    rows={str(feature(row)):row for row in list(main.values())+check}
    for t in p.tasks():
        for row in p.train_rows(t):rows[str(feature(row))]=row
    c.write(RUN/'private/FEATURE_ROWS.json',list(rows.values()))
    c.write(RUN/'private/SOURCE_CHECK_ROWS.json',check)
    c.write(RUN/'private/MAIN_BASELINE.json',baseline)
    c.write(RUN/'private/REFERENCE_WEIGHTS.json',c.read(exp.SOURCE/'private/REFERENCE_WEIGHTS.json'))


def cache():
    from llava.constants import IMAGE_TOKEN_INDEX
    part=int(os.environ['PARTITION']);gpu=exp.GPUS[part]
    rows=c.read(RUN/'private/FEATURE_ROWS.json')
    with p.lease(gpu):
        runtime,_=c.load(gpu);layer=runtime.target_lock['balancedit']['targets'][0]
        for i,row in enumerate(rows[part::2]):
            if feature(row).exists():continue
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            assert raw['image_sha256']==row['image_sha256']
            with torch.inference_mode():
                embeds,att,pos,_=runtime._expand_multimodal(raw_input_ids=raw['input_ids'],attention_mask=raw['attention_mask'],labels=torch.full_like(raw['input_ids'],-100),images=raw['images'])
                masks=s.token_masks(raw['input_ids'][0],embeds.shape[1],None if att is None else att[0],IMAGE_TOKEN_INDEX)
                captures=[];handle=runtime.get_module(layer).register_forward_pre_hook(lambda _,args:captures.append(s.pool(args[0][0],masks)))
                try:runtime.model(inputs_embeds=embeds,attention_mask=att,position_ids=pos,labels=None,use_cache=False,return_dict=True)
                finally:handle.remove()
            vec,counts=captures[0]
            c.save(feature(row),dict(route_vectors=vec.cpu(),counts=counts,binding=[row['image_sha256'],row['question']],answer_used=False))
            if i%100==0:print('ROUTE_FEATURE',part,i,flush=True)
    p.done('CACHE_'+str(part))


def routes():
    ts=p.tasks();entries=[]
    for t in ts:entries+=p.load_state(p.initial(t).parent.parent/'ROUTER.pt')['entries']
    router=MemoryRouter.from_state(dict(distance='euclidean',entries=entries),device='cpu')
    keys=[torch.stack([p.load_state(feature(row))['route_vectors'] for row in p.train_rows(t)]) for t in ts]
    baseline=c.read(RUN/'private/MAIN_BASELINE.json');result={};mismatches=[]
    rows=list(p.queries(146).values())+c.read(RUN/'private/SOURCE_CHECK_ROWS.json')
    for row in rows:
        query=p.load_state(feature(row))['route_vectors'];decision=decision_as_json(router.route(query[0]))
        selected=None;candidate=None;advantage=None
        if decision['activated']:
            nearest=router.logical_ids.index(decision['logical_edit_id'])
            distances=euclidean_distances(router._key_matrix(),query[0])
            candidates=[i for i,(v,radius) in enumerate(zip(distances,router.radii)) if float(v)<=radius]
            assert nearest in candidates
            scores=[float(s.key_score(query,k,True).max()) for k in keys]
            candidate=max(candidates,key=lambda i:(scores[i],-i));advantage=scores[candidate]-scores[nearest]
            selected=router.logical_ids[candidate if advantage>=.02 else nearest]
        if row['query_id'] in baseline:
            old=c.read(baseline[row['query_id']]['source_path'])
            if selected!=old['effective_expert']:mismatches.append(row['query_id'])
        result[row['query_id']]=dict(effective_expert=selected,original=decision,candidate=candidate,advantage=advantage)
    c.write(RUN/'private/ROUTES.json',result)
    c.write(RUN/'public/ROUTE_PARITY.json',dict(status='PASS' if not mismatches else 'BLOCKED',
        original_inputs=1513,mismatches=len(mismatches),new_CHECK_inputs=96,threshold=.02,answers_used=False))
    assert not mismatches,'Original MARGIN_002 reconstruction failed; do not substitute forced routes'
    return result


def plan():
    assert all((RUN/'private'/f'MECHANICAL_{part}.json').exists() for part in range(2))
    route=routes();ts=p.tasks();chosen={t['edit_id'] for t in exp.selected()}
    main=p.queries(146);check=c.read(RUN/'private/SOURCE_CHECK_ROWS.json')
    baseline=c.read(RUN/'private/MAIN_BASELINE.json')
    primary={'W0':{},'CE192':{}}
    for qid in main:
        primary['W0'][qid]=str(exp.historical('W0',qid))
        primary['CE192'][qid]=str(exp.historical('CE192',qid))
        for arm in primary:
            d=c.read(primary[arm][qid]);assert d['effective_expert']==route[qid]['effective_expert']
            assert d['binding']['judge_input']==c.read(primary['W0'][qid])['binding']['judge_input']
    consumers=[];jobs={}
    def add(arm,mode,row,expert,old=None,produced_arm=None,node=192,variant=None):
        producer=produced_arm or arm;path=str(old or exp.output(producer,row['query_id']))
        consumers.append(dict(arm=arm,mode=mode,query_id=row['query_id'],path=path,produced_arm=producer))
        if old is None:
            key=producer,row['query_id']
            job=dict(arm=producer,mode=mode,row=row,expert=expert,node=node,variant=variant,path=path)
            if key in jobs:assert jobs[key]==job
            else:jobs[key]=job
    for qid,row in main.items():
        expert=route[qid]['effective_expert'];affected=expert in chosen
        add('W0','bank_R0',row,expert,primary['W0'][qid])
        add('CE192','bank_R0',row,expert,primary['CE192' if affected else 'W0'][qid])
        for arm in exp.ARMS:
            for step in exp.NODES:
                if step!=192 and not affected:continue
                add(exp.label(arm,step),'bank_R0',row,expert,None if affected else primary['W0'][qid],node=step)
        if affected:
            for variant in ('local','shared'):add('E_'+variant,'bank_R0',row,expert,variant=variant)
    for row in check:
        qid=row['query_id'];expert=ts[row['forced_expert_index']]['edit_id'];affected=expert in chosen
        paths={arm:exp.historical(arm,qid) for arm in ('BASE','W0','CE192')}
        for arm in paths:add(arm,'forced_source_CHECK',row,expert if arm!='BASE' else None,paths[arm if arm!='CE192' or affected else 'W0'])
        for arm in exp.ARMS:
            for step in exp.NODES:
                if step!=192 and not affected:continue
                add(exp.label(arm,step),'forced_source_CHECK',row,expert,None if affected else paths['W0'],node=step)
        if affected:
            for variant in ('local','shared'):add('E_'+variant,'forced_source_CHECK',row,expert,variant=variant)
        natural=route[qid]['effective_expert']
        add('W0_NAT','natural_source_CHECK',row,natural)
        add('BASE_NAT','natural_source_CHECK',row,None,paths['BASE'])
        for arm in exp.ARMS:
            final=exp.label(arm,192)+'_NAT'
            if natural in chosen:add(final,'natural_source_CHECK',row,natural,node=192)
            else:add(final,'natural_source_CHECK',row,natural,produced_arm='W0_NAT')
    lock=dict(experts=8,selected_edits=[t['edit_id'] for t in exp.selected()],arms=list(exp.ARMS),nodes=list(exp.NODES),
        formal_trajectories=40,formal_updates=7680,mechanical_updates=8,GPUs=list(exp.GPUS),
        affected_main=sum(route[q]['effective_expert'] in chosen for q in main),affected_forced_CHECK=24,
        affected_natural_CHECK=sum(route[x['query_id']]['effective_expert'] in chosen for x in check),
        new_generations=len(jobs),new_Judge_upper_bound=len(jobs),consumers=len(consumers),
        replay=False,final_checkpoint_retention=40,intermediate_nodes_not_selected=True,
        natural_CE_reference='UNAVAILABLE_IF_CHANGED_EXPERT: historical weights deleted; no retraining',
        scope='eight-expert local intervention in frozen146 bank',training_roles=['native','FIT'])
    assert lock['affected_main']==84
    c.write(RUN/'private/SCOPED_LOCK.json',lock);c.write(RUN/'private/CONSUMERS.json',consumers)
    c.write(RUN/'private/JOBS.json',list(jobs.values()))
    c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**{k:v for k,v in lock.items() if k!='selected_edits'}))
    c.write(RUN/'public/NATIVE_PARITY_AUDIT.json',dict(status='PASS',parts=[c.read(RUN/'public'/f'NATIVE_PARITY_{i}.json') for i in range(2)]))
    p.done('PLAN_COMPLETE')


if __name__=='__main__':
    {'rows':prepare_rows,'scoped_cache':cache,'plan':plan}[os.environ['ACTION']]()
