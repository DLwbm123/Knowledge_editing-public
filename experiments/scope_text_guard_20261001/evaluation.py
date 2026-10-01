"""Frozen effective-expert generation cache, with verified historical R0 reuse."""
import hashlib,copy,fcntl
from dataclasses import asdict
from pathlib import Path
from resources import ROOT,read,write,check
from judge_protocol import digest
def router_entries(router):
    return [dict(edit=e,radius=r,key_sha256=hashlib.sha256(k.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()) for e,k,r in zip(router.logical_ids,router.keys,router.radii,strict=True)]
def effective_binding(ib,protocol,expert):
    assert ib['generation']['do_sample'] is False and ib['generation']['num_beams']==1
    return dict(input=ib,**protocol,expert_sha256=expert,hook='complete unscaled frozen LR4 residual once' if expert else 'OFF Base',generation_deterministic=True)
def historical_cohort(ref):
    cohort=ref.get('cohort') or ('DEV' if ref['order']<=24 else 'REG')
    assert cohort in ['DEV','REG'],'Unknown historical cohort'
    return cohort
def validate_historical(ref,ib,protocol,features,weights):
    old=Path(read(ROOT/'PREDECESSOR.json')['root']);saved=read(old/'private/generations'/f'{ref["execution_id"]}.json');b=saved['execution_binding']
    assert b['input']==ib and all(b[k]==v for k,v in protocol.items()),'Historical input/model/backend differs'
    assert b['science_id']==read(old/'SCIENCE_LOCK.json')['id'] and not b['forced_diagnostic']
    cohort=historical_cohort(ref)
    entries=features['banks'][cohort]
    if ref['mode']=='single':entries=[next(e for e in entries if e['edit']==ref['edit'])]
    else:entries=entries[:ref['prefix']]
    from routers import ScopeRouter
    original=ScopeRouter(entries,'R0')
    assert b['prefix']==ref['prefix'] and b['router']==dict(kappa=1.,mu=0.,entries=router_entries(original))
    assert [v['sha256'] for v in b['bank']]==[weights[str(e['order'])]['sha256'] for e in entries]
    assert all(v['actual_steps']==80 and v['origin']=='INIT_POST80' for v in b['bank'])
    assert saved['output']['raw_token_ids']==ref['output']['raw_token_ids'] and saved['output']['binding']==ib
    return saved
def generate_direct(runtime,raw,ib,expert):
    from scripts.medtrace import stage15
    from methods.medtrace import MedTraceLayerHook
    from freshstart import runtime as rt
    hook=MedTraceLayerHook(runtime.get_module(rt.LAYER),expert) if expert is not None else None
    if hook:hook.attach()
    try:return stage15.generate(runtime,raw,ib,hook)
    finally:
        if hook:hook.detach()
def evaluate(runtime,tasks,bank,router,label,mode,prefix,folder,protocol,refs,features,weights):
    import worker_v3 as old
    from scripts.medtrace import stage15
    from judge_protocol import request
    index={(r['mode'],r['prefix'],r['edit'],r['task'],r['input_id']):r for r in refs};byedit={e['edit']:weights[str(e['order'])]['sha256'] for e in features['entries']};consumers=[]
    count=dict(historical_R0_reuse=0,effective_route_reuse=0,Base_OFF_reuse=0,new_generation=0)
    full_router=dict(method=label,entries=router_entries(router),tau=0 if label in ['NEG0','TXT'] else None,prototype_lock=read(ROOT/'SCIENCE_LOCK.json')['id'])
    for t in tasks:
        for row in t['evaluation']:
            check();raw,_,ib=stage15.prepared(runtime,row,old.record(t));q=old.key(runtime,row,old.record(t));decision,extra=router.diagnostic(q,question=row['question']);route=asdict(decision)
            selected=route['logical_edit_id'];expert=bank.get(selected);eb=effective_binding(ib,protocol,byedit[selected] if selected else None);effective=digest(eb);memo=ROOT/'private/effective'/f'{effective}.json'
            ref=index.get((mode,prefix,t['canonical_edit_id'],row['task'],old.input_id(row)));historical=None
            if ref:
                historical=validate_historical(ref,ib,protocol,features,weights)
                if label=='R0':assert route==ref['route'],'R0 historical route differs'
                assert selected in [None,ref['route']['logical_edit_id']],'Frozen variants may veto, never reroute or fall back'
            lockpath=ROOT/'private/effective_locks'/f'{effective}.lock'
            lockpath.parent.mkdir(parents=True,exist_ok=True)
            with lockpath.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if historical and selected==ref['route']['logical_edit_id']:
                    out=historical['output'];count['historical_R0_reuse']+=1
                    if memo.exists():assert read(memo)['output']['raw_token_ids']==out['raw_token_ids']
                    else:write(memo,dict(binding=eb,output=out,source='verified historical full execution',execution_id=ref['execution_id']))
                elif memo.exists():
                    saved=read(memo);assert saved['binding']==eb;out=saved['output'];count['effective_route_reuse']+=1
                elif selected is None:
                    out,_=old.base(runtime,row,old.record(t),score=False);assert out['binding']==ib
                    write(memo,dict(binding=eb,output=out,source='verified Base OFF'));count['Base_OFF_reuse']+=1
                else:
                    out=generate_direct(runtime,raw,ib,expert);write(memo,dict(binding=eb,output=out,source='new deterministic generation'));count['new_generation']+=1
            logical=dict(effective_binding=eb,router_binding=full_router,science_id=read(ROOT/'SCIENCE_LOCK.json')['id'],prefix=prefix)
            ident=digest(logical);write(ROOT/'private/generations'/f'{ident}.json',dict(execution_binding=logical,output=out,reuse_historical_execution=ref['execution_id'] if historical and selected==ref['route']['logical_edit_id'] else None))
            base,bj=old.base(runtime,row,old.record(t));assert base['binding']==ib
            jk=request(row,out)
            if ref:assert jk in [ref['judge_key'],ref['base_judge_key']],'Veto-only routing must retain historical H/Base payload binding'
            consumers.append(dict(arm=label,seed=20260929,mode=mode,prefix=prefix,edit=t['canonical_edit_id'],order=t['order'],task=row['task'],query_id=row['query_id'],source_group=row['source_group'],input_id=old.input_id(row),execution_id=ident,judge_key=jk,base_judge_key=bj,route=route,route_diagnostics=extra,output=out,exact_Base_token_consistency=out['raw_token_ids']==base['raw_token_ids'],frozen_base_correct=ref.get('frozen_base_correct') if ref else None))
    write(folder/'CONSUMERS.json',consumers);write(folder/'GENERATION_REUSE.json',count);return consumers
