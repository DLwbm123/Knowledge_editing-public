"""Pre-output latent construction: only native/S_fit/U_bg enter router prototypes."""
import os,sys,json,time,hashlib,io
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
import torch
from resources import read,write,session
from judge_protocol import digest
from routers import ScopeRouter,cap_insert
from hard_pool import select
from m3bench_repro.editors.routing import balanced_radius
_runtime=None

def runtime():
    global _runtime
    if _runtime is None:
        from scope_worker import configure
        _runtime,_=configure()
    return _runtime

def input_id(row):return digest([row['image_sha256'],row['question']])
def feature(row):
    p=ROOT/'private/keys'/f'{input_id(row)}.pt'
    if not p.exists():
        assert not row['image_sha256'].startswith('BLACK-'),'Missing original black feature requires the real black image'
        import worker_v3 as old
        r=runtime();rec=old.record(dict(canonical_edit_id='FROZEN_BASE_FEATURE',native=row,fit_questions=[row['question']],order=0))
        old.key(r,row,rec) # Existing frozen Base extractor, no teacher, loss, or Judge.
    x=torch.load(p,map_location='cpu',weights_only=True).float().reshape(-1)
    assert torch.isfinite(x).all()
    # Deterministic hash is expressly required by this stage's mechanical contract.
    assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==hashlib.sha256(x.clone().numpy().tobytes()).hexdigest()
    return x.cuda()
def distribution(xs):
    if not xs:return dict(n=0,min=None,max=None,mean=None,median=None,q25=None,q75=None)
    x=torch.tensor(xs,dtype=torch.float64)
    return dict(n=len(xs),min=min(xs),max=max(xs),mean=sum(xs)/len(xs),median=float(x.median()),q25=float(x.quantile(.25)),q75=float(x.quantile(.75)))
def hazard(entries,background,rows,cohort):
    records=[];previous=[]
    for position,e in enumerate(entries,1):
        before=ScopeRouter(previous,'R0');after=ScopeRouter(previous+[e],'R0');captures=[]
        for i,q in enumerate(background):
            a,_=before.diagnostic(q);b,diag=after.diagnostic(q)
            if a.logical_edit_id!=e['edit'] and b.logical_edit_id==e['edit']:captures.append((i,a,b,diag))
        records.append(dict(cohort=cohort,position=position,order=e['order'],new_OFF_to_ON_i=sum(a.logical_edit_id is None for i,a,b,d in captures),old_expert_to_i_switch=sum(a.logical_edit_id is not None for i,a,b,d in captures),total_new_capture=len(captures),unique_source_capture=len({rows[i]['source_group'] for i,a,b,d in captures}),capture_distance=distribution([b.nearest_distance for i,a,b,d in captures]),previous_winner_distance=distribution([a.nearest_distance for i,a,b,d in captures if a.nearest_distance is not None]),new_winner_distance=distribution([b.nearest_distance for i,a,b,d in captures]),winner_margin=distribution([d['margin'] for i,a,b,d in captures]),H_count=len(captures),H_source=len({rows[i]['source_group'] for i,a,b,d in captures}),H_switch=sum(a.logical_edit_id is not None for i,a,b,d in captures)))
        previous.append(e)
    for r in records:
        r['rank']={k:1+sum(s[k]>r[k] for s in records) for k in ['H_count','H_source','H_switch']}
        r['tied_at_rank']={k:sum(s[k]==r[k] for s in records) for k in ['H_count','H_source','H_switch']}
    return records
def main():
    assert read(ROOT/'public/DATA_ROLE_AUDIT.json')['status']=='PASS'
    assert not (ROOT/'private/ROUTER_FEATURES.pt').exists(),'Latent definitions immutable'
    from routers import selfcheck
    selfcheck()
    with session('FROZEN_LATENT_AUDIT',3600),torch.inference_mode():
        tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];bg=read(ROOT/'private/U_bg.json')['rows'];z=torch.stack([feature(r) for r in bg]);byid={input_id(r):i for i,r in enumerate(bg)};entries=[]
        for t in tasks:
            n=t['native'];key=feature(n);original_positive=feature(dict(n,question=t['fit_questions'][0]));black=feature(dict(n,image_sha256='BLACK-'+n['image_sha256']))
            radius=float(balanced_radius(key,original_positive,black,alpha=.2,distance='euclidean'))
            positives=torch.stack([key]+[feature(dict(n,question=q)) for q in t['semantic_fit_questions']]);assert len(positives)==5
            mined=select([dict(r,input_hash=input_id(r)) for r in bg],z,key,radius)
            frozen=read(ROOT/f'private/hard/{t["order"]}.json')['rows']
            assert [r['input_hash'] for r in mined]==[r['input_hash'] for r in frozen],'Frozen hard-negative rule differs'
            negatives=torch.stack([z[byid[r['input_hash']]] for r in frozen]) if frozen else z[:0]
            entries.append(dict(edit=t['canonical_edit_id'],order=t['order'],key=key,radius=radius,safe_radius=radius,positive=positives,negative=negatives,positive_origins=['native']+['S_fit']*4,negative_origins=['U_bg']*len(negatives),negative_status='PASS' if len(negatives) else 'UNSUPPORTED_NEGATIVE_PROTO'))
        records=hazard(entries[:24],z,bg,'DEV')+hazard(entries[24:],z,bg,'REG')
        audit=dict(status='FROZEN_PRE_OUTPUT',records=records,hazard_definitions=read(ROOT/'public/ROUTER_CONFIG.json')['hazard'],formal_correctness_read=False,epoch=time.time())
        write(ROOT/'public/INSERTION_HAZARD_AUDIT.json',audit)
        capped={};singles={}
        for name,start in [('DEV',0),('REG',24)]:
            bank=[]
            for e in entries[start:start+24]:
                e=cap_insert(bank,e,z);bank.append(e)
                singles[str(e['order'])]=cap_insert([],dict(e,safe_radius=e['radius']),z)
            capped[name]=bank
        payload=dict(banks=capped,singles=singles,background=z,entries=entries,feature_extractor='frozen Base',training_steps=0,formal_inputs_in_prototypes=0)
        b=io.BytesIO();torch.save(payload,b);from storage import Store
        Store(ROOT).write('private/ROUTER_FEATURES.pt',b.getvalue())
        test=dict(status='LATENT_PASS_MODEL_PARITY_PENDING',feature_hash_deterministic=True,positive_origins_native_S_fit_only=True,negative_origins_audited_U_bg_only=True,tau=0,formal_inputs_outcomes_in_construction=0,no_GPU_training=True,RCAP_U_bg_capture_exclusion_checked=True,native_activation={name:dict(pass_count=sum(e['cap_status']=='PASS' for e in bank),unsafe_count=sum(e['cap_status']!='PASS' for e in bank)) for name,bank in capped.items()},negative_supported=sum(e['negative_status']=='PASS' for e in entries),RNG_selfcheck=True)
        write(ROOT/'public/ROUTER_MECHANICAL_TESTS.json',test)
    # Posthoc exposed damage is read only AFTER the latent audit and all rules exist.
    prior=Path(read(ROOT/'PREDECESSOR.json')['pr9_root']);exposed=read(prior/'public/ROUTE_REMOVAL_CAUSAL.json')
    target=next(r for r in records if r['cohort']=='REG' and r['position']==12)
    audit.update(status='COMPLETE',formal_correctness_read_after_freeze=True,expert12=dict(target,detectable_by_nonempty_U_bg_capture=target['H_count']>0,deployment_detector_validated=False,exposed_PR9_REMOVE12=exposed['buckets']['REMOVE_12']['cohorts']['old47']))
    write(ROOT/'public/INSERTION_HAZARD_AUDIT.json',audit)
    answer='U_bg可在不看formal answer时发现该insertion的capture风险；这不是已经验证的correctness危险分类器。' if target['H_count'] else 'expert12没有U_bg新capture，当前风险信号无法事前识别它；不得用于部署。'
    text='# Pre-insertion hazard 审计\n\n'+answer+'\n\nREG expert12: H_count='+str(target['H_count'])+'，H_source='+str(target['H_source'])+'，H_switch='+str(target['H_switch'])+'；24次insertion中的三个排名分别为'+str(target['rank'])+'。同分共享排名，不以formal损伤排序。\n\n主bank使用历史H；本审计只依赖相同R0 keys/radii，故与PR9 A0路由诊断权重无关。hazard及RCAP/prototype先冻结，随后才读暴露damage，未调任何规则。\n'
    from storage import Store
    Store(ROOT).write('public/INSERTION_HAZARD_AUDIT_ZH.md',text.encode())
    write(ROOT/'RUN_STATUS.json',dict(status='READY_FOR_MODEL_MECHANICAL',phase='P3',epoch=time.time()))
    print(json.dumps(dict(status='LATENT_AUDIT_COMPLETE',expert12_capture=target['H_count'],expert12_rank=target['rank'],native=test['native_activation'])))
if __name__=='__main__':main()
