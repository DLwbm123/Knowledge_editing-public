"""Cached exact R0 replay, no generation and no fitted thresholds."""
import os,sys,json,collections
from pathlib import Path
import torch
ROOT=Path(os.environ['RUN_ROOT']);OLD=Path(os.environ['PREDECESSOR_ROOT'])
sys.path[:0]=[str(ROOT),str(ROOT/'source_patch'),str(ROOT/'source')]
from router_r3 import RejectRouter
from m3bench_repro.editors.routing import balanced_radius
from scripts.medtrace.stage17_prepare import digest

def read(p):return json.loads(p.read_text())
def iid(r):return digest([r['image_sha256'],r['question']])
def feature(r):return torch.load(OLD/'private/keys'/f'{iid(r)}.pt',map_location='cpu',weights_only=True)
def bucket(a,b):
    if not a.activated:return 'OFF_ON' if b.activated else 'OFF_OFF'
    if not b.activated:return 'ON_OFF'
    return 'ON_ON_SAME' if a.logical_edit_id==b.logical_edit_id else 'ON_ON_SWITCH'
def main():
    tasks=read(OLD/'private/TASKS_R2_LOCKED.json')['tasks'];scores={p.stem:read(p)['is_correct'] for p in (OLD/'private/judge/scores').glob('*.json')}
    output=[];private=[]
    for panel,ts,bankprefix in [('DEV24',tasks[:24],'P2'),('REG24',tasks[24:],'P4')]:
        routers={};r=RejectRouter();entries=[]
        for i,t in enumerate(ts,1):
            n=t['native'];k=feature(n);positive=feature(dict(n,question=t['fit_questions'][0]));negative=feature(dict(n,image_sha256='BLACK-'+n['image_sha256']))
            radius=balanced_radius(k,positive,negative,alpha=.2,distance='euclidean');r.add(t['canonical_edit_id'],k,radius);entries.append((t['canonical_edit_id'],k,radius))
            if i in [8,9,10,11,12,24]:
                router=RejectRouter()
                for e in entries:router.add(*e)
                routers[i]=router
        cohorts={'old47':read(OLD/'private/LOCALITY_STRESS_HOLDOUT.json')['rows'],'early8_positive':[x for t in ts[:8] for x in t['evaluation'] if x['task'] in ['T0','T1G','T2G']]}
        for cohort,rows in cohorts.items():
            features={iid(x):feature(x) for x in rows};decisions={i:{k:router.route(f) for k,f in features.items()} for i,router in routers.items()}
            for a,b in [(8,9),(9,10),(10,11),(11,12),(8,12),(12,24)]:
                counts=collections.Counter(bucket(decisions[a][k],decisions[b][k]) for k in features)
                output.append(dict(panel=panel,cohort=cohort,from_prefix=a,to_prefix=b,unique_inputs=len(features),transitions=dict(counts),outcomes_available=a in [8,12,24] and b in [8,12,24]))
            for method in ['E_orig','A0','AH','AHS_01']:
                observed={}
                for prefix in [8,12,24]:
                    p=OLD/f'auxiliary/holdout/{panel}-{method}/p{prefix}/CONSUMERS.json' if cohort=='old47' else OLD/f'jobs/{bankprefix}-bank-{method}/p{prefix}/CONSUMERS.json'
                    records=[x for x in read(p) if x['input_id'] in features and (cohort=='old47' or x['task'] in ['T0','T1G','T2G'])]
                    observed[prefix]={x['input_id']:x for x in records};assert set(observed[prefix])==set(features)
                    for k,x in observed[prefix].items():
                        d=decisions[prefix][k];assert d.activated==x['route']['activated'] and d.logical_edit_id==x['route']['logical_edit_id'],'Cached exact R0 replay mismatch'
                for a,b in [(8,12),(12,24)]:
                    counts=collections.Counter()
                    for k in features:
                        x,y=observed[a][k],observed[b][k];av,bv=scores.get(x['judge_key']),scores.get(y['judge_key'])
                        outcome='missing' if av is None or bv is None else 'correct_to_wrong' if av and not bv else 'wrong_to_correct' if bv and not av else 'same'
                        counts[bucket(decisions[a][k],decisions[b][k])+'/'+outcome]+=1
                    output.append(dict(panel=panel,cohort=cohort,method=method,from_prefix=a,to_prefix=b,unique_inputs=len(features),route_correctness=dict(counts)))
            private.append(dict(panel=panel,cohort=cohort,decisions={i:{k:vars(v) for k,v in d.items()} for i,d in decisions.items()}))
    (ROOT/'private/ROUTE_DECISIONS.json').write_text(json.dumps(private))
    result=dict(status='COMPLETE_CACHED_R0',actual_algorithm='nearest first, argmin first-entry ties, test nearest radius, kappa=1 mu=0',groups=output,associated_ID_not_unique_correct_scope=True,intermediate_prefix_outcomes='NOT_GENERATED; only route changes at 9/10/11; no expert-ID-only answer reuse',Judge_attempts=0)
    (ROOT/'public/ROUTE_TRANSITIONS.json').write_text(json.dumps(result,indent=2))
    (ROOT/'public/ROUTE_REMOVAL_DIAGNOSTIC.json').write_text(json.dumps(dict(status='PLANNED_NOT_ADMITTED',conditions=[9,10,11,12],reason='Optional complete paired block needs separate scoring reservation; cached route diagnostics completed',reestimate_radii=False),indent=2))
    print(json.dumps(dict(status=result['status'],groups=len(output),Judge_attempts=0)))
if __name__=='__main__':main()
