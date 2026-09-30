"""Complete frozen REG12 counterfactual block; real generation, never inferred scores."""
import copy,hashlib
from resources import ROOT,read
def exposed_task():
    rows=read(ROOT/'private/LOCALITY_STRESS_HOLDOUT.json')['rows'];assert len(rows)==47
    return dict(canonical_edit_id='EXPOSED_REGRESSION',order=0,native=rows[0],fit_questions=[rows[0]['question']],evaluation=[dict(x,task='EXPOSED_REGRESSION',query_id='exposed-'+hashlib.sha256((x['image_sha256']+'\0'+x['question']).encode()).hexdigest()) for x in rows])
def early_panel(selected):
    panel=[]
    for t in selected[:8]:
        t=copy.deepcopy(t);t['evaluation']=[r for r in t['evaluation'] if r['task'] in ['T0','T1G','T2G']];panel.append(t)
    return panel
def generate(runtime,condition,tasks,protocol,folder):
    import evaluation,worker_v3 as old
    from router_r3 import RejectRouter
    from scope_worker import load_expert
    selected=[next(t for t in tasks if t['order']==o) for o in range(25,37)]
    bank={};router=RejectRouter();bindings=[]
    removed=int(condition.split('_')[1]) if condition!='FULL' else None
    for position,t in enumerate(selected,1):
        if position==removed:continue
        ex,b=load_expert(t,'A0');bank[t['canonical_edit_id']]=ex;bindings.append(b)
        key,radius=old.router_entry(runtime,t);router.add(t['canonical_edit_id'],key,radius)
    return evaluation.evaluate(runtime,[exposed_task()]+early_panel(selected),bank,router,condition,'ROUTE_REMOVAL',12,folder,bindings,protocol)
def report(rows,scores):
    from exact import identity
    from collections import Counter
    full={identity(r):r for r in rows if r['arm']=='FULL'};assert full
    result={}
    for condition in ['REMOVE_9','REMOVE_10','REMOVE_11','REMOVE_12']:
        other={identity(r):r for r in rows if r['arm']==condition};assert full.keys()==other.keys()
        cohorts={name:{x:Counter() for x in ['OFF_OFF','OFF_ON','ON_OFF','ON_ON_same','ON_ON_different']} for name in ['old47','early_positive']}
        buckets={x:Counter() for x in ['OFF_OFF','OFF_ON','ON_OFF','ON_ON_same','ON_ON_different']}
        for k,a in full.items():
            b=other[k];ra=a['route']['logical_edit_id'];rb=b['route']['logical_edit_id']
            route='OFF_OFF' if ra is None and rb is None else 'OFF_ON' if ra is None else 'ON_OFF' if rb is None else 'ON_ON_same' if ra==rb else 'ON_ON_different'
            va=scores.get(a['judge_key']);vb=scores.get(b['judge_key'])
            transition='missing' if va is None or vb is None else 'correct_wrong' if va and not vb else 'wrong_correct' if not va and vb else 'both_correct' if va else 'both_wrong'
            buckets[route][transition]+=1;buckets[route]['n']+=1
            cohort='old47' if a['task']=='EXPOSED_REGRESSION' else 'early_positive';cohorts[cohort][route][transition]+=1;cohorts[cohort][route]['n']+=1
        clean=lambda values:{k:{x:v.get(x,0) for x in ['n','correct_wrong','wrong_correct','both_correct','both_wrong','missing']} for k,v in values.items()}
        result[condition]=dict(all=clean(buckets),cohorts={k:clean(v) for k,v in cohorts.items()})
    return dict(status='COMPLETE',conditions=['FULL','REMOVE_9','REMOVE_10','REMOVE_11','REMOVE_12'],frozen_bank='historical REG A0 bank12 original positions25..36',rows_per_condition=len(full),non_target_inputs=47,early_positive_inputs=sum(r['task']!='EXPOSED_REGRESSION' for r in full.values()),buckets=result,causal_diagnostic_only=True,selection_usage=False,answers_from_actual_generation=True)
