"""Router-specific joint gates with exact shared-missing dependence."""
from collections import defaultdict,Counter
from exact import new_errors,delta,identity
def stats(rows,scores):
    n=len(rows);known=sum(scores.get(r['judge_key']) is True for r in rows);missing=sum(r['judge_key'] not in scores for r in rows)
    return dict(n=n,known_correct=known,missing=missing,bounds=[known/n,(known+missing)/n] if n else None)
def errors(a,b,scores,maximum):
    if not a and not b:return dict(status='NOT_APPLICABLE',reason='frozen qualified denominator is zero',n=0)
    result=new_errors(a,b,scores)
    return dict(result,status='PASS' if result['new_error_max']<=maximum else 'FAIL' if result['new_error_min']>maximum else 'INCONCLUSIVE',maximum_allowed=maximum)
def route_bucket(a,b):
    x=a['route']['logical_edit_id'];y=b['route']['logical_edit_id']
    return 'OFF_OFF' if x is None and y is None else 'OFF_ON' if x is None else 'ON_OFF' if y is None else 'ON_ON_same' if x==y else 'ON_ON_different'
def transitions(a,b,scores,shared_only=False):
    aa={identity(r):r for r in a};bb={identity(r):r for r in b}
    if not shared_only:assert aa.keys()==bb.keys() and aa
    buckets=defaultdict(Counter);matrix=Counter();special=Counter();n=0
    for k in aa.keys()&bb.keys():
        c=aa[k];r=bb[k];route=route_bucket(r,c);vc=scores.get(c['judge_key']);vr=scores.get(r['judge_key']);label='missing' if vc is None or vr is None else 'wrong_correct' if not vr and vc else 'correct_wrong' if vr and not vc else 'both_correct' if vr else 'both_wrong'
        buckets[route][label]+=1;buckets[route]['n']+=1;matrix[(r['route']['logical_edit_id'],c['route']['logical_edit_id'])]+=1;n+=1
        changed=r['route']['logical_edit_id']!=c['route']['logical_edit_id']
        if changed and label=='wrong_correct':special['Negative_Rescue']+=1
        if changed and label=='correct_wrong':special['Negative_Damage']+=1
        legal=r['task'] in ['T0','T1G','T2G'] and r['route']['logical_edit_id']==r['edit']
        if legal and vr is True and c['route']['logical_edit_id'] is None:special['Positive_Rejection']+=1
    fields=['n','wrong_correct','correct_wrong','both_correct','both_wrong','missing']
    result=dict(n=n,activation_count=sum(r['route']['activated'] for r in a if identity(r) in bb),buckets={k:{f:v.get(f,0) for f in fields} for k,v in buckets.items()},expert_switch_matrix=[dict(from_expert=x,to_expert=y,count=count) for (x,y),count in sorted(matrix.items(),key=lambda v:str(v[0]))],**{k:special[k] for k in ['Positive_Rejection','Negative_Rescue','Negative_Damage']})
    result['NetRescue']=sum(v['wrong_correct']-v['correct_wrong'] for v in result['buckets'].values());return result
def joint(rows,refs,scores,masks,entries):
    base={(r['edit'],r['task'],r['query_id']):r['base_correct'] for r in masks['rows']}
    qualified=lambda rs:[r for r in rs if r['task'] not in ['T1L','T2L'] or base.get((r['edit'],r['task'],r['query_id'])) is True]
    baseline=qualified(refs);out={}
    for method in ['RCAP','NEG0','SAFE']:
        selected=qualified([r for r in rows if r['arm']==method]);gates={};summary={};rescue=0
        for mode,prefix in [('single',1)]+[('sequential',p) for p in [4,8,12,24]]:
            panel=lambda rs:[r for r in rs if r['mode']==mode and r['prefix']==prefix]
            a=panel(selected);b=panel(baseline);name=mode+'/'+str(prefix)
            for task,maximum in [('T0',0),('T1G',0),('T2G',1),('T2L',1)]:
                taskrows=lambda rs:[r for r in rs if r['task']==task]
                gates[name+'/'+task]=errors(taskrows(a),taskrows(b),scores,maximum)
            summary[name]={task:stats([r for r in a if r['task']==task],scores) for task in ['T0','T1G','T2G','T1L','T2L','T2L_PRESSURE']}
            rescue+=transitions(a,b,scores)['Negative_Rescue']
            if mode=='sequential' and prefix==24:
                aa=[r for r in a if r['task']=='T2L_PRESSURE'];bb=[r for r in b if r['task']=='T2L_PRESSURE'];d=delta(aa,bb,scores);gain=stats(aa,scores)['known_correct']-stats(bb,scores)['known_correct']
                gates['bank24/Pressure']=dict(d,known_correct_gain=gain,status='PASS' if gain>=3 and d['delta_min']>=-1e-12 else 'FAIL' if gain<3 or d['delta_max']<0 else 'INCONCLUSIVE')
        for prefix in [4,8,12,24]:
            panel=lambda rs:[r for r in rs if r['mode']=='EXPOSED_REGRESSION' and r['prefix']==prefix]
            a=panel(selected);b=panel(baseline);eligible=lambda rs:[r for r in rs if r.get('frozen_base_correct') is True]
            assert len(eligible(a))==len(eligible(b))==35
            gates['EXPOSED_old35/'+str(prefix)]=errors(eligible(a),eligible(b),scores,1);rescue+=transitions(a,b,scores)['Negative_Rescue']
        if method in ['RCAP','SAFE']:gates['native_radius_separation']=dict(status='PASS' if all(e['cap_status']=='PASS' for e in entries) else 'FAIL')
        if method in ['NEG0','SAFE']:gates['negative_prototype_support']=dict(status='PASS' if all(e['negative_status']=='PASS' for e in entries) else 'FAIL')
        statuses=[g['status'] for g in gates.values()];accuracy='FAIL' if 'FAIL' in statuses else 'INCONCLUSIVE' if 'INCONCLUSIVE' in statuses else 'PASS'
        status='NO_ROUTING_EFFECT' if accuracy=='PASS' and rescue==0 else accuracy
        out[method]=dict(status=status,accuracy_status=accuracy,gates=gates,summary=summary,known_route_rescue_occurrences=rescue,route_rescue_count_counts_panels_not_unique_inputs=True)
    passing=[m for m in ['RCAP','NEG0','SAFE'] if out[m]['status']=='PASS'];selected=passing[0] if passing else None
    return dict(status='ADMITTED' if selected else 'NOT_ADMITTED',candidates=out,selected=selected,REG_allowed=selected is not None,REG_methods=['R0',selected] if selected else [],complexity_order=['RCAP','NEG0','SAFE'],weighted_total_score=False,dependency_aware_missing=True,REG='EXPOSED_REGRESSION; not independent confirmation')
if __name__=='__main__':
    row=lambda i,k:dict(edit='e',task='T2G',input_id=str(i),judge_key=k)
    assert errors([row(1,'z')],[row(1,'z')],{},0)['status']=='PASS'
    assert errors([row(1,'z')],[row(1,'yes')],{'yes':True},0)['status']=='INCONCLUSIVE'
    assert errors([row(1,'no'),row(2,'no')],[row(1,'yes'),row(2,'yes')],{'yes':True,'no':False},1)['status']=='FAIL'
    assert errors([],[],{},0)['status']=='NOT_APPLICABLE'
    print('PASS: shared missing, known failure, frozen empty denominator')
