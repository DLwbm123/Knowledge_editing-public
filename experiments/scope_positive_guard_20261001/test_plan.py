"""Small executable admission checks; no model, data, GPU, or Judge."""
import copy
from decision import joint
def fixture():
    refs=[];scores={'yes':True,'no':False};masks=[]
    def row(mode,prefix,task,i,key):
        return dict(arm='R0',mode=mode,prefix=prefix,edit='e',task=task,input_id=str(i),query_id=str(i),judge_key=key,source_group='s',route=dict(logical_edit_id='e',activated=True),frozen_base_correct=True)
    for mode,prefix in [('single',1)]+[('sequential',p) for p in [4,8,12,24]]:
        for task in ['T0','T1G','T2G','T2L']:
            r=row(mode,prefix,task,1,'yes');refs.append(r)
            if task=='T2L':masks.append(dict(edit='e',task=task,query_id='1',base_correct=True))
        for i in range(5):refs.append(row(mode,prefix,'T2L_PRESSURE',i,'yes' if i<2 else 'no'))
    for prefix in [4,8,12,24]:
        for i in range(35):refs.append(row('EXPOSED_REGRESSION',prefix,'EXPOSED_REGRESSION',i,'yes'))
    candidates=[]
    for method in ['PLOO']:
        for r in refs:
            r=copy.deepcopy(r);r['arm']=method
            if r['task']=='T2L_PRESSURE' and r['judge_key']=='no':r.update(judge_key='yes',route=dict(logical_edit_id=None,activated=False))
            candidates.append(r)
    return candidates,refs,scores,dict(rows=masks),[dict(cap_status='PASS',negative_status='PASS')]
def selfcheck():
    c,b,s,m,e=fixture();x=joint(c,b,s,m,e);assert x['selected']=='PLOO' and x['REG_methods']==['R0','PLOO']
    for r in c:
        if r['task']=='T1G':r['judge_key']='no'
    assert not joint(c,b,s,m,e)['REG_allowed']
    c,b,s,m,e=fixture()
    for r in c:r['route']=dict(logical_edit_id='e',activated=True)
    x=joint(c,b,s,m,e);assert not x['REG_allowed'] and all(d['status']=='NO_ROUTING_EFFECT' for d in x['candidates'].values())
    c,b,s,m,e=fixture()
    for r in c:
        if r['task']=='T1G':r['judge_key']='unknown'
    x=joint(c,b,s,m,e);assert not x['REG_allowed'] and all(d['status']=='INCONCLUSIVE' for d in x['candidates'].values())
    print('PASS: scientific gates, complexity tie-break, missing, required routing effect, no extra REG')
if __name__=='__main__':selfcheck()
