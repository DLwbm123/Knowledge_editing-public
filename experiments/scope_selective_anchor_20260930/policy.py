"""Predeclared CHECK selection and DEV joint gates; no training dependencies."""
from collections import defaultdict

def stats(rows,scores):
    values=[scores.get(x['judge_key']) for x in rows];n=len(values);correct=sum(x is True for x in values);missing=sum(x is None for x in values)
    sources=defaultdict(list)
    for r,v in zip(rows,values):sources[r['source_group']].append(v)
    edits=defaultdict(list)
    for r,v in zip(rows,values):edits[r['edit']].append(v)
    return dict(n=n,correct=correct,missing=missing,bounds=[correct/n,(correct+missing)/n] if n else None,unique_inputs=len({r['input_id'] for r in rows}),sources=len(sources),source_macro_bounds=[sum(sum(v is True for v in vs)/len(vs) for vs in sources.values())/len(sources),sum(sum(v is not False for v in vs)/len(vs) for vs in sources.values())/len(sources)] if sources else None,edits=len(edits),edit_macro_bounds=[sum(sum(v is True for v in vs)/len(vs) for vs in edits.values())/len(edits),sum(sum(v is not False for v in vs)/len(vs) for vs in edits.values())/len(edits)] if edits else None)

def key(r):return (r['edit'],r['task'],r['input_id'])
def no_new_failures(candidate,baseline,scores):
    a={key(r):scores.get(r['judge_key']) for r in candidate};b={key(r):scores.get(r['judge_key']) for r in baseline}
    if not a or set(a)!=set(b):return 'INCONCLUSIVE'
    if any(b[k] is True and a[k] is False for k in a):return 'FAIL'
    if any((b[k] is True and a[k] is None) or (b[k] is None and a[k] is not True) for k in a):return 'INCONCLUSIVE'
    return 'PASS'

def select_grid(grid,baseline,scores):
    """Use SP CHECK for shared beta, or scaled AH CHECK for gamma; tie larger."""
    result={};eligible=[]
    bp=[r for r in baseline if r['task']=='CHECK_POS']
    for value,rows in grid.items():
        pos=[r for r in rows if r['task']=='CHECK_POS'];neg=[r for r in rows if r['task']=='CHECK_NEG']
        status=no_new_failures(pos,bp,scores);summary=stats(neg,scores)
        if status=='PASS' and (not summary['n'] or summary['missing']):status='INCONCLUSIVE'
        result[str(value)]=dict(status=status,positive=stats(pos,scores),negative=summary)
        if status=='PASS':eligible.append((summary['source_macro_bounds'][0],value))
    winner=max(eligible)[1] if eligible else None
    return dict(status='SELECTED' if winner is not None else 'INCONCLUSIVE' if any(x['status']=='INCONCLUSIVE' for x in result.values()) else 'NOT_ADMITTED',selected=winner,grid=result,rule='no new known CHECK_POS errors vs H; max complete CHECK_NEG source-macro; tie higher value')

def delta_gate(candidate,baseline,scores,minimum):
    a,b=stats(candidate,scores),stats(baseline,scores)
    if not a['n'] or a['n']!=b['n'] or {key(r) for r in candidate}!={key(r) for r in baseline}:return dict(status='INCONCLUSIVE',reason='unmatched or empty denominator')
    lo=a['bounds'][0]-b['bounds'][1];hi=a['bounds'][1]-b['bounds'][0]
    return dict(status='PASS' if lo>=minimum-1e-12 else 'FAIL' if hi<minimum-1e-12 else 'INCONCLUSIVE',delta_bounds=[lo,hi],minimum=minimum,candidate=a,baseline=b)

def joint(candidates,references,scores):
    """Rows for one full DEV: single, bank4/8/12/24 and old47; no pool inflation."""
    decisions={};vectors={}
    for method,rows in candidates.items():
        gates={};vector={}
        for mode,prefix in [('single',1)]+[('sequential',i) for i in [4,8,12,24]]:
            panel=lambda rs:[r for r in rs if r['mode']==mode and r['prefix']==prefix]
            c=panel(rows);ref={k:panel(v) for k,v in references.items()};name=f'{mode}/{prefix}'
            cp=[r for r in c if r['task']=='T0'];hp=[r for r in ref['H'] if r['task']=='T0']
            t0=no_new_failures(cp,hp,scores)
            if any(scores.get(r['judge_key']) is None for r in cp+hp):t0='INCONCLUSIVE' if t0!='FAIL' else t0
            gates[name+'/T0']=dict(status=t0)
            for task in ['T1G','T2G']:
                for reference in ['A0','E_orig']:
                    gates[name+'/'+task+'/'+reference]=delta_gate([r for r in c if r['task']==task],[r for r in ref[reference] if r['task']==task],scores,-.01)
            gates[name+'/generality/H']=delta_gate([r for r in c if r['task'] in ['T1G','T2G']],[r for r in ref['H'] if r['task'] in ['T1G','T2G']],scores,0.)
            for task in ['T0','T1G','T2G','T1L','T2L','T2L_PRESSURE']:
                ss=stats([r for r in c if r['task']==task],scores)
                if ss['n']:vector[name+'/'+task]=ss
            if prefix==24:
                gates['pressure_gain']=delta_gate([r for r in c if r['task']=='T2L_PRESSURE'],[r for r in ref['H'] if r['task']=='T2L_PRESSURE'],scores,.05)
        for prefix in [4,8,12,24]:
            eligible=lambda rs:[r for r in rs if r['mode']=='holdout' and r['prefix']==prefix and r.get('frozen_base_correct') is True]
            c,h=eligible(rows),eligible(references['H']);n=len(c)
            gates[f'old47/{prefix}']=delta_gate(c,h,scores,-1/n) if n else dict(status='INCONCLUSIVE')
            if n:vector[f'old47/{prefix}']=stats(c,scores)
        statuses=[g['status'] for g in gates.values()];status='FAIL' if 'FAIL' in statuses else 'INCONCLUSIVE' if 'INCONCLUSIVE' in statuses else 'PASS'
        decisions[method]=dict(status=status,gates=gates,dominated_by=[]);vectors[method]=vector
    for method,v in vectors.items():
        for other,w in vectors.items():
            if method==other or not v or set(v)!=set(w):continue
            if all(v[k]['missing']==w[k]['missing']==0 for k in v):
                if all(w[k]['bounds'][0]>=v[k]['bounds'][0] for k in v) and any(w[k]['bounds'][0]>v[k]['bounds'][0] for k in v):
                    decisions[method]['dominated_by'].append(other);decisions[method]['status']='FAIL'
            # Missing fields are preserved. No unsupported dominance claim.
            elif decisions[method]['status']=='PASS' and all(w[k]['bounds'][1]>=v[k]['bounds'][0] for k in v) and any(w[k]['bounds'][1]>v[k]['bounds'][0] for k in v):
                decisions[method]['status']='INCONCLUSIVE';decisions[method].setdefault('possible_dominance_missing',[]).append(other)
    admitted=any(d['status']=='PASS' for d in decisions.values())
    return dict(status='ADMITTED' if admitted else 'NOT_ADMITTED',candidates=decisions,REG_allowed=admitted,missing_bounds_use_full_denominator=True,statistical_noninferiority_proof=False)

if __name__=='__main__':
    r=lambda edit,task,k:dict(edit=edit,task=task,judge_key=k,input_id=edit+task,source_group=edit)
    h=[r('a','CHECK_POS','hp'),r('a','CHECK_NEG','hn')]
    a=[r('a','CHECK_POS','ap'),r('a','CHECK_NEG','an')]
    b=[r('a','CHECK_POS','bp'),r('a','CHECK_NEG','bn')]
    s=dict(hp=True,hn=False,ap=True,an=True,bp=True,bn=True)
    assert select_grid({.1:a,1.:b},h,s)['selected']==1.
    del s['bp'];assert select_grid({.1:a,1.:b},h,s)['selected']==.1
    s['ap']=False;assert select_grid({.1:a,1.:b},h,s)['selected'] is None
    del s['an']
    assert delta_gate(a,h,s,-.01)['status']=='INCONCLUSIVE'
    x=stats([r('a','T0','yes'),r('a','T1G','no'),r('b','T0','yes')],dict(yes=True,no=False))
    assert x['correct']==2 and x['edits']==2 and x['edit_macro_bounds']==[.75,.75]
    print('PASS: shared-grid selection, higher tie, missing and no-extra-positive-error gates')
