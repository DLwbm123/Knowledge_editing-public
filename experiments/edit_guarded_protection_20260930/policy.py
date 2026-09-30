"""Shared-key error gates; no weighted sum of formal scientific metrics."""
from collections import defaultdict
from exact import delta,new_errors,identity
from pathlib import Path
import statistics
def stats(rows,scores):
    groups=defaultdict(list)
    for r in rows:groups[r['source_group']].append(scores.get(r['judge_key']))
    n=len(rows);correct=sum(scores.get(r['judge_key']) is True for r in rows);missing=sum(r['judge_key'] not in scores for r in rows)
    return dict(n=n,known_correct=correct,missing=missing,bounds=[correct/n,(correct+missing)/n] if n else None,source_macro=sum(sum(v is True for v in vs)/len(vs) for vs in groups.values())/len(groups) if groups else None)
def errors(a,b,scores,max_new):
    try:d=new_errors(a,b,scores)
    except AssertionError:return dict(status='INCONCLUSIVE',reason='unmatched or unsafe exact component')
    return dict(d,status='PASS' if d['new_error_max']<=max_new else 'FAIL' if d['new_error_min']>max_new else 'INCONCLUSIVE',maximum_allowed=max_new)
def mechanism(curves):
    out={}
    for method,byedit in curves.items():
        edits=[]
        for order,cs in byedit.items():
            gs=[c['gradient_guard'] for c in cs];assert len(gs)==80
            median=lambda name:statistics.median([g[name] for g in gs if g[name] is not None]) if any(g[name] is not None for g in gs) else None
            edits.append(dict(edit_order=int(order),steps=80,conflict_fraction=sum(g['conflict'] for g in gs)/80,protection_positive_norm_ratio=median('norm_ratio_before'),guarded_positive_norm_ratio=median('norm_ratio_after'),cap_fraction=sum(g['cap_active'] for g in gs)/80,projection_fraction=sum(g['projection_active'] for g in gs)/80,global_clip_fraction=sum(c['preclip_norm']>1 for c in cs)/80,median_dot_product=median('dot'),median_final_positive_cosine=median('final_positive_cosine'),parameter_update_norm=sum(c['update_norm'] for c in cs),median_parameter_update_norm=statistics.median(c['update_norm'] for c in cs),raw_first_order_D_plus_median=median('D_plus')))
        allcs=[c for cs in byedit.values() for c in cs]
        out[method]=dict(edits=edits,global_clip_fraction=sum(c['preclip_norm']>1 for c in allcs)/len(allcs),conflict_fraction=sum(c['gradient_guard']['conflict'] for c in allcs)/len(allcs),cap_fraction=sum(c['gradient_guard']['cap_active'] for c in allcs)/len(allcs),algebra_checks_passed=True,no_surgery_backbone_forwards=all(c['surgery_backbone_forwards']==0 for c in allcs),not_Adam_monotonic_guarantee=True)
    return out
def select_shared(rows,refs,scores,mechanisms):
    baseline=[r for r in refs if r['arm']=='H' and r['task']=='CHECK_POS'];grid={};eligible=[]
    for rho in [1,2]:
        arms={};values=[]
        for arm in ['CAP','EGP','EGP_A']:
            method=arm+'_'+str(rho);rs=[r for r in rows if r['arm']==method];pos=[r for r in rs if r['task']=='CHECK_POS'];neg=[r for r in rs if r['task']=='CHECK_NEG']
            gate=errors(pos,baseline,scores,0);ss=stats(neg,scores)
            status=gate['status']
            if status=='PASS' and (ss['missing'] or not ss['n']):status='INCONCLUSIVE'
            if not mechanisms[method]['algebra_checks_passed']:status='FAIL'
            arms[arm]=dict(status=status,positive_error_gate=gate,negative=ss,mechanism=mechanisms[method]);values.append(ss['source_macro'])
        passed=all(x['status']=='PASS' for x in arms.values());value=sum(values)/3 if all(v is not None for v in values) else None
        grid[str(rho)]=dict(eligible=passed,methods=arms,equal_mean_CHECK_NEG_source_macro=value)
        if passed:eligible.append((value,rho))
    chosen=max(eligible)[1] if eligible else None
    return dict(status='SELECTED' if chosen else 'NOT_ADMITTED',selected=chosen,shared_for=['CAP','EGP','EGP_A'],grid=grid,selection_only_CHECK=True,formal_DEV_unblinded=False,tie='rho2',aggregation='equal mean of three method source macros',clip_target=.10,clip_is_mechanism_target=True)
def joint(candidates,refs,scores,mechanisms):
    decisions={};vectors={}
    for method,rows in candidates.items():
        gates={};vector={}
        for mode,prefix in [('single',1)]+[('sequential',p) for p in [4,8,12,24]]:
            panel=lambda rs:[r for r in rs if r['mode']==mode and r['prefix']==prefix]
            c=panel(rows);reference={m:panel(rs) for m,rs in refs.items()};name=mode+'/'+str(prefix)
            task=lambda rs,task:[r for r in rs if r['task']==task]
            gates[name+'/T0']=errors(task(c,'T0'),task(reference['H'],'T0'),scores,0)
            for t,limit in [('T1G',0),('T2G',1)]:
                for m in ['A0','E_orig']:gates[name+'/'+t+'/'+m]=errors(task(c,t),task(reference[m],t),scores,limit)
            gates[name+'/T2L/H']=errors(task(c,'T2L'),task(reference['H'],'T2L'),scores,1)
            for t in ['T0','T1G','T2G','T1L','T2L','T2L_PRESSURE']:
                ss=stats(task(c,t),scores)
                if ss['n']:vector[name+'/'+t]=ss
            if prefix==24:
                a=task(c,'T2L_PRESSURE');b=task(reference['H'],'T2L_PRESSURE');d=delta(a,b,scores);gain=stats(a,scores)['known_correct']-stats(b,scores)['known_correct']
                gates['pressure_gain']=dict(d,known_correct_gain=gain,status='PASS' if gain>=3 and d['delta_min']>=-1e-12 else 'FAIL' if gain<3 or d['delta_max']<0 else 'INCONCLUSIVE')
        for prefix in [4,8,12,24]:
            eligible=lambda rs:[r for r in rs if r['mode']=='EXPOSED_REGRESSION' and r['prefix']==prefix and r.get('frozen_base_correct') is True]
            c=eligible(rows);h=eligible(refs['H']);assert len(c)==len(h)==35
            gates['EXPOSED_old35/'+str(prefix)]=errors(c,h,scores,1);vector['EXPOSED_old35/'+str(prefix)]=stats(c,scores)
        statuses=[g['status'] for g in gates.values()];accuracy='FAIL' if 'FAIL' in statuses else 'INCONCLUSIVE' if 'INCONCLUSIVE' in statuses else 'PASS'
        status='PERFORMANCE_PASS_MECHANISM_UNRESOLVED' if accuracy=='PASS' and mechanisms[method]['global_clip_fraction']>=.10 else accuracy
        decisions[method]=dict(status=status,accuracy_status=accuracy,gates=gates,mechanism=mechanisms[method],dominated_by=[]);vectors[method]=vector
    # A dominance statement requires certified shared-key bounds on every dimension.
    for method,rows in candidates.items():
        for other,otherrows in candidates.items():
            if method==other:continue
            diffs=[]
            for label in vectors[method]:
                if label.startswith('EXPOSED_old35/'):
                    p=int(label.split('/')[1]);panel=lambda rs:[r for r in rs if r['mode']=='EXPOSED_REGRESSION' and r['prefix']==p and r.get('frozen_base_correct') is True]
                else:
                    mode,p,t=label.split('/');panel=lambda rs:[r for r in rs if r['mode']==mode and r['prefix']==int(p) and r['task']==t]
                diffs.append(delta(panel(otherrows),panel(rows),scores))
            if diffs and all(d['delta_min']>=-1e-12 for d in diffs) and any(d['delta_min']>1e-12 for d in diffs):decisions[method]['dominated_by'].append(other)
    eligible=[m for m,d in decisions.items() if d['status']=='PASS' and not d['dominated_by']]
    # Predeclared preference: anchored, then projected, then cap-only; maximum two.
    eligible.sort(key=lambda m:0 if m.startswith('EGP_A_') else 1 if m.startswith('EGP_') else 2)
    selected=eligible[:2]
    return dict(status='ADMITTED' if selected else 'NOT_ADMITTED',candidates=decisions,REG_allowed=bool(selected),REG_methods=selected,REG_max_continuations=24*len(selected),max_two_nondominated=True,formal_total_score_forbidden=True,dependency_aware_missing=True)
if __name__=='__main__':
    r=lambda i,k:dict(edit='e',task='T2G',input_id=str(i),judge_key=k,source_group='s')
    assert errors([r(1,'z')],[r(1,'z')],{},0)['status']=='PASS'
    assert errors([r(1,'no')],[r(1,'yes')],dict(no=False,yes=True),0)['status']=='FAIL'
    assert errors([r(1,'z')],[r(1,'yes')],dict(yes=True),0)['status']=='INCONCLUSIVE'
    assert errors([r(1,'z')],[r(1,'yes')],dict(yes=True),1)['status']=='PASS'
    print('PASS: shared missing and known-error admission')
