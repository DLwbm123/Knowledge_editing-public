"""Actual terminal cohorts only; historical J unknowns never become new Judge requests."""
from collections import Counter
import json,os,sqlite3,time,sys
from pathlib import Path
sys.path[:0]=[os.environ['RUN_ROOT']+'/private/tools',os.environ['RUN_ROOT']+'/private/source']
import astra_report as ar
from panels import protection_ids
from logic import transitions
from report_fs import audit_requests
from historical_binding import binding_key
R,c,r=ar.RUN,ar.common,ar.r
ACTIVE=tuple(c.read(R/'private/FROZEN_PROTOCOL.json').get('active_arms',('HR',)))
assert ACTIVE==('HR',)
ARMS=('SH00','SH10','PM','PU','HM','HU',*ACTIVE);CONTRASTS=(('HR','HM'),('HR','SH00'),('HR','HU'),('HR','SH10'),('SH10','SH00'))

def main():
    marker=R/'private/REPORT_COMPLETE.json'
    if marker.exists():print(c.read(R/'public/RESULTS.json')['status']);return
    cost=c.read(R/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in cost['gpu_sessions'])
    gen=c.read(R/'private/GENERATION_COMPLETE.json');terminals=c.read(R/'private/ARM_TERMINALS.json')
    assert (R/'private/FINAL_CONSUMED.json').exists() and (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    rows=list(db.execute('SELECT * FROM consumer'));assert len(rows)==gen['new_consumers']
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')};scores={k:x['correct'] for k,x in payloads.items()};audit_requests(payloads,cost)
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json');epoch=c.read(R/'private/judge_purew146/EPOCH_MANIFEST.json')
    records=[];lookup={};coverage={a:Counter() for a in ARMS}
    old=Path(c.read(R/'private/FROZEN_PROTOCOL.json')['reference_run'])
    projection=Path(c.read(R/'private/FROZEN_PROTOCOL.json')['projection_reference_run'])
    hinge=Path(c.read(R/'private/FROZEN_PROTOCOL.json')['hinge_reference_run'])
    roots=dict(SH00=old,SH10=old,PM=projection,PU=projection,HM=hinge,HU=hinge,HR=R)
    files=[(arm,p,None) for arm in ('SH00','SH10','PM','PU','HM','HU') for p in (roots[arm]/'private/outputs'/arm).glob('*/*/*/*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
    assert len(files)==522
    files.extend((x['method'],Path(x['path']),x['output_binding']) for x in rows)
    for arm,path,locked in files:
        d=c.read(path)
        if locked:assert c.digest(d)==locked
        bb=d['binding'];phase=bb['phase'];query=ledger['queries'][bb['input']['query_id']];assert bb['input']==query
        key,full=binding_key(query,bb['judge_input'],d['R0'],epoch)
        if key in payloads:assert json.loads(payloads[key]['binding'])==full;status=payloads[key]['status']
        else:assert arm not in ACTIVE;scores[key]=None;status='UNKNOWN_UNATTEMPTED'
        coverage[arm][status]+=1
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:phase['prefix']]} if phase['mode']=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=arm,node=0,prefix=phase['prefix']),input=query,mode=phase['mode'],owner_order=phase['owner']),active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),key))
        lookup[arm,phase['mode'],phase['prefix'],phase['owner'],query['query_id']]=key
    panels=[];coefs={};groups={}
    for arm in ARMS:
        for prefix in (1,):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                m,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,tasks[:prefix],prefix);m.pop('route_ON',None);n=m['observations']
                m['micro_bounds']=[100*m['known_correct']/n,100*(m['known_correct']+m['missing'])/n] if n else None
                panels.append(m);coefs[arm,prefix,task]=co;groups.update(g)
    prefix1=[]
    for a,b in CONTRASTS:
        ca,cb=coefs[a,1,'T2G'],coefs[b,1,'T2G'];assert set(ca)==set(cb)
        delta={e:r.combine([ca[e],{k:-v for k,v in cb[e].items()}]) for e in ca}
        bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in x.items()} for x in delta.values()]),scores)
        prefix1.append(dict(comparison=a+'-'+b,prefix=1,secondary_only=False,status='ESTIMATED',negative_edit_bounds=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),delta_macro_bounds_pp=[100*x for x in bounds],edit_ci=None,source_ci=None,source_groups=1,interpretation='NA_SINGLE_SOURCE; fixed-panel shared-missing bounds only'))
    contrasts=prefix1
    protection=[];endpoints=[];gates={};diagnostics={}
    for arm in ARMS:
        terminal=1;protected=True
        for task in ('T1L','T2L'):
            ids=protection_ids(tasks,ledger,task);keys=[lookup[arm,'protect100_R0',100,0,q] for q in ids]
            correct=sum(scores[k]==1 for k in keys);unknown=sum(scores[k] is None for k in keys)
            protection.append(dict(arm=arm,terminal_prefix=terminal,task=task,observations=len(ids),known_correct=correct,missing=unknown,Base_correct_to_wrong_bounds=[len(ids)-correct-unknown,len(ids)-correct]));protected &=correct==len(ids)
            for a,b in CONTRASTS:
                if arm==a and 1==terminal:
                    endpoints.append(dict(task=task,comparison=b+'-to-'+a,matched_terminal_prefix=terminal,**transitions([(lookup[b,'protect100_R0',100,0,q],lookup[a,'protect100_R0',100,0,q]) for q in ids],scores)))
        t1=next(x for x in panels if x['arm']==arm and x['prefix']==terminal and x['task']=='T1G')
        keys=[key for k,key in lookup.items() if k[0]==arm]
        gates[arm]=dict(zero_individual_Base_damage=protected,complete_first_source=(arm not in ACTIVE or terminals[arm]['slots']==20),T0_insertion_and_terminal=all(scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]]==scores[lookup[arm,'bank_R0',terminal,0,t['edit_id']]]==1 for t in tasks[:terminal]),T1G_full_retention=t1['known_correct']==t1['observations'],full_known_scoring=all(scores[k] is not None for k in keys))
        if arm in ACTIVE:
            ls=[]
            for owner in range(1,terminal+1):
                es=sorted([c.read(p) for p in (R/'private/steps'/arm/str(owner)).glob('*.json')],key=lambda e:e['slot']);assert len(es)==terminals[arm]['slots']
                source=c.read(R/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'))
                flags=Counter(z for e in es if not e['accepted'] for z in set([e['rejection_reason']] if e.get('rejection_reason') else [v for a in e['candidates'] for v in a['reasons']]))
                ls.append(dict(source_position=owner,slots=len(es),accepted=sum(e['accepted'] for e in es),zero_fraction=sum(e['zero_native'] for e in es)/len(es),rejected_slot_flags=dict(flags),measurements=source,final_path_and_net=c.read(R/'public'/('PATH_'+arm+'_'+str(owner)+'.json')),last_solver={k:es[-1].get(k) for k in ('multipliers','predicted_risks','current_actual_KLs','current_linear_predicted_KLs','projected_KKT_residual')},last_candidate_KLs=[a.get('current_removed_KLs') for a in es[-1]['candidates']]))
            diagnostics[arm]=ls
            m=R/'public'/('MARGIN_DIAGNOSTIC_'+arm+'.json')
            if m.exists():diagnostics[arm+'_margin']=c.read(m)
            gates[arm]['ten_medical_margins_positive_finite']=m.exists() and c.read(m)['finite']==c.read(m)['positive']==10
            diagnostics[arm+'_solver']=dict(fallback_slots=sum('fallback' in e for e in es),fallback_certified=sum(e.get('solver_path')=='R_FINITE_ACTIVE_SET' for e in es),fallback_status_counts=dict(Counter(e['fallback']['status'] for e in es if 'fallback' in e)),active_set_checks=sum(e.get('fallback',{}).get('active_sets_checked',0) for e in es),candidate_checks=sum(len(e['candidates']) for e in es),legacy_rejects_resolved=sum(e.get('solver_path')=='R_FINITE_ACTIVE_SET' and e['accepted'] for e in es),fallback_certificates=[dict(slot=e['slot'],**e['fallback']) for e in es if 'fallback' in e])
            matched=[x for x in contrasts if x['comparison'] in ('HR-HM','HR-SH00')]
            gates[arm].update(all_source_CE_below_point1=terminal==1 and all(x['measurements']['final']['weighted_all_FIT_CE']<.1 for x in ls),per_edit_T2G_not_worse=bool(matched) and all(x['status']=='ESTIMATED' and x['negative_edit_bounds']==0 for x in matched))
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks[:terminal]:
                ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]
                pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',terminal,0,q]) for q in dict.fromkeys(ids))
            endpoints.append(dict(arm=arm,task=task,comparison='actual-insertion-to-actual-terminal',terminal_prefix=terminal,**transitions(pairs,scores)))
    cross_T2G=[]
    ids=list(dict.fromkeys(q for e in tasks[0]['events'] if e['task']=='T2G' for q in e['all_probe_query_ids']))
    assert len(ids)==4
    for a,b in CONTRASTS:
        cross_T2G.append(dict(comparison=b+'-to-'+a,**transitions([(lookup[b,'bank_R0',1,0,q],lookup[a,'bank_R0',1,0,q]) for q in ids],scores)))
    interaction_coeff=dict(HR=1,HM=-1,SH10=-1,SH00=1)
    interaction=r.combine([{lookup[a,'bank_R0',1,0,q]:w/len(ids)} for a,w in interaction_coeff.items() for q in ids])
    unknown=sum(sum(v for k,v in x.items() if k in ('MISSING','UNKNOWN_UNATTEMPTED')) for x in coverage.values())
    status='PARTIAL_SCIENTIFIC_STOP' if not gen['full_protocol_complete'] else 'COMPLETE_WITH_UNKNOWN' if unknown else 'COMPLETE'
    result=dict(status=status,scientific_success=not unknown and gen['full_protocol_complete'] and all(all(gates[a].values()) for a in ACTIVE),full_protocol_complete=gen['full_protocol_complete'],actual_new_consumer_count=len(rows),historical_external_consumers=522,all_consumers=len(files),coverage={a:dict(v) for a,v in coverage.items()},terminals=terminals,panels=panels,contrasts=contrasts,cross_arm_T2G=cross_T2G,target_shape_protection_interaction=dict(comparison='(HR-HM)-(SH10-SH00)',T2G_delta_bounds_pp=[100*v for v in r.score_bounds(interaction,scores)],population_CI='NA_SINGLE_SOURCE'),prefix1_secondary=prefix1,protection=protection,endpoints=endpoints,gates=gates,diagnostics=diagnostics,unknown_consumers=unknown,new_Judge=cost['Judge_attempts'],GPU_process_hours=cost['gpu_seconds_used']/3600,actual_counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ('forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts','active_set_checks','candidate_checks')},owned_matrix_peak_bytes=cost.get('weights_observed_peak_bytes'),owned_GPU_peak_MiB=cost.get('GPU_memory_observed_peak_MiB'),failures=[dict(kind=p.stem,epoch=c.read(p).get('epoch')) for p in (R/'private/failures').glob('*.json')],historical_missing_not_rejudged=True,automatic146=False,K_matched_first3='NA_NO_REBUILD',nonlinear_clinical_privacy_certificate=False,calibration_coverage='Eight CT/MRI source groups; no KG/XRay; patient independence unknown')
    mechanical={}
    for arm in ARMS:
        base=roots[arm];internal=arm
        source=c.read(base/'public'/('SOURCE_'+internal+'_1.json'))
        events=sorted([c.read(p) for p in (base/'private/steps'/internal/'1').glob('*.json')],key=lambda x:x['slot'])
        mechanical[arm]=dict(final_complete_CE=source['final']['weighted_all_FIT_CE'] if len(events)==20 else None,last_available_CE=source['final']['weighted_all_FIT_CE'],slots=len(events),accepted=sum(e['accepted'] for e in events),rejected=sum(not e['accepted'] for e in events),zero_fraction=sum(e['zero_native'] for e in events)/len(events),first_hit_CE={str(t):next((e['slot'] for e in events if e.get('complete_source_after_slot',{}).get('weighted_all_FIT_CE',float('inf'))<t),None) for t in (1,.5,.1)})
    weights=[(a+'-'+b,{a:1,b:-1}) for a,b in CONTRASTS]
    mechanical_contrasts={name:{metric:(sum(w*mechanical[a][metric] for a,w in coeff.items()) if all(mechanical[a][metric] is not None for a in coeff) else None) for metric in ('final_complete_CE','accepted','zero_fraction')} for name,coeff in weights}
    # Fixed protection correctness uses the same key coefficients; loss count reverses their sign.
    protection_factorial=[]
    for task in ('T1L','T2L'):
        ids=protection_ids(tasks,ledger,task)
        for name,coeff in weights:
            co=r.combine([{lookup[a,'protect100_R0',100,0,q]:-w} for a,w in coeff.items() for q in ids])
            protection_factorial.append(dict(task=task,comparison=name,damage_count_difference_bounds=list(r.score_bounds(co,scores))))
    result.update(mechanical=mechanical,mechanical_contrasts=mechanical_contrasts,protection_contrasts=protection_factorial,hard_path_labels={a:'HARD_KL_NOT_ENFORCED' for a in ARMS},effect_scope='HR-HM changes only removed full-vocabulary soft KL weight 0 to 1 on fixed HM background; other contrasts descriptive',projection_alpha=.95)
    for arm in ACTIVE:
        result['diagnostics'][arm+'_gradients']={phase:c.read(R/'public'/('GRADIENT_'+phase+'_'+arm+'.json')) for phase in ('INITIAL','FINAL')}
        result['diagnostics'][arm+'_projection']=[dict(slot=e['slot'],values=e.get('complete_source_after_slot')) for e in sorted([c.read(p) for p in (R/'private/steps'/arm/'1').glob('*.json')],key=lambda x:x['slot'])]
    result['scientific_success']=not unknown and gen['full_protocol_complete'] and all(gates['HR'].values())
    result['diagnostics']['HR_evidence']={phase:c.read(R/'public'/('EVIDENCE_'+phase+'_HR.json')) for phase in ('INITIAL','FINAL')}
    mechanism={}
    for arm in ARMS:
        source=c.read(roots[arm]/'public'/('SOURCE_'+arm+'_1.json'))
        margin=roots[arm]/'public'/('MARGIN_DIAGNOSTIC_'+arm+'.json')
        mechanism[arm]=dict(actual_removed_KL=source['final'].get('scope_KL'),margin=c.read(margin) if margin.exists() else None)
    result['mechanism_endpoints']=mechanism
    result['HR_minus_HM_removed_KL']=mechanism['HR']['actual_removed_KL']-mechanism['HM']['actual_removed_KL']
    hr=gates['HR'];lower_KL=result['HR_minus_HM_removed_KL']<0
    if result['scientific_success']:decision='JOINT_GATES_PASS_NO_AUTOMATIC_EXPANSION'
    elif not hr['full_known_scoring']:decision='INCOMPLETE_SCORING_NO_SUCCESS'
    elif hr['all_source_CE_below_point1'] and hr['ten_medical_margins_positive_finite'] and not hr['zero_individual_Base_damage']:decision='SOURCE_GATES_PASS_EXTERNAL_PROTECTION_FAIL'
    elif not hr['all_source_CE_below_point1'] and hr['ten_medical_margins_positive_finite']:decision='EVIDENCE_FIT_TRADEOFF_REMAINS'
    elif lower_KL and not hr['ten_medical_margins_positive_finite']:decision='REMOVED_KL_REDUCTION_INSUFFICIENT_FOR_EVIDENCE'
    elif not lower_KL:decision='FIXED_SOFT_POLICY_DID_NOT_LOWER_REMOVED_KL'
    else:decision='JOINT_GATES_FAIL'
    result['analysis']=dict(decision=decision,removed_KL_lower_than_HM=lower_KL,
        failed_gates=[k for k,v in hr.items() if not v],
        scope='Single-source development contrast only; no unique causal attribution or clinical extrapolation',
        next_action='STOP_AFTER_DELIVERY_NO_RETRY_NO_TUNING_NO_NEXT_ARM')
    c.write(R/'public/RESULTS.json',result)
    text='# HR 首源 removed 软保护单因素对照\n\n'+status+'。完整评分不等科学成功；固定36保护与prefix1任务面板分开。\n\n|臂|任务|正确/分母|未知|微平均界%|\n|---|---|---:|---:|---|\n'
    for x in panels:text+='|'+ '|'.join(map(str,[x['arm'],x['task'],str(x['known_correct'])+'/'+str(x['observations']),x['missing'],x['micro_bounds']]))+'|\n'
    text+='\n跨臂T2G逐题转移：'+json.dumps(cross_T2G,ensure_ascii=False)+'\n\n描述性形状与removed软保护交互：'+json.dumps(result['target_shape_protection_interaction'],ensure_ascii=False)+'\n'
    text+='\n固定保护：'+json.dumps(protection,ensure_ascii=False)+'\n\n逐题新增损伤/修复及实际插入端点：'+json.dumps(endpoints,ensure_ascii=False)+'\n'
    text+='\n共同完整CE/接受拒绝zero：'+json.dumps(mechanical,ensure_ascii=False)+'\n\nT2G共享缺分界：'+json.dumps(contrasts,ensure_ascii=False)+'\n'
    text+='\nHR-HM仅改变removed完整词表forward KL权重0到1；HR-SH10及交互仅作开发解释。软KL并非硬约束，正间隔不保证外部零损伤。单源总体区间NA_SINGLE_SOURCE；无自动146或下一轮。原146/M暂停、O/L等待。\n'
    text+='\n机制终点：'+json.dumps(mechanism,ensure_ascii=False)+'；HR-HM实际removed KL差='+str(result['HR_minus_HM_removed_KL'])+'\n'
    text+='\n冻结规则分析：'+json.dumps(result['analysis'],ensure_ascii=False)+'\n'
    text+='\n实际成本：'+json.dumps(result['actual_counts'])+'；GPU进程小时='+str(result['GPU_process_hours'])+'；新Judge='+str(result['new_Judge'])+'。六个历史臂不重跑不补Judge，永久缺分/未生成/未尝试分开。\n'
    (R/'public/REPORT_ZH.md').write_text(text);c.write(marker,dict(status=status,epoch=time.time()));print(status)

if __name__=='__main__':main()
