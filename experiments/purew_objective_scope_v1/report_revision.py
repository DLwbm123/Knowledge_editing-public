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
ACTIVE=tuple(c.read(R/'private/FROZEN_PROTOCOL.json').get('active_arms',('N','J','NJ')))
assert ACTIVE==('SH00','SH10','SH01')
ARMS=('SH11',*ACTIVE);CONTRASTS=(('SH10','SH00'),('SH11','SH01'),('SH01','SH00'),('SH11','SH10'),*((a,'SH11') for a in ACTIVE))

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
    files=[('SH11',p,None) for p in (old/'private/outputs/R').glob('*/*/*/*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
    assert len(files)==87
    files.extend((x['method'],Path(x['path']),x['output_binding']) for x in rows)
    for arm,path,locked in files:
        d=c.read(path)
        if locked:assert c.digest(d)==locked
        bb=d['binding'];phase=bb['phase'];query=ledger['queries'][bb['input']['query_id']];assert bb['input']==query
        key,full=binding_key(query,bb['judge_input'],d['R0'],epoch)
        if key in payloads:assert json.loads(payloads[key]['binding'])==full;status=payloads[key]['status']
        else:assert arm=='SH11';scores[key]=None;status='UNKNOWN_UNATTEMPTED'
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
        prefix1.append(dict(comparison=a+'-'+b,prefix=1,secondary_only=False,status='ESTIMATED',negative_edit_bounds=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),delta_macro_bounds_pp=[100*x for x in bounds],**ar.bootstrap(delta,scores,groups)))
    contrasts=prefix1
    # Combine shared payload variables before bounding, including four-way interactions.
    interaction_edits={e:r.combine([coefs['SH11',1,'T2G'][e],{k:-v for k,v in coefs['SH10',1,'T2G'][e].items()},{k:-v for k,v in coefs['SH01',1,'T2G'][e].items()},coefs['SH00',1,'T2G'][e]]) for e in coefs['SH00',1,'T2G']}
    interaction=dict(comparison='SH11-SH10-SH01+SH00',delta_macro_bounds_pp=[100*x for x in r.score_bounds(r.combine([{k:v/len(interaction_edits) for k,v in x.items()} for x in interaction_edits.values()]),scores)],**ar.bootstrap(interaction_edits,scores,groups))
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
        gates[arm]=dict(zero_individual_Base_damage=protected,complete_first_source=(arm=='SH11' or terminals[arm]['slots']==20),T0_insertion_and_terminal=all(scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]]==scores[lookup[arm,'bank_R0',terminal,0,t['edit_id']]]==1 for t in tasks[:terminal]),T1G_full_retention=t1['known_correct']==t1['observations'],full_known_scoring=all(scores[k] is not None for k in keys))
        if arm!='SH11':
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
            matched=[x for x in contrasts if x['comparison']==arm+'-SH11']
            gates[arm].update(all_source_CE_below_point1=terminal==1 and all(x['measurements']['final']['weighted_all_FIT_CE']<.1 for x in ls),per_edit_T2G_not_worse=bool(matched) and all(x['status']=='ESTIMATED' and x['negative_edit_bounds']==0 for x in matched))
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks[:terminal]:
                ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]
                pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',terminal,0,q]) for q in dict.fromkeys(ids))
            endpoints.append(dict(arm=arm,task=task,comparison='actual-insertion-to-actual-terminal',terminal_prefix=terminal,**transitions(pairs,scores)))
    unknown=sum(sum(v for k,v in x.items() if k in ('MISSING','UNKNOWN_UNATTEMPTED')) for x in coverage.values())
    status='PARTIAL_SCIENTIFIC_STOP' if not gen['full_protocol_complete'] else 'COMPLETE_WITH_UNKNOWN' if unknown else 'COMPLETE'
    result=dict(status=status,scientific_success=not unknown and gen['full_protocol_complete'] and all(all(gates[a].values()) for a in ARMS[1:]),full_protocol_complete=gen['full_protocol_complete'],actual_new_consumer_count=len(rows),historical_SH11_external_consumers=87,all_consumers=len(files),coverage={a:dict(v) for a,v in coverage.items()},terminals=terminals,panels=panels,contrasts=contrasts,prefix1_secondary=prefix1,protection=protection,endpoints=endpoints,gates=gates,diagnostics=diagnostics,unknown_consumers=unknown,new_Judge=cost['Judge_attempts'],GPU_process_hours=cost['gpu_seconds_used']/3600,actual_counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ('forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts','active_set_checks','candidate_checks')},owned_matrix_peak_bytes=cost.get('weights_observed_peak_bytes'),owned_GPU_peak_MiB=cost.get('GPU_memory_observed_peak_MiB'),failures=[dict(kind=p.stem,epoch=c.read(p).get('epoch')) for p in (R/'private/failures').glob('*.json')],old_R_missing_not_rejudged=True,automatic146=False,K_matched_first3='NA_NO_REBUILD',nonlinear_clinical_privacy_certificate=False,calibration_coverage='Eight CT/MRI source groups; no KG/XRay; patient independence unknown')
    mechanical={}
    for arm in ARMS:
        base=old if arm=='SH11' else R;internal='R' if arm=='SH11' else arm
        source=c.read(base/'public'/('SOURCE_'+internal+'_1.json'))
        events=sorted([c.read(p) for p in (base/'private/steps'/internal/'1').glob('*.json')],key=lambda x:x['slot'])
        mechanical[arm]=dict(final_complete_CE=source['final']['weighted_all_FIT_CE'] if len(events)==20 else None,last_available_CE=source['final']['weighted_all_FIT_CE'],slots=len(events),accepted=sum(e['accepted'] for e in events),rejected=sum(not e['accepted'] for e in events),zero_fraction=sum(e['zero_native'] for e in events)/len(events),first_hit_CE={str(t):next((e['slot'] for e in events if e.get('complete_source_after_slot',{}).get('weighted_all_FIT_CE',float('inf'))<t),None) for t in (1,.5,.1)} if arm!='SH11' else 'NA_HISTORICAL_NOT_MEASURED')
    weights=[('SH10-SH00',{'SH10':1,'SH00':-1}),('SH11-SH01',{'SH11':1,'SH01':-1}),('SH01-SH00',{'SH01':1,'SH00':-1}),('SH11-SH10',{'SH11':1,'SH10':-1}),('SH11-SH10-SH01+SH00',{'SH11':1,'SH10':-1,'SH01':-1,'SH00':1})]
    mechanical_contrasts={name:{metric:(sum(w*mechanical[a][metric] for a,w in coeff.items()) if all(mechanical[a][metric] is not None for a in coeff) else None) for metric in ('final_complete_CE','accepted','zero_fraction')} for name,coeff in weights}
    # Fixed protection correctness uses the same key coefficients; loss count reverses their sign.
    protection_factorial=[]
    for task in ('T1L','T2L'):
        ids=protection_ids(tasks,ledger,task)
        for name,coeff in weights:
            co=r.combine([{lookup[a,'protect100_R0',100,0,q]:-w} for a,w in coeff.items() for q in ids])
            protection_factorial.append(dict(task=task,comparison=name,damage_count_difference_bounds=list(r.score_bounds(co,scores))))
    result.update(interaction=interaction,mechanical=mechanical,mechanical_contrasts=mechanical_contrasts,protection_factorial=protection_factorial,hard_path_labels={'SH00':'HARD_KL_NOT_ENFORCED','SH10':'HARD_KL_NOT_ENFORCED','SH01':'HARD_KL_ENFORCED','SH11':'HISTORICAL_R_HARD_KL_ENFORCED'},effect_scope='Conditional on Base16 and removed4 Fisher; H combines direction and actual native gate, no post-check-only causal claim',new_projection_objective_tested=False)
    c.write(R/'public/RESULTS.json',result)
    text='# removed软KL与实际硬约束路径首源20槽对照\n\n'+status+'。SH11只读复用R；其他三臂独立从Base开始。缺分不补判，未知不通过逐题零损伤。\n\n|臂|任务|正确/分母|未知|微平均界%|\n|---|---|---:|---:|---|\n'
    for x in panels:text+='|'+ '|'.join(map(str,[x['arm'],x['task'],str(x['known_correct'])+'/'+str(x['observations']),x['missing'],x['micro_bounds']]))+'|\n'
    text+='\n主机械指标为共同的完整native+allFIT CE。各臂CE/接受拒绝与零比例：'+json.dumps(mechanical,ensure_ascii=False)+'\n\n四个条件效应及交互：'+json.dumps(mechanical_contrasts,ensure_ascii=False)+'\n'
    text+='\nT2G共享缺分交互：'+json.dumps(interaction,ensure_ascii=False)+'。逐题保护及相对损伤/修复、真实插入端点与固定34T2L+2T1L见RESULTS.json。所有界先合并同一payload变量，未假设缺分独立。单来源10000次配对重采样不是总体推广证据。\n'
    text+='\nH0仍有Base16/removed4 Fisher保护，但不执行五视图真实KL方向和硬门，明确HARD_KL_NOT_ENFORCED。H1保留原阈值.01和R求解器。H是整个实际约束路径因子，不能只归因末端硬门。新信息投影目标未测试；有限Fisher、粗框及非线性误差不构成临床或隐私保证。原146/M暂停，O/L等待，不自动扩展。\n'
    text+='\n实际调用成本：'+json.dumps(result['actual_counts'])+'；GPU进程小时='+str(result['GPU_process_hours'])+'；新Judge='+str(result['new_Judge'])+'。科学止步、永久缺分与未生成分开记录。\n'
    (R/'public/REPORT_ZH.md').write_text(text);c.write(marker,dict(status=status,epoch=time.time()));print(status)

if __name__=='__main__':main()
