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
assert ACTIVE==('R',)
ARMS=('J','R');CONTRASTS=(('R','J'),)

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
    files=[('J',p,None) for p in (old/'private/outputs/J').glob('*/*/*/*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
    assert len(files)==87
    files.extend((x['method'],Path(x['path']),x['output_binding']) for x in rows)
    for arm,path,locked in files:
        d=c.read(path)
        if locked:assert c.digest(d)==locked
        bb=d['binding'];phase=bb['phase'];query=ledger['queries'][bb['input']['query_id']];assert bb['input']==query
        key,full=binding_key(query,bb['judge_input'],d['R0'],epoch)
        if key in payloads:assert json.loads(payloads[key]['binding'])==full;status=payloads[key]['status']
        else:assert arm=='J';scores[key]=None;status='UNKNOWN_UNATTEMPTED'
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
        gates[arm]=dict(zero_individual_Base_damage=protected,complete_first_source=terminal==1,T0_insertion_and_terminal=all(scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]]==scores[lookup[arm,'bank_R0',terminal,0,t['edit_id']]]==1 for t in tasks[:terminal]),T1G_full_retention=t1['known_correct']==t1['observations'],full_known_scoring=all(scores[k] is not None for k in keys))
        if arm!='J':
            ls=[]
            for owner in range(1,terminal+1):
                es=sorted([c.read(p) for p in (R/'private/steps'/arm/str(owner)).glob('*.json')],key=lambda e:e['slot']);assert len(es)==20
                source=c.read(R/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'))
                flags=Counter(z for e in es if not e['accepted'] for z in set([e['rejection_reason']] if e.get('rejection_reason') else [v for a in e['candidates'] for v in a['reasons']]))
                ls.append(dict(source_position=owner,slots=20,accepted=sum(e['accepted'] for e in es),zero_fraction=sum(e['zero_native'] for e in es)/20,rejected_slot_flags=dict(flags),measurements=source,final_path_and_net=c.read(R/'public'/('PATH_'+arm+'_'+str(owner)+'.json')),last_solver={k:es[-1].get(k) for k in ('multipliers','predicted_risks','current_actual_KLs','current_linear_predicted_KLs','projected_KKT_residual')},last_candidate_KLs=[a.get('current_removed_KLs') for a in es[-1]['candidates']]))
            diagnostics[arm]=ls
            m=R/'public'/('MARGIN_DIAGNOSTIC_'+arm+'.json')
            if m.exists():diagnostics[arm+'_margin']=c.read(m)
            gates[arm]['ten_medical_margins_positive_finite']=m.exists() and c.read(m)['finite']==c.read(m)['positive']==10
            diagnostics[arm+'_solver']=dict(fallback_slots=sum('fallback' in e for e in es),fallback_certified=sum(e.get('solver_path')=='R_FINITE_ACTIVE_SET' for e in es),fallback_status_counts=dict(Counter(e['fallback']['status'] for e in es if 'fallback' in e)),active_set_checks=sum(e.get('fallback',{}).get('active_sets_checked',0) for e in es),candidate_checks=sum(len(e['candidates']) for e in es),legacy_rejects_resolved=sum(e.get('solver_path')=='R_FINITE_ACTIVE_SET' and e['accepted'] for e in es),fallback_certificates=[dict(slot=e['slot'],**e['fallback']) for e in es if 'fallback' in e])
            matched=[x for x in contrasts if x['comparison'].startswith(arm+'-')]
            gates[arm].update(all_source_CE_below_point1=terminal==1 and all(x['measurements']['final']['weighted_all_FIT_CE']<.1 for x in ls),per_edit_T2G_not_worse=bool(matched) and all(x['status']=='ESTIMATED' and x['negative_edit_bounds']==0 for x in matched))
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks[:terminal]:
                ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]
                pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',terminal,0,q]) for q in dict.fromkeys(ids))
            endpoints.append(dict(arm=arm,task=task,comparison='actual-insertion-to-actual-terminal',terminal_prefix=terminal,**transitions(pairs,scores)))
    unknown=sum(sum(v for k,v in x.items() if k in ('MISSING','UNKNOWN_UNATTEMPTED')) for x in coverage.values())
    status='PARTIAL_SCIENTIFIC_STOP' if not gen['full_protocol_complete'] else 'COMPLETE_WITH_UNKNOWN' if unknown else 'COMPLETE'
    result=dict(status=status,scientific_success=not unknown and gen['full_protocol_complete'] and all(all(gates[a].values()) for a in ARMS[1:]),full_protocol_complete=gen['full_protocol_complete'],actual_new_consumer_count=len(rows),historical_J_external_consumers=87,all_consumers=len(files),coverage={a:dict(v) for a,v in coverage.items()},terminals=terminals,panels=panels,contrasts=contrasts,prefix1_secondary=prefix1,protection=protection,endpoints=endpoints,gates=gates,diagnostics=diagnostics,unknown_consumers=unknown,new_Judge=cost['Judge_attempts'],GPU_process_hours=cost['gpu_seconds_used']/3600,actual_counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ('forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts','active_set_checks','candidate_checks')},owned_matrix_peak_bytes=cost.get('weights_observed_peak_bytes'),owned_GPU_peak_MiB=cost.get('GPU_memory_observed_peak_MiB'),failures=[dict(kind=p.stem,epoch=c.read(p).get('epoch')) for p in (R/'private/failures').glob('*.json')],old_J_missing_not_rejudged=True,automatic146=False,K_matched_first3='NA_NO_REBUILD',nonlinear_clinical_privacy_certificate=False,calibration_coverage='Eight CT/MRI source groups; no KG/XRay; patient independence unknown')
    c.write(R/'public/RESULTS.json',result)
    text='# R与J首源20槽有限对照\n\n'+status+'。全部实际消费者终态不等于科学成功；J的18个永久缺分保持共享绑定，不补判。\n\n|臂|任务|正确/分母|未知|微平均界%|\n|---|---|---:|---:|---|\n'
    for x in panels:
        text+='|'+ '|'.join(map(str,[x['arm'],x['task'],str(x['known_correct'])+'/'+str(x['observations']),x['missing'],x['micro_bounds']]))+'|\n'
    text+='\nR-J的T2G宏平均精确共享缺分差值界、10000次seed20260912配对edit/source区间、34个T2L与2个T1L逐题损伤/修复及真实插入端点见RESULTS.json。只有一个来源，重采样区间不能作为总体推广证据。未知不通过零新增损伤门，平均提升不能抵消逐题损伤。\n\nR仅在原J有限对偶拒绝后尝试五仿射约束有限主动集证书；原PASS路径、每块.01和实际五视图KL硬门不变。证书限于浮点代理问题；不能据未获得证书断言真实问题无可行方向。有限Base覆盖、Fisher采样、Jacobian漂移和非线性误差均无全局保证；粗框不是医生反事实，病例不重放也不构成隐私保证。不会扩至源2或146。原146/M暂停，O/L等待。\n'
    text+='\n实际成本：'+json.dumps(result['actual_counts'],ensure_ascii=False)+'；GPU进程小时='+str(result['GPU_process_hours'])+'；新Judge尝试='+str(result['new_Judge'])+'。源前后完整CE、十间隔门、接受拒绝/零更新、量化/路径和所有已记录失败见匿名JSON。\n'
    (R/'public/REPORT_ZH.md').write_text(text);c.write(marker,dict(status=status,epoch=time.time()));print(status)

if __name__=='__main__':main()
