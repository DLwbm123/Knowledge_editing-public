"""Frozen-support evaluation of initial curriculum and consolidation ablation."""
from collections import Counter,defaultdict
import sqlite3
import statistics
import functionmatch as e
import function_queue as queue
import astra_report as ar
from replay_report import summary,transitions
from scoped_report import contrast

c,p,RUN,r=e.c,e.p,e.RUN,ar.r
TASKS=('T0','T1G','T2G','T1L','T2L')
PAIRS=[('RAW_A_SCHEDULE','W0'),('RAW_DIR_540','ADAM_R_SCHEDULE'),('RAW_A_SCHEDULE','RAW_DIR_540'),('ADAM_R_SCHEDULE','W0')]


def main():
    assert (queue.q.ROOT/'ALL_WORKERS_COMPLETE.json').exists();ar.selfcheck()
    db=sqlite3.connect('file:'+str(queue.q.ROOT/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    assert not db.execute("SELECT 1 FROM payload WHERE status NOT IN ('FORMAT_VALID','MISSING')").fetchone()
    records=[];mapping={}
    for row in db.execute('SELECT * FROM consumer'):
        d=c.read(row['path']);assert c.digest(d)==row['output_binding']
        d=dict(d,binding=dict(d['binding'],arm=row['method'],mode=row['mode'],phase=dict(d['binding']['phase'],arm=row['method'],node=0,prefix=146)))
        records.append((d,row['payload_key']));mapping[row['method'],row['mode'],row['query_id']]=d,row['payload_key']
    lock=c.read(RUN/'private/DIRECTION_LOCK.json');assert len(records)==lock['consumers']==7012
    routes=c.read(RUN/'private/ROUTES.json');chosen=set(lock['selected_edits']);main_queries=p.queries(146)
    affected={q for q in main_queries if routes[q]['effective_expert'] in chosen}
    scopes={'full':set(main_queries),'affected':affected,'unaffected':set(main_queries)-affected}
    panels=[];coeff={};groups={};paired=[]
    for scope,qs in scopes.items():
        subset=[(d,k) for d,k in records if d['binding']['input']['query_id'] in qs and d['binding']['mode']=='bank_R0']
        for label in e.LABELS:
            for task in TASKS:
                m,co,g=r.panel(subset,scores,scope,label,0,'bank_R0',task,p.tasks(),146)
                m.update(ar.bootstrap(co,scores,g));m['support_scope']=scope;panels.append(m)
                coeff[scope,label,task]=co;groups.update(g)
        for a,b in PAIRS:
            for task in TASKS:paired.append(dict(scope=scope,a=a,b=b,task=task,**contrast(coeff[scope,a,task],coeff[scope,b,task],scores,groups)))
    old=c.read(e.PARENT/'public/RESULTS.json')
    for task in TASKS:
        now=next(x for x in panels if x['support_scope']=='full' and x['arm']=='W0' and x['task']==task)
        prior=next(x for x in old['panels'] if x['support_scope']=='full' and x['arm']=='W0' and x['task']==task)
        assert all(now[k]==prior[k] for k in ('macro_bounds','observations','known_correct','missing','edit_units'))
    source_rows=c.read(RUN/'private/SOURCE_CHECK_ROWS.json');checks=[];check_co={};source_groups={}
    for mode in ('forced','natural'):
        keymode=mode+'_source_CHECK';base_label='BASE' if mode=='forced' else 'BASE_NAT'
        base={q['query_id']:scores[mapping[base_label,keymode,q['query_id']][1]] for q in source_rows}
        assert all(v in (0,1) for v in base.values())
        labels=e.LABELS
        for scope in ('all','updated','unchanged'):
            qs=[q for q in source_rows if scope=='all' or
                ((p.tasks()[q['forced_expert_index']]['edit_id'] in chosen if mode=='forced' else routes[q['query_id']]['effective_expert'] in chosen)==(scope=='updated'))]
            for label in labels:
                actual=label if mode=='forced' else label+'_NAT';kept=[q for q in qs if base[q['query_id']]==1]
                values={q['query_id']:scores[mapping[actual,keymode,q['query_id']][1]] for q in qs}
                checks.append(dict(mode=mode,scope=scope,arm=label,
                    accuracy=summary([(q['source_group'],values[q['query_id']]) for q in qs]),
                    retention=summary([(q['source_group'],values[q['query_id']]) for q in kept]) if kept else None,
                    Base_correct_count=len(kept),transitions_from_Base=transitions([(base[q['query_id']],values[q['query_id']]) for q in qs]),
                    by_answer_type={kind:summary([(q['source_group'],values[q['query_id']]) for q in qs if q['answer_kind']==kind]) for kind in ('yes','no','open') if any(q['answer_kind']==kind for q in qs)},
                    exact_Base_tokens=sum(mapping[actual,keymode,q['query_id']][0]['R0']['raw_token_ids']==mapping[base_label,keymode,q['query_id']][0]['R0']['raw_token_ids'] for q in qs)/len(qs)))
                for metric,subset in [('accuracy',qs),('retention',kept)]:
                    group=defaultdict(list)
                    for q in subset:group[q['source_group']].append(mapping[actual,keymode,q['query_id']][1])
                    co={g:{k:n/len(ks) for k,n in Counter(ks).items()} for g,ks in group.items()}
                    check_co[mode,scope,label,metric]=co;source_groups.update({g:g for g in co})
            for a,b in PAIRS:
                if b not in labels:continue
                for metric in ('accuracy','retention'):
                    ka=(mode,scope,a,metric);kb=(mode,scope,b,metric)
                    paired.append(dict(scope=mode+'_'+scope,a=a,b=b,task=metric,**contrast(check_co[ka],check_co[kb],scores,source_groups)))
    training=[]
    for t in e.selected():
        for arm in e.ARMS:
            d=c.read(e.point(t,arm).parent/'TRAINING.json')
            assert d['status']=='COMPLETE' and d['steps']==320 and d['save_load_exact'] and not d['Base_gradient']
            training.append(dict(expert_order=t['order'],arm=arm,steps=d['steps'],forwards=d['forwards'],backwards=d['backwards'],
                diagnostic_forwards=d['diagnostic_forwards'],final=d['final'],curve=d['curve']))
    parity=c.read(RUN/'private/ENDPOINT_PARITY.json');assert parity['status']=='PASS' and parity['exact_all_cores']
    steps=sum(x['steps'] for x in training)+parity['extra_steps'];assert steps==10240
    forwards=sum(x['forwards'] for x in training)+parity['extra_forwards'];assert forwards==20480
    backwards=sum(x['backwards'] for x in training)+parity['extra_backwards'];assert backwards==20480
    def delta(scope,task):return next(x for x in paired if x['a']==e.PRIMARY and x['b']=='W0' and x['scope']==scope and x['task']==task)
    edits={t:delta('affected',t) for t in ('T0','T1G','T2G')}
    protection={t:delta('forced_updated',t) for t in ('accuracy','retention')}
    safe=all(edits[t]['delta_bounds_pp'][0] is not None and edits[t]['delta_bounds_pp'][0]>=floor for t,floor in [('T0',0),('T1G',0),('T2G',-1)])
    protection_safe=all(x['delta_bounds_pp'][0] is not None and x['delta_bounds_pp'][0]>=0 for x in protection.values())
    gain=edits['T2G']['delta_bounds_pp'][0] is not None and edits['T2G']['delta_bounds_pp'][0]>0
    ci_gain=edits['T2G'].get('edit_ci') is not None and edits['T2G']['edit_ci'][0]>0
    ci_protection=all(x.get('source_ci') is not None and x['source_ci'][0]>=0 for x in protection.values())
    if not safe:decision='FAIL_EDIT_PRESERVATION_SCREEN'
    elif not protection_safe:decision='FAIL_PROTECTION_SCREEN'
    elif not gain:decision='NO_GENERALIZATION_GAIN'
    elif not (ci_gain and ci_protection):decision='INCONCLUSIVE_POSITIVE_POINT_ONLY'
    else:decision='DEVELOPMENT_GENERALIZATION_SIGNAL_NOT_CONFIRMED'
    lengths=[]
    for mode,qs in [('bank_R0',affected),('forced_source_CHECK',{q['query_id'] for q in source_rows if p.tasks()[q['forced_expert_index']]['edit_id'] in chosen})]:
        for label in e.LABELS:
            outputs=[mapping[label,mode,q][0]['R0']['raw_token_ids'] for q in qs]
            lengths.append(dict(mode=mode,arm=label,n=len(outputs),at1024=sum(len(x)==1024 for x in outputs),median_tokens=statistics.median(map(len,outputs)),
                exact_W0_tokens=sum(mapping[label,mode,q][0]['R0']['raw_token_ids']==mapping['W0',mode,q][0]['R0']['raw_token_ids'] for q in qs)))
    cap_safe=all(next(x['at1024'] for x in lengths if x['mode']==mode and x['arm']==e.PRIMARY)<=next(x['at1024'] for x in lengths if x['mode']==mode and x['arm']=='W0') for mode in ('bank_R0','forced_source_CHECK'))
    primary_pass=decision=='DEVELOPMENT_GENERALIZATION_SIGNAL_NOT_CONFIRMED' and cap_safe
    if not cap_safe:decision='FAIL_GENERATION_CAP_SCREEN'
    reverse_edits={t:next(x for x in paired if x['a']=='RAW_DIR_540' and x['b']=='ADAM_R_SCHEDULE' and x['scope']=='affected' and x['task']==t) for t in ('T0','T1G','T2G')}
    reverse_protection={t:next(x for x in paired if x['a']=='RAW_DIR_540' and x['b']=='ADAM_R_SCHEDULE' and x['scope']=='forced_updated' and x['task']==t) for t in ('accuracy','retention')}
    reverse_pass=all(x['delta_bounds_pp'][0] is not None and x['delta_bounds_pp'][0]>=0 for x in reverse_edits.values()) and reverse_edits['T2G'].get('edit_ci') is not None and reverse_edits['T2G']['edit_ci'][0]>0 and all(x['delta_bounds_pp'][0] is not None and x['delta_bounds_pp'][0]>=0 and x.get('source_ci') is not None and x['source_ci'][0]>=0 for x in reverse_protection.values())
    matches=[z['function_match'] for x in training for z in x['curve']]
    assert len(matches)==5120 and all(abs(x['actual']-x['target'])<=e.fm.ATOL+e.fm.RTOL*x['target'] for x in matches)
    reference_audits=[c.read(e.schedule(t,mode)) for t in e.selected() for mode in ('ADAM','RAW')]
    assert len(reference_audits)==16 and all(x['endpoint_exact'] for x in reference_audits)
    c.write(RUN/'public/FUNCTION_SCHEDULES.json',[dict(expert_order=x['expert_order'],mode=x['mode'],endpoint_exact=True,norms=[z['actual_map_norm'] for z in x['curve']]) for x in reference_audits])
    function_audit=dict(status='PASS',matched_steps=len(matches),max_relative_error=max(x['relative_error'] for x in matches),scale_min=min(x['scale'] for x in matches),scale_max=max(x['scale'] for x in matches),scale_median=statistics.median(x['scale'] for x in matches),reference_endpoints_exact=16)
    c.write(RUN/'public/FUNCTION_MATCH_AUDIT.json',function_audit)
    resource=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json');assert all(x.get('ended_epoch') for x in resource['gpu_sessions'])
    new=dict(db.execute('SELECT p.status,count(*) FROM payload p LEFT JOIN inherited i ON p.key=i.key WHERE i.key IS NULL GROUP BY p.status').fetchall())
    missing=sum(v is None for v in scores.values())
    result=dict(status='TERMINAL_WITH_MISSING' if missing else 'COMPLETE',panels=panels,source_CHECK=checks,contrasts=paired,training=training,endpoint_parity=dict(status='PASS',exact_all_cores=True,reconstruction_steps=5120,reference_trajectories=16),
        decision=decision,main_candidate=e.PRIMARY,edit_deltas=edits,protection_deltas=protection,lengths=lengths,training_forwards=forwards,training_backwards=backwards,diagnostic_forwards=160,optimizer_steps=steps,
        scoring=dict(payloads=len(scores),valid=len(scores)-missing,missing=missing,consumers=len(records),new_statuses=new),
        final_TT_retained=16,new_generations=220,replay=False,independent_confirmation=False,
        Judge_attempts=resource['Judge_attempts'],new_Judge_attempts=resource['Judge_attempts']-inherited['Judge_attempts'],
        GPU_process_hours=resource['gpu_seconds_used']/3600,new_GPU_process_hours=(resource['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600)
    result.update(function_match=function_audit,primary_screen_pass=primary_pass,reverse_direction_screen_pass=reverse_pass,
        mechanism_decision='DEVELOPMENT_DIRECTION_SIGNAL_NOT_CONFIRMED' if primary_pass and reverse_pass else 'DIRECTION_MECHANISM_NOT_ESTABLISHED',
        generation_cap_pass=cap_safe,eligible146=False)
    c.write(RUN/'public/RESULTS.json',result);c.write(RUN/'public/PAIRED_CONTRASTS.json',paired);c.write(RUN/'public/TRAINING_AUDIT.json',training)
    c.write(RUN/'public/COMPLETION_AUDIT.json',dict(status=result['status'],all_GPU_sessions_ended=True,new_generations=220,consumers=7012,scoring=result['scoring'],review_required=True,decision=decision))
    c.write(RUN/'public/RESOURCE_LEDGER.json',{k:result[k] for k in ('Judge_attempts','new_Judge_attempts','GPU_process_hours','new_GPU_process_hours')})
    lines=['# TT映射幅度序列与方向策略的固定交叉对照','', '| 条件 | 受影响T0 | 受影响T1G | 受影响T2G | 强制CHECK更新子集 |','|---|---|---|---|---|']
    fmt=lambda b:'NA' if b is None or b[0] is None else f'{b[0]:.3f}' if b[0]==b[1] else f'[{b[0]:.3f}, {b[1]:.3f}]'
    for label in e.LABELS:
        v=[fmt(next(x['macro_bounds'] for x in panels if x['support_scope']=='affected' and x['arm']==label and x['task']==t)) for t in ('T0','T1G','T2G')]
        v.append(fmt(next(x['accuracy']['macro_bounds'] for x in checks if x['mode']=='forced' and x['scope']=='updated' and x['arm']==label)))
        lines.append('| '+label+' | '+' | '.join(v)+' |')
    lines+=['',f'主候选RAW_A_SCHEDULE判定：{decision}。区间为缺失赋值界，不是置信区间。完整配对区间见RESULTS.json。','',
        f'训练{steps}步（含5120步历史端点复现），前向/反向各{forwards}次、训练激活与最终诊断160次；220条新生成，评分{len(scores)-missing}/{len(scores)}有效。',
        '固定原220状态加320步末段，交叉方向策略与冻结映射变化幅度序列；16条历史Adam/RAW末段端点精确复现通过。当前为8专家开发对照，自然受影响CHECK只有2条，不支持独立确认或医学保护保证。']
    (RUN/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n');p.done('REPORT_COMPLETE')


if __name__=='__main__':main()
