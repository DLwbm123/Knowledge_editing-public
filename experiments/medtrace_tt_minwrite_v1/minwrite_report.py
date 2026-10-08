"""Frozen supports, paired comparisons and feasibility audit for TT contraction."""
from collections import Counter,defaultdict
import sqlite3
import statistics
import minwrite as e
import minwrite_queue as queue
import astra_report as ar
from replay_report import summary,transitions
from scoped_report import contrast

c,p,RUN,r=e.c,e.p,e.RUN,ar.r
TASKS=('T0','T1G','T2G','T1L','T2L')
PAIRS=[('NATIVE_FIT',b) for b in ('W0','CE192','SCOPE_E','NATIVE')]+[('NATIVE','W0')]


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
    lock=c.read(RUN/'private/MINWRITE_LOCK.json');assert len(records)==lock['consumers']==8621
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
        labels=e.LABELS if mode=='forced' else tuple(x for x in e.LABELS if x!='CE192')
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
    calibration=[]
    for t in e.selected():
        for arm in e.ARMS:
            d=c.read(e.point(t,arm).parent/'CALIBRATION.json')
            assert d['no_optimizer_steps'] and d['fixed_cores_unchanged'] and d['zero_hook_parity']
            calibration.append(dict(expert_order=t['order'],arm=arm,status=d['status'],alpha=d['binding']['alpha'],
                initial=d['initial'],zero=d['zero'],final=d['final'],trace=d['trace'],probes=d['probes']))
    forwards=sum(c.read(RUN/'private/calibration_cost'/f"{t['order']}.json")['forwards'] for t in e.selected());assert forwards<=752
    def delta(scope,task):return next(x for x in paired if x['a']=='NATIVE_FIT' and x['b']=='W0' and x['scope']==scope and x['task']==task)
    edits={t:delta('affected',t) for t in ('T0','T1G','T2G')}
    protection={t:delta('forced_updated',t) for t in ('accuracy','retention')}
    eligible=all(x['status']!='BASELINE_INFEASIBLE' for x in calibration if x['arm']=='NATIVE_FIT')
    safe=all(edits[t]['delta_bounds_pp'][0] is not None and edits[t]['delta_bounds_pp'][0]>=floor for t,floor in [('T0',0),('T1G',0),('T2G',-1)])
    positive=any(x['delta_bounds_pp'][0] is not None and x['delta_bounds_pp'][0]>0 for x in protection.values())
    ci_safe=all(x.get('edit_ci') is not None and x['edit_ci'][0]>=(-1 if t=='T2G' else 0) for t,x in edits.items())
    ci_positive=any(x.get('source_ci') is not None and x['source_ci'][0]>0 for x in protection.values())
    if not eligible:decision='PARTIAL_BASELINE_INFEASIBLE'
    elif not safe:decision='FAIL_EDIT_PRESERVATION_SCREEN'
    elif not positive:decision='NO_PROTECTION_GAIN'
    elif not (ci_safe and ci_positive):decision='INCONCLUSIVE_POSITIVE_POINT_ONLY'
    else:decision='DEVELOPMENT_MECHANISM_SIGNAL_NOT_CONFIRMED'
    lengths=[]
    for mode,qs in [('bank_R0',affected),('forced_source_CHECK',{q['query_id'] for q in source_rows if p.tasks()[q['forced_expert_index']]['edit_id'] in chosen})]:
        for label in e.LABELS:
            outputs=[mapping[label,mode,q][0]['R0']['raw_token_ids'] for q in qs]
            lengths.append(dict(mode=mode,arm=label,n=len(outputs),at1024=sum(len(x)==1024 for x in outputs),median_tokens=statistics.median(map(len,outputs)),
                exact_W0_tokens=sum(mapping[label,mode,q][0]['R0']['raw_token_ids']==mapping['W0',mode,q][0]['R0']['raw_token_ids'] for q in qs)))
    resource=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json');assert all(x.get('ended_epoch') for x in resource['gpu_sessions'])
    new=dict(db.execute('SELECT p.status,count(*) FROM payload p LEFT JOIN inherited i ON p.key=i.key WHERE i.key IS NULL GROUP BY p.status').fetchall())
    missing=sum(v is None for v in scores.values())
    result=dict(status='TERMINAL_WITH_MISSING' if missing else 'COMPLETE',panels=panels,source_CHECK=checks,contrasts=paired,calibration=calibration,
        decision=decision,main_candidate='NATIVE_FIT',edit_deltas=edits,protection_deltas=protection,lengths=lengths,calibration_forwards=forwards,optimizer_steps=0,
        scoring=dict(payloads=len(scores),valid=len(scores)-missing,missing=missing,consumers=len(records),new_statuses=new),
        final_TT_retained=16,new_generations=220,replay=False,independent_confirmation=False,
        Judge_attempts=resource['Judge_attempts'],new_Judge_attempts=resource['Judge_attempts']-inherited['Judge_attempts'],
        GPU_process_hours=resource['gpu_seconds_used']/3600,new_GPU_process_hours=(resource['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600)
    c.write(RUN/'public/RESULTS.json',result);c.write(RUN/'public/PAIRED_CONTRASTS.json',paired);c.write(RUN/'public/CALIBRATION_AUDIT.json',calibration)
    c.write(RUN/'public/COMPLETION_AUDIT.json',dict(status=result['status'],all_GPU_sessions_ended=True,new_generations=220,consumers=8621,scoring=result['scoring'],review_required=True,decision=decision))
    c.write(RUN/'public/RESOURCE_LEDGER.json',{k:result[k] for k in ('Judge_attempts','new_Judge_attempts','GPU_process_hours','new_GPU_process_hours')})
    lines=['# 无回放TT写入收缩：8专家机制对照','', '| 条件 | 受影响T0 | 受影响T1G | 受影响T2G | 强制CHECK更新子集 |','|---|---|---|---|---|']
    fmt=lambda b:'NA' if b is None or b[0] is None else f'{b[0]:.3f}' if b[0]==b[1] else f'[{b[0]:.3f}, {b[1]:.3f}]'
    for label in e.LABELS:
        v=[fmt(next(x['macro_bounds'] for x in panels if x['support_scope']=='affected' and x['arm']==label and x['task']==t)) for t in ('T0','T1G','T2G')]
        v.append(fmt(next(x['accuracy']['macro_bounds'] for x in checks if x['mode']=='forced' and x['scope']=='updated' and x['arm']==label)))
        lines.append('| '+label+' | '+' | '.join(v)+' |')
    lines+=['',f'主候选NATIVE_FIT判定：{decision}。区间为缺失赋值界，不是置信区间。完整配对区间见RESULTS.json。','',
        f'校准前向{forwards}，无梯度优化步骤；220条新生成，评分{len(scores)-missing}/{len(scores)}有效，历史及本次缺失保持可追溯。',
        '幅度只根据native/FIT确定，未使用CHECK。当前为8专家开发对照，自然受影响CHECK只有2条，不支持独立确认或医学保护保证。']
    (RUN/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n');p.done('REPORT_COMPLETE')


if __name__=='__main__':main()
