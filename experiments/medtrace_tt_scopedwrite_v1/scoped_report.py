"""All prespecified no-replay pilot panels, paired bounds and diagnostic decisions."""
from collections import Counter,defaultdict
import sqlite3
import statistics
import scoped as exp
import scoped_queue as queue
import astra_report as ar
from replay_report import summary,transitions

c,RUN,p,r=exp.c,exp.RUN,exp.p,ar.r
TASKS=('T0','T1G','T2G','T1L','T2L')
FINAL=['W0','CE192']+[exp.label(a,192) for a in exp.ARMS]


def contrast(a,b,scores,groups):
    assert set(a)==set(b)
    co={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
    bounds=r.score_bounds(r.combine([{k:v/len(co) for k,v in x.items()} for x in co.values()]),scores) if co else [None,None]
    return dict(delta_bounds_pp=[None if x is None else 100*x for x in bounds],**ar.bootstrap(co,scores,groups))


def main():
    assert (queue.q.ROOT/'ALL_WORKERS_COMPLETE.json').exists();ar.selfcheck()
    db=sqlite3.connect('file:'+str(queue.q.ROOT/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    records=[];mapping={}
    for row in db.execute('SELECT * FROM consumer'):
        d=c.read(row['path']);assert c.digest(d)==row['output_binding']
        d=dict(d,binding=dict(d['binding'],arm=row['method'],mode=row['mode'],phase=dict(d['binding']['phase'],arm=row['method'],node=0,prefix=146)))
        records.append((d,row['payload_key']));mapping[row['method'],row['query_id']]=d,row['payload_key']
    lock=c.read(RUN/'private/SCOPED_LOCK.json');assert len(records)==lock['consumers']
    routes=c.read(RUN/'private/ROUTES.json');chosen=set(lock['selected_edits']);main_queries=p.queries(146)
    affected={q for q in main_queries if routes[q]['effective_expert'] in chosen}
    scopes={'full':set(main_queries),'affected':affected,'unaffected':set(main_queries)-affected}
    panels=[];coeff={};group={}
    for scope,qs in scopes.items():
        subset=[(d,k) for d,k in records if d['binding']['input']['query_id'] in qs]
        labels=FINAL if scope!='affected' else FINAL+[exp.label(a,n) for a in exp.ARMS for n in (32,96)]+['E_local','E_shared']
        for label in labels:
            for task in TASKS:
                m,co,g=r.panel(subset,scores,scope,label,0,'bank_R0',task,p.tasks(),146)
                m.update(ar.bootstrap(co,scores,g));m['support_scope']=scope
                panels.append(m);coeff[scope,label,task]=co;group.update(g)
    # The full-bank W0 denominator remains the original 146-edit protocol.
    old=c.read(exp.SOURCE/'public/RECOVERED_RESULTS.json')
    for task in TASKS:
        current=next(x for x in panels if x['support_scope']=='full' and x['arm']=='W0' and x['task']==task)
        prior=next(x for x in old['panels'] if x['arm']=='W0' and x['task']==task)
        for field in ('macro_bounds','observations','known_correct','missing','edit_units'):assert current[field]==prior[field]
    contrasts=[]
    A,B,C,D,E=[exp.label(a,192) for a in exp.ARMS]
    pairs=[(A,'CE192'),(B,A),(C,B),(E,C),(E,D),(E,'W0')]
    for scope in scopes:
        for a,b in pairs:
            for task in TASKS:
                contrasts.append(dict(scope=scope,a=a,b=b,task=task,**contrast(coeff[scope,a,task],coeff[scope,b,task],scores,group)))
    check_rows=c.read(RUN/'private/SOURCE_CHECK_ROWS.json');checks=[];check_co={};check_groups={}
    for mode,labels in [('forced',FINAL+[exp.label(a,n) for a in exp.ARMS for n in (32,96)]+['E_local','E_shared']),
                        ('natural',['W0']+[exp.label(a,192) for a in exp.ARMS])]:
        base_label='BASE' if mode=='forced' else 'BASE_NAT'
        base={x['query_id']:scores[mapping[base_label,x['query_id']][1]] for x in check_rows}
        assert all(x in (0,1) for x in base.values())
        for label in labels:
            actual_label=label if mode=='forced' else label+'_NAT'
            available=[q for q in check_rows if (actual_label,q['query_id']) in mapping]
            for scope in ('all','updated','unchanged','ON'):
                if mode=='forced' and scope=='ON':continue
                qs=[q for q in available if scope=='all' or
                    (scope=='updated' and (p.tasks()[q['forced_expert_index']]['edit_id'] in chosen if mode=='forced' else routes[q['query_id']]['effective_expert'] in chosen)) or
                    (scope=='unchanged' and (p.tasks()[q['forced_expert_index']]['edit_id'] not in chosen if mode=='forced' else routes[q['query_id']]['effective_expert'] not in chosen)) or
                    (scope=='ON' and routes[q['query_id']]['effective_expert'] is not None)]
                if not qs:
                    checks.append(dict(mode=mode,arm=label,scope=scope,status='NO_SUPPORT',observations=0));continue
                values={q['query_id']:scores[mapping[actual_label,q['query_id']][1]] for q in qs}
                kept=[q for q in qs if base[q['query_id']]==1]
                checks.append(dict(mode=mode,arm=label,scope=scope,
                    accuracy=summary([(q['source_group'],values[q['query_id']]) for q in qs]),
                    retention=summary([(q['source_group'],values[q['query_id']]) for q in kept]) if kept else None,
                    Base_correct_count=len(kept),transitions_from_Base=transitions([(base[q['query_id']],values[q['query_id']]) for q in qs]),
                    by_answer_type={kind:summary([(q['source_group'],values[q['query_id']]) for q in qs if q['answer_kind']==kind]) for kind in ('yes','no','open') if any(q['answer_kind']==kind for q in qs)},
                    exact_Base_tokens=sum(mapping[actual_label,q['query_id']][0]['R0']['raw_token_ids']==mapping[base_label,q['query_id']][0]['R0']['raw_token_ids'] for q in qs)/len(qs),
                    route_ON=sum(routes[q['query_id']]['effective_expert'] is not None for q in qs) if mode=='natural' else len(qs)))
                for metric,subset in [('accuracy',qs),('retention',kept)]:
                    grouped=defaultdict(list)
                    for q in subset:grouped[q['source_group']].append(mapping[actual_label,q['query_id']][1])
                    co={g:{k:n/len(ks) for k,n in Counter(ks).items()} for g,ks in grouped.items()}
                    check_co[mode,scope,label,metric]=co;check_groups.update({g:g for g in co})
    for mode in ('forced','natural'):
        for scope in ('all','updated'):
            for a,b in pairs:
                for metric in ('accuracy','retention'):
                    ka=(mode,scope,a,metric);kb=(mode,scope,b,metric)
                    if ka in check_co and kb in check_co:
                        contrasts.append(dict(scope=mode+'_'+scope,a=a,b=b,task=metric,**contrast(check_co[ka],check_co[kb],scores,check_groups)))
    transitions_three={}
    for label in [A,B,C,D,E]:
        patterns=Counter()
        for qid in affected:
            values=[scores[mapping[z,qid][1]] for z in ('W0','CE192',label)]
            patterns['missing' if None in values else ''.join(str(x) for x in values)]+=1
        transitions_three[label]=dict(patterns)
    per_edit=[]
    for t in exp.selected():
        for label in FINAL:
            for task in TASKS:
                m,_,_=r.panel(records,scores,'selected_edit',label,0,'bank_R0',task,[t],146)
                per_edit.append(dict(expert_order=t['order'],**m))
    training=[]
    for t in exp.selected():
        for arm in exp.ARMS:
            d=c.read(exp.node(t,arm,192).parent/'TRAINING.json');assert d['updates']==192 and d['fixed_cores_unchanged'] and not d['Base_gradient']
            training.append(dict(expert_order=t['order'],arm=arm,curve=d['curve'],history_spectrum=d['history_spectrum'],
                unique_history_keys=d['unique_history_keys'],gate_values=d['gate_values'],key_norms=d['key_norms'],
                gate_range=max(d['gate_values'])-min(d['gate_values']),gate_signal='NEAR_CONSTANT' if max(d['gate_values'])-min(d['gate_values'])<1e-3 else 'VARIES',
                step_gate_range=.5*(max(d['gate_values'][1:])-min(d['gate_values'][1:])),
                cumulative_actual_norm=sum(x['actual_norm'] for x in d['curve']),cumulative_candidate_norm=sum(x['candidate_norm'] for x in d['curve'])))
    assert len(training)==40
    def delta(a,b,task,scope):return next(x for x in contrasts if x['a']==a and x['b']==b and x['task']==task and x['scope']==scope)
    system={task:delta(E,'W0',task,'affected') for task in ('T0','T1G','T2G')}
    t2=system['T2G'];bounds=t2['delta_bounds_pp']
    if bounds[1] is None:status='INCONCLUSIVE'
    elif bounds[1]<-1:status='FAIL_DEVELOPMENT_T2G_SCREEN'
    elif bounds[0]<-1 or t2.get('edit_ci') is None or t2['edit_ci'][0]<-1 or t2.get('source_ci') is None or t2['source_ci'][0]<-1:status='INCONCLUSIVE'
    else:status='T2G_SCREEN_ONLY_NOT_SYSTEM_SUCCESS'
    def component(a,b):
        protection={m:delta(a,b,m,'forced_updated') for m in ('accuracy','retention')}
        edits={m:delta(a,b,m,'affected') for m in ('T0','T1G','T2G')}
        if any(v['delta_bounds_pp'][1] is not None and v['delta_bounds_pp'][1]<=0 for v in protection.values()):state='NO_JOINT_PROTECTION_GAIN'
        elif any(v.get('edit_ci') is not None and v['edit_ci'][1]<0 for v in edits.values()):state='FAIL_EDIT_GENERALIZATION_DAMAGE'
        elif all(v.get('source_ci') is not None and v['source_ci'][0]>0 for v in protection.values()) and all(v['delta_bounds_pp'][0] is not None and v['delta_bounds_pp'][0]>=0 for v in edits.values()):state='DEVELOPMENT_SIGNAL_ONLY'
        else:state='INCONCLUSIVE'
        return dict(status=state,protection=protection,edit_generalization=edits)
    decisions=dict(GEOMETRY=component(C,B),GATING=component(E,D),
        SYSTEM=dict(status=status,T2G_screen=t2,other_tasks=system,protection=component(E,'W0'),automatic_promotion=False))
    if all(x['step_gate_range']<1e-3 for x in training):decisions['GATING']['status']='INCONCLUSIVE_NO_INPUT_DEPENDENT_SIGNAL'
    decisions['GATING']['constant_control_limit']='Not an exactly update-norm-matched control'
    resource=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    assert all(x.get('ended_epoch') for x in resource['gpu_sessions'])
    result=dict(status='TERMINAL_WITH_MISSING' if any(v is None for v in scores.values()) else 'COMPLETE',
        panels=panels,source_CHECK=checks,contrasts=contrasts,per_intervened_edit=per_edit,
        three_way_transition_order=['W0','CE192','new_arm'],three_way_transitions=transitions_three,decisions=decisions,
        scoring=dict(payloads=len(scores),valid=sum(v is not None for v in scores.values()),missing=sum(v is None for v in scores.values()),consumers=len(records)),
        formal_updates=7680,formal_trajectories=40,mechanical_updates=8,new_generations=lock['new_generations'],
        replay=False,final_checkpoints_retained=40,independent_confirmation=False,medical_scope_qualification=False,
        Judge_attempts=resource['Judge_attempts'],new_Judge_attempts=resource['Judge_attempts']-inherited['Judge_attempts'],
        GPU_process_hours=resource['gpu_seconds_used']/3600,new_GPU_process_hours=(resource['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
        natural_CE_reference='NOT_AVAILABLE_NO_RETRAIN',scope='eight-expert local intervention in full146 bank')
    c.write(RUN/'public/RESULTS.json',result);c.write(RUN/'public/PAIRED_CONTRASTS.json',contrasts)
    c.write(RUN/'public/GATE_AND_GEOMETRY_DIAGNOSTICS.json',training)
    c.write(RUN/'public/TRAINING_AUDIT.json',dict(status='PASS',trajectories=40,updates=7680,Base_gradient=False,fixed_cores_unchanged=True,replay=False,parameters=7168,trainable_parameters=2048))
    c.write(RUN/'public/RESOURCE_LEDGER.json',{k:result[k] for k in ('Judge_attempts','new_Judge_attempts','GPU_process_hours','new_GPU_process_hours')})
    c.write(RUN/'public/COMPLETION_AUDIT.json',dict(status=result['status'],scoring=result['scoring'],formal_updates=7680,final_weights=40,intermediate_weights_deleted=80,all_GPU_sessions_ended=True,review_required=True))
    lines=['# 无回放 ScopedWrite：8专家先导结果','', '完整146库仅干预前8专家，其他138保持原W0。区间为缺失赋值界，不是置信区间。','',
        '| 条件 | 受影响T0 | 受影响T1G | 受影响T2G | 强制CHECK已更新子集 | 自然CHECK全部 |','|---|---|---|---|---|---|']
    fmt=lambda b:'NA' if b is None or b[0] is None else f'{b[0]:.3f}' if b[0]==b[1] else f'[{b[0]:.3f}, {b[1]:.3f}]'
    for label in FINAL:
        vals=[fmt(next(x['macro_bounds'] for x in panels if x['support_scope']=='affected' and x['arm']==label and x['task']==task)) for task in ('T0','T1G','T2G')]
        for mode,scope in [('forced','updated'),('natural','all')]:
            row=next((x for x in checks if x['mode']==mode and x['scope']==scope and x['arm']==label),None)
            vals.append(fmt(row['accuracy']['macro_bounds']) if row and 'accuracy' in row else 'NA')
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines+=['', '主候选E的开发T2G筛选：'+status+'。GEOMETRY/GATING/SYSTEM仍需结合全部配对区间、开放题、Base正确子集保持和真实更新范数审阅；不自动宣布成功或晋级。', '',
        f"评分有效{result['scoring']['valid']}/{len(scores)}，缺失{result['scoring']['missing']}；累计GPU进程小时{result['GPU_process_hours']:.6f}，Judge{result['Judge_attempts']}。", '',
        '已保留40个最终小型TT及哈希。旧评分/原W0不变，无回放、无自动扩至24/146专家、无自动重试。数据均为开发暴露，患者独立性与医学作用域未确认。自然路由CE历史权重已删除，未重训补齐该参考。']
    (RUN/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n');p.done('REPORT_COMPLETE')


if __name__=='__main__':main()
