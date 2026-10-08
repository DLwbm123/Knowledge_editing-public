"""Report every frozen node; never select a checkpoint on exposed CHECK outcomes."""
from collections import Counter,defaultdict
import sqlite3
import trajectory as exp
import trajectory_queue as queue
import astra_report as ar
from replay_report import summary

c,RUN,r=exp.c,exp.RUN,ar.r


def contrast(a,b,scores,groups):
    assert set(a)==set(b)
    coef={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
    bounds=r.score_bounds(r.combine([{k:v/len(coef) for k,v in z.items()} for z in coef.values()]),scores) if coef else [None,None]
    return dict(delta_bounds_pp=[None if v is None else v*100 for v in bounds],**ar.bootstrap(coef,scores,groups))


def verdict(bounds,gain=False):
    lo,hi=bounds
    if lo is None:return 'UNRESOLVED'
    if (lo>0 if gain else lo>=0):return 'PASS'
    if (hi<=0 if gain else hi<0):return 'FAIL'
    return 'UNRESOLVED'


def main():
    assert verdict([0,0])=='PASS' and verdict([0,0],True)=='FAIL'
    assert verdict([-1,1])=='UNRESOLVED' and verdict([-2,-1])=='FAIL'
    root=queue.q.ROOT;assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    records=[];mapping={}
    for x in db.execute('SELECT * FROM consumer'):
        d=c.read(x['path']);assert c.digest(d)==x['output_binding']
        d=dict(d,binding=dict(d['binding'],arm=x['method'],mode=x['mode'],phase=dict(d['binding']['phase'],arm=x['method'],node=0,prefix=146)))
        records.append((d,x['payload_key']));mapping[x['method'],x['query_id']]=d,x['payload_key']
    assert len(records)==c.read(RUN/'public/ADMISSION.json')['consumers']
    tasks=exp.p.tasks();panels=[];coeff={};groups={};check_coeff={};check_groups={};checks=[]
    rows=[x['row'] for x in c.read(RUN/'private/TRAJECTORY_JOBS.json') if x['fresh']]
    assert len(rows)==48
    base={q['query_id']:scores[mapping['BASE',q['query_id']][1]] for q in rows}
    assert all(v in (0,1) for v in base.values())
    for arm in exp.ARMS:
        for node in (0,)+exp.NODES+(192,):
            label=exp.label(arm,node)
            for task in ('T0','T1G','T2G','T1L','T2L'):
                m,co,g=r.panel(records,scores,'ROUTED16',label,0,'bank_R0',task,tasks,146)
                m.update(ar.bootstrap(co,scores,g));m.update(training_arm=arm,training_step=node,scope='fixed selected-expert query subset')
                panels.append(m);coeff[arm,node,task]=co;groups.update(g)
            check={q['query_id']:scores[mapping[label,q['query_id']][1]] for q in rows}
            chosen=[q for q in rows if base[q['query_id']]==1]
            for name,qs in [('accuracy',rows),('retention',chosen)]:
                bygroup=defaultdict(list)
                for q in qs:bygroup[q['source_group']].append(mapping[label,q['query_id']][1])
                co={g:{k:n/len(keys) for k,n in Counter(keys).items()} for g,keys in bygroup.items()}
                check_coeff[arm,node,name]=co;check_groups.update({g:g for g in co})
            checks.append(dict(arm=arm,node=node,all_accuracy=summary([(q['source_group'],check[q['query_id']]) for q in rows]),
                Base_correct_retention=summary([(q['source_group'],check[q['query_id']]) for q in chosen]),
                retention_micro_bounds=[sum(check[q['query_id']] if check[q['query_id']] is not None else missing for q in chosen)/len(chosen)*100 for missing in (0,1)],
                Base_correct_count=len(chosen),by_answer_type={kind:summary([(q['source_group'],check[q['query_id']]) for q in rows if q['answer_kind']==kind]) for kind in ('yes','no','open')},
                exact_Base_tokens=sum(mapping[label,q['query_id']][0]['R0']['raw_token_ids']==mapping['BASE',q['query_id']][0]['R0']['raw_token_ids'] for q in rows)/len(rows),
                forced_expert=True,natural_bank_route=False))
    contrasts=[];lookup={}
    for node in exp.NODES+(192,):
        for a,an,b,bn in [('SOURCE_REPLAY192',node,'CE192',0),('SOURCE_REPLAY192',node,'CE192',node),('CE192',node,'CE192',0)]:
            for task in ('T0','T1G','T2G','T1L','T2L'):
                value=contrast(coeff[a,an,task],coeff[b,bn,task],scores,groups)
                contrasts.append(dict(a=a,a_node=an,b=b,b_node=bn,task=task,**value));lookup[a,an,b,bn,task]=value
            for name in ('accuracy','retention'):
                value=contrast(check_coeff[a,an,name],check_coeff[b,bn,name],scores,check_groups)
                contrasts.append(dict(a=a,a_node=an,b=b,b_node=bn,task='CHECK_'+name,**value));lookup[a,an,b,bn,'CHECK_'+name]=value
    windows=[]
    for node in exp.NODES+(192,):
        tests={task:verdict(lookup['SOURCE_REPLAY192',node,'CE192',0,task]['delta_bounds_pp']) for task in ('T0','T1G','T2G')}
        tests.update({name:verdict(lookup['SOURCE_REPLAY192',node,'CE192',0,name]['delta_bounds_pp'],True) for name in ('CHECK_accuracy','CHECK_retention')})
        state='NO_WINDOW_AT_NODE' if 'FAIL' in tests.values() else 'UNRESOLVED' if 'UNRESOLVED' in tests.values() else 'DESCRIPTIVE_CANDIDATE_ONLY'
        windows.append(dict(node=node,tests=tests,status=state,independent_confirmation=False))
    shapes=[]
    for arm in exp.ARMS:
        for node in (0,)+exp.NODES+(192,):
            label=exp.label(arm,node)
            for domain,qs in [('T2G',None),('source_CHECK_open',[q['query_id'] for q in rows if q['answer_kind']=='open'])]:
                if qs is None:
                    ledger=c.read(RUN/'private/EVAL_LEDGER.json')
                    qs=list(dict.fromkeys(q for t in tasks for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if ledger['Base_correctness'][q] is False and (label,q) in mapping))
                ds=[mapping[label,q][0] for q in qs];answers=[x['R0']['raw_answer'].strip().lower().strip('.,! ') for x in ds]
                lengths=sorted(len(x['R0']['raw_token_ids']) for x in ds)
                shapes.append(dict(arm=arm,node=node,domain=domain,n=len(ds),empty=sum(not a for a in answers),
                    binary_instead_of_open=sum(a in ('yes','no') for a in answers),mean_raw_tokens=sum(lengths)/len(lengths),
                    median_raw_tokens=lengths[len(lengths)//2],max_raw_tokens=max(lengths)))
    training=[c.read(x) for x in (RUN/'private/training').glob('*/TRAINING.json')]
    assert len(training)==16 and sum(z['updates'] for x in training for z in x['arms'])==3264
    nodes=[c.read(x) for x in (RUN/'private/weights').glob('*/*/step[0-9][0-9][0-9].json')];assert len(nodes)==96
    resource=c.read(RUN/'RESOURCE_LEDGER.json');old=c.read(RUN/'private/INHERITED_COST.json')
    result=dict(status='TERMINAL_WITH_MISSING' if any(v is None for v in scores.values()) else 'COMPLETE',
        panels=panels,source_CHECK=checks,contrasts=contrasts,descriptive_windows=windows,output_shape=shapes,
        scoring=dict(payloads=len(scores),valid=sum(v is not None for v in scores.values()),missing=sum(v is None for v in scores.values()),consumers=len(records)),
        training_experts=16,training_trajectories=32,training_updates=3264,new_outputs=exp.lock()['new_outputs'],
        Judge_attempts=resource['Judge_attempts'],new_Judge_attempts=resource['Judge_attempts']-old['Judge_attempts'],
        GPU_process_hours=c.used()/3600,new_GPU_process_hours=(c.used()-old['gpu_seconds_used'])/3600,
        coverage=[dict(node=n,replay_samples=n,replay_source_groups=n//3,each_answer_type=n//3) for n in (0,)+exp.NODES+(192,)],
        deletion=c.read(RUN/'private/DELETION.json'),endpoint_reconstruction=c.read(RUN/'public/GPU_MECHANICAL.json'),
        independent_confirmation=False,medical_scope_qualification=False,automatic_checkpoint_selection=False,
        inference_limit='Fixed16 routed subset and forced16-source stress test; step/coverage confounded; all nodes exposed development')
    c.write(RUN/'public/TRAJECTORY_RESULTS.json',result)
    lines=['# 固定回放轨迹：全部节点读出','', '以下为固定16专家调用子集；来源CHECK为48题/16组强制激活，不能当作全146库或独立确认。区间为缺失界。','',
        '| 步数 | CE T2G | 回放T2G | CE来源CHECK | 回放来源CHECK | 回放空答/二元替代（T2G） |','|---|---|---|---|---|---|']
    for node in (0,)+exp.NODES+(192,):
        vals=[]
        for arm in exp.ARMS:vals.append(next(x['macro_bounds'] for x in panels if x['training_arm']==arm and x['training_step']==node and x['task']=='T2G'))
        for arm in exp.ARMS:vals.append(next(x['all_accuracy']['macro_bounds'] for x in checks if x['arm']==arm and x['node']==node))
        shape=next(x for x in shapes if x['arm']==exp.ARMS[1] and x['node']==node and x['domain']=='T2G')
        fmt=lambda x:f'{x[0]:.3f}' if x[0]==x[1] else f'[{x[0]:.3f},{x[1]:.3f}]'
        lines.append('| '+str(node)+' | '+' | '.join(map(fmt,vals))+f" | {shape['empty']}/{shape['binary_instead_of_open']}，分母{shape['n']} |")
    lines+=['','描述性窗口：'+str(windows),'',f"评分有效{result['scoring']['valid']}/{len(scores)}，缺失{result['scoring']['missing']}；累计Judge{resource['Judge_attempts']}，累计GPU进程小时{result['GPU_process_hours']:.6f}。",'',
        '所有节点、配对区间、T0/T1G/T2G/T1L/T2L、来源保持率和答案类型见TRAJECTORY_RESULTS.json。没有选择或晋级checkpoint；没有追加参数/种子/节点。新增权重在冻结消费者完成且绑定落盘后删除，历史资产保持只读。']
    (RUN/'public/TRAJECTORY_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    exp.p.done('TRAJECTORY_REPORT_COMPLETE');exp.p.progress('TRAJECTORY_COMPLETE_REVIEW_PUBLICATION_PENDING')


if __name__=='__main__':main()
