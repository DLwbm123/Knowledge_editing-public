"""Anonymous finite diagnostic: exact shared missing bounds and unchanged protection gate."""
from collections import Counter
import json
import os
import sys
import sqlite3
import time
sys.path.insert(0, os.environ["RUN_ROOT"] + "/private/tools")
import astra_report as ar
from panels import protection_ids
from logic import transitions
R,c,r=ar.RUN,ar.common,ar.r


def audit_requests(payloads,cost):
    from scripts.medtrace.astra_judge_bundle import validate
    attempted=[]
    for batch in cost['Astra_purew146_batches']:
        attempted+=batch['keys'];ev=c.read(R/'private/judge_purew146/evidence'/(batch['id']+'.json'))
        assert ev['batch']['records']==[json.loads(payloads[k]['record']) for k in batch['keys']]
        if batch['status']=='FORMAT_VALID':
            e=ev['evidence'];assert e['actual_model']=='gpt-6-astra' and e['reasoning_effort']=='medium'
            assert e['input_binding']==c.digest(ev['batch']) and not e['tool_event_types'] and all(e['isolation_checks'].values())
            assert [int(x['is_correct']) for x in validate(ev['batch'],ev['response'])]==[payloads[k]['correct'] for k in batch['keys']]
        else:assert batch['status']=='FAILED_NO_RETRY' and all(payloads[k]['correct'] is None for k in batch['keys'])
    assert len(attempted)==len(set(attempted))==cost['Judge_attempts']


def partial_coverage(generated,db):
    statuses={a:Counter() for a in generated}
    if db is not None:
        for row in db.execute("SELECT consumer.method,payload.status FROM consumer JOIN payload ON consumer.payload_key=payload.key"):
            statuses[row[0]][row[1]]+=1
    terminal={a:sum(n for state,n in states.items() if state in ('FORMAT_VALID','MISSING')) for a,states in statuses.items()}
    assert all(0<=terminal[a]<=sum(statuses[a].values())<=generated[a]<=161 for a in generated)
    return dict(consumer_status_counts={a:dict(states) for a,states in statuses.items()},
        terminal_consumers=terminal,generated_not_terminal={a:generated[a]-terminal[a] for a in generated},
        generated_not_ingested={a:generated[a]-sum(statuses[a].values()) for a in generated})


def main():
    cost=c.read(R/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in cost['gpu_sessions'])
    if not (R/'private/GENERATION_COMPLETE.json').exists() or not (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists():
        assert list((R/'private/failures').glob('*.json')),'Healthy incomplete run cannot be declared terminal'
        generated={a:sum(c.read(p)['outputs'] for p in (R/'private/outputs'/a).glob('*/*/*/COMPLETE.json')) for a in ('M','F','S')}
        queue=R/'private/judge_purew146/queue.sqlite'
        db=sqlite3.connect('file:'+str(queue)+'?mode=ro',uri=True) if queue.exists() else None
        if db is not None:
            db.row_factory=sqlite3.Row
            audit_requests({x['key']:x for x in db.execute('SELECT * FROM payload')},cost)
        else:assert cost['Judge_attempts']==0
        coverage=partial_coverage(generated,db)
        if db is not None:db.close()
        counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ('forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts')}
        result=dict(status='PARTIAL_DIAGNOSTIC_NOT_COMPLETE',scientific_success=False,diagnostic_only=True,generated=generated,
            not_generated={a:161-generated[a] for a in generated},**coverage,actual_compute_counts=counts,
            new_Judge=cost['Judge_attempts'],GPU_process_hours=cost['gpu_seconds_used']/3600,
            failures=len(list((R/'private/failures').glob('*.json'))),metrics='NA_UNFINISHED',permanent_missing_not_used_for_unattempted=True,
            original146='PAUSED',original_M='PAUSED',OL='WAITING',automatic_replay=False)
        c.write(R/'public/RESULTS.json',result)
        (R/'public/REPORT_ZH.md').write_text('# F/S有限诊断partial\n\n执行故障或预算止步，所需轨迹/重载/评分屏障未完成。指标NA，未生成/未评分不伪作永久missing；失败、实际成本和恢复点全部保留，不能声称科学成功。详见RESULTS.json及私有故障回执。原146/M暂停，O/L等待，无盲重放。\n')
        c.write(R/'private/REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time()))
        print(json.dumps(dict(status=result['status'])));return
    assert (R/'private/GENERATION_COMPLETE.json').exists() and (R/'private/FINAL_CONSUMED.json').exists()
    assert (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    rows=list(db.execute('SELECT * FROM consumer'));assert len(rows)==483
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')};scores={k:x['correct'] for k,x in payloads.items()}
    audit_requests(payloads,cost)
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json')
    records=[];lookup={}
    for x in rows:
        d=c.read(x['path']);assert c.digest(d)==x['output_binding']
        full=json.loads(payloads[x['payload_key']]['binding']);assert full['output']==d['R0']
        query=ledger['queries'][x['query_id']];assert full['query_id']==query['query_id']
        assert all(full[k]==query[k] for k in ('question','reference','image_sha256'))
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:x['prefix']]} if x['mode']=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=x['method'],node=0,prefix=x['prefix']),input=query,mode=x['mode'],owner_order=x['edit_order']),
            active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),x['payload_key']))
        lookup[x['method'],x['mode'],x['prefix'],x['edit_order'],x['query_id']]=x['payload_key']
    panels=[];coef={};groups={}
    for arm in ('M','F','S'):
        for prefix in (1,3):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                metric,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,tasks[:prefix],prefix)
                metric.pop('route_ON');n=metric['observations'];metric['micro_bounds']=[100*metric['known_correct']/n,100*(metric['known_correct']+metric['missing'])/n] if n else None
                panels.append(metric);coef[arm,prefix,task]=co;groups.update(g)
    contrasts=[]
    for aa,bb in (('F','M'),('S','F'),('S','M')):
        a,b=coef[aa,3,'T2G'],coef[bb,3,'T2G'];assert set(a)==set(b)
        delta={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
        bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in x.items()} for x in delta.values()]),scores)
        contrasts.append(dict(comparison=aa+'-minus-'+bb,task='T2G',delta_macro_bounds_pp=[100*x for x in bounds],
            negative_edit_bounds=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),**ar.bootstrap(delta,scores,groups)))
    protection=[];endpoints=[];gates={}
    for arm in ('M','F','S'):
        gate=True
        for task in ('T1L','T2L'):
            ids=protection_ids(tasks,ledger,task);assert len(ids)==(2 if task=='T1L' else 34)
            keys=[lookup[arm,'protect100_R0',100,0,q] for q in ids]
            correct=sum(scores[k]==1 for k in keys);unknown=sum(scores[k] is None for k in keys)
            protection.append(dict(arm=arm,task=task,observations=len(ids),known_correct=correct,missing=unknown,
                Base_correct_to_wrong_bounds=[len(ids)-correct-unknown,len(ids)-correct],known_correct_bounds=[correct,correct+unknown]))
            gate &= correct==len(ids)
            if arm!='M':
                pairs=[(lookup['M','protect100_R0',100,0,q],lookup[arm,'protect100_R0',100,0,q]) for q in ids]
                endpoints.append(dict(arm=arm,task=task,comparison='M3-reference-to-'+arm,**transitions(pairs,scores)))
        gates[arm]=dict(zero_individual_Base_damage=gate,
            all_T0_insertion_and_final_correct=all(scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]]==scores[lookup[arm,'bank_R0',3,0,t['edit_id']]]==1 for t in tasks[:3]),
            T1G_full_retention=next(x for x in panels if x['arm']==arm and x['prefix']==3 and x['task']=='T1G')['known_correct']==next(x for x in panels if x['arm']==arm and x['prefix']==3 and x['task']=='T1G')['observations'],
            full_required_scoring=all(scores[x['payload_key']] is not None for x in rows if x['method']==arm))
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks[:3]:
                ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]
                pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',3,0,q]) for q in dict.fromkeys(ids))
            endpoints.append(dict(arm=arm,task=task,comparison='actual-insertion-to-prefix3',**transitions(pairs,scores)))
    diagnostics={}
    for arm in ('F','S'):
        events=[c.read(p) for p in (R/'private/steps'/arm).glob('*/*.json')];assert len(events)==480
        reasons=Counter(reason for e in events for trial in e['candidate_diagnostics'] for reason in trial['rejection_reasons'])
        diagnostics[arm]=dict(slots=480,accepted=sum(e['accepted'] for e in events),actual_zero=sum(e['actual_native_norm']==0 for e in events),
            candidate_reasons=dict(reasons),mean_CE=sum(e['source_CE'] for e in events)/480,max_risk=max(e['risk_after'] for e in events),
            per_source=[dict(accepted=sum(e['accepted'] for e in events if e['owner']==i),
                first20_CE=sum(e['source_CE'] for e in events if e['owner']==i and e['slot']<=20)/20,
                last20_CE=sum(e['source_CE'] for e in events if e['owner']==i and e['slot']>140)/20) for i in (1,2,3)])
    counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ('forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts')}
    assert counts['attempt_slots']==960
    full=all(scores[x['payload_key']] is not None for x in rows)
    result=dict(status='COMPLETE' if full else 'COMPLETE_WITH_MISSING',scientific_success=False,diagnostic_only=True,
        panels=panels,contrasts=contrasts,protection=protection,endpoints=endpoints,gates=gates,diagnostics=diagnostics,
        actual_compute_counts=counts,GPU_process_hours=cost['gpu_seconds_used']/3600,new_Judge=cost['Judge_attempts'],
        owned_weight_peak_bytes=cost['weights_observed_peak_bytes'],failures=len(list((R/'private/failures').glob('*.json'))),
        K_matched_prefix3='NA_NOT_MEASURED_NO_OLD_WEIGHT_RECONSTRUCTION',M_reference_retrained=False,
        new_inference_parameters=0,original146='PAUSED',OL='WAITING',
        limitations=['Only first3 observed development sources; no full100 scalability/medical claim',
            'Fisher units differ from original raw-logit M epsilon: not equal-risk or equal-compute',
            'S retains complete finite sampled subspace; not a scalable100 K-FAC implementation',
            'Four stochastic Fisher probes and local real-valued Taylor proxy do not certify old correctness',
            'Fixed-size parameter statistics are not a privacy guarantee',
            'Full36 protection mask retained; unknown does not pass individual zero-damage gate'])
    c.write(R/'public/RESULTS.json',result)
    lines=['# F/S前三项有限机制诊断','',f'状态：{result["status"]}。诊断试验，不宣称完整科学/医学成功。','',
        '|方法|T0|T1G宏界%|T2G宏界%|完整T1L损伤界/2|完整T2L损伤界/34|','|---|---:|---:|---:|---:|---:|']
    for arm in ('M','F','S'):
        ps={x['task']:x for x in panels if x['arm']==arm and x['prefix']==3};pp={x['task']:x for x in protection if x['arm']==arm}
        lines.append(f'|{arm}|{ps["T0"]["known_correct"]}/3|{ps["T1G"]["macro_bounds"]}|{ps["T2G"]["macro_bounds"]}|{pp["T1L"]["Base_correct_to_wrong_bounds"]}|{pp["T2L"]["Base_correct_to_wrong_bounds"]}|')
    lines+=['','M复用前三项真实训练及插入输出；新读取验证后的前三项末端W生成登记面板，不重训。K无同轨迹前三项测量，记NA，不把前100当作匹配对照。',
        'F与S使用相同token平均Fisher采样和0.01累计代理能量；S前三项保留最多12个旋转参数统计方向，不做FD各向同性补全，不保存旧病例/问答/特征。',
        '逐候选ZERO_NATIVE/RISK/PROXIMAL原因、接受/零更新、共享缺分差界、10000编辑/来源区间、真实插入端点与成本见RESULTS.json。',
        '完整评分终态含永久missing也不能认定全部门槛通过；未知与损伤不以修复或平均补偿。原146和原M保持暂停，O/L继续等待。']
    (R/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n');c.write(R/'private/REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time()))
    print(json.dumps(dict(status=result['status'],diagnostic_only=True)))


if __name__=='__main__':main()
