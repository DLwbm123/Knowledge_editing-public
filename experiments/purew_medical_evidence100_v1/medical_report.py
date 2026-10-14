"""Terminal anonymous comparison, exact shared missing bounds, no clinical claims."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
import time
import astra_report as ar
from logic import transitions
R,c,r=ar.RUN,ar.common,ar.r
ARMS=('M','M-S','M-E')


def main():
    cost=c.read(R/'RESOURCE_LEDGER.json');assert not any(not x.get('ended_epoch') for x in cost['gpu_sessions'])
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=list(db.execute('SELECT * FROM consumer'));payloads={x['key']:x for x in db.execute('SELECT * FROM payload')}
    scores={k:x['correct'] for k,x in payloads.items()};coverage=dict(db.execute('SELECT method,count(*) FROM consumer GROUP BY method'))
    unresolved=db.execute("SELECT count(*) FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()[0]
    complete=(R/'private/GENERATION_COMPLETE.json').exists() and (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists() and not unresolved
    attempted=[]
    from scripts.medtrace.astra_judge_bundle import validate
    for batch in cost['Astra_purew146_batches']:
        attempted+=batch['keys']
        if batch['status']=='RESERVED':assert not complete;continue
        ev=c.read(R/'private/judge_purew146/evidence'/(batch['id']+'.json'))
        assert ev['batch']['records']==[json.loads(payloads[k]['record']) for k in batch['keys']]
        if batch['status']=='FORMAT_VALID':
            e=ev['evidence'];assert e['actual_model']=='gpt-6-astra' and e['reasoning_effort']=='medium'
            assert e['input_binding']==c.digest(ev['batch']) and not e['tool_event_types'] and all(e['isolation_checks'].values())
            assert [int(x['is_correct']) for x in validate(ev['batch'],ev['response'])]==[scores[k] for k in batch['keys']]
        else:assert batch['status']=='FAILED_NO_RETRY' and all(scores[k] is None for k in batch['keys'])
    assert len(attempted)==len(set(attempted))==cost['Judge_attempts']
    attempts=[c.read(p) for p in (R/'private/attempts').glob('*.json')]
    counts={k:sum(x.get(k,0) for x in attempts) for k in ('forwards','backwards','committed_updates','generation_attempts','attempt_slots','vision_forwards')}
    generated={arm:sum(c.read(p)['outputs'] for p in (R/'private/outputs'/arm).glob('*/*/*/COMPLETE.json')) for arm in ARMS}
    result=dict(status='COMPLETE_WITH_MISSING' if complete else 'PARTIAL_NOT_ALL_CONSUMERS_COMPLETE',scientific_success=False,
        coverage=coverage,expected_consumers_per_arm=2181,generated=generated,not_generated={a:2181-generated[a] for a in ARMS},
        unattempted_or_unscored_generated={a:max(0,generated[a]-coverage.get(a,0)) for a in ARMS},
        scoring_statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),new_Judge_attempts=cost['Judge_attempts'],
        failed_new_batches=sum(x['status']=='FAILED_NO_RETRY' for x in cost['Astra_purew146_batches']),actual_compute_counts=counts,
        GPU_process_hours=cost['gpu_seconds_used']/3600,owned_weight_peak_bytes=cost['weights_observed_peak_bytes'],
        GPU_memory_observed_peak_MiB=cost.get('GPU_memory_observed_peak_MiB'),failures=len(list((R/'private/failures').glob('*.json'))),
        panels=[],contrasts=[],endpoint_transitions=[],gates={},diagnostics={},new_inference_parameters=0,
        original146='PAUSED_RETAINED',OL='WAIT_ORIGINAL_FULL_PREDECESSOR',
        limitations=['Current-case-only source objective; no historical sample replay or BASIS KL',
            'Sampled GGN + FD energy is an approximate functional proxy, not correctness or clinical-risk guarantee',
            'FP16 Jacobian uses real-valued shadow derivative; Taylor and sampling errors unknown',
            'Parameter statistics are not a privacy guarantee; Python I/O boundary is not an adversarial OS sandbox',
            'Only12 source-defined brainMRI coarse-box sources; no masks or neutral/background certificate',
            'K and new arms have different replay/data conditions and compute; no isolated causal KL comparison',
            'Developmental observed sequence; patient independence unknown; no clinical/SOTA validation'])
    if complete:
        assert all(coverage.get(a)==2181 for a in ARMS) and counts['attempt_slots']==48000
        assert all((R/'private'/('FINAL_CONSUMED_'+a+'.json')).exists() for a in ARMS)
        tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][:100];ledger=c.read(R/'private/EVAL_LEDGER.json')
        records=[];lookup={}
        for x in rows:
            query=ledger['queries'][x['query_id']];full=json.loads(payloads[x['payload_key']]['binding'])
            assert full['query_id']==query['query_id'] and all(full[k]==query[k] for k in ('question','reference','image_sha256'))
            if x['method'] in ARMS:
                d=c.read(x['path']);assert c.digest(d)==x['output_binding'] and full['output']==d['R0']
                assert d['binding']['phase']['protocol']==c.digest(c.read(R/'private/FROZEN_PROTOCOL.json'))
            active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:x['prefix']]} if x['mode']=='bank_R0' else set()
            records.append((dict(binding=dict(phase=dict(arm=x['method'],node=0,prefix=x['prefix']),input=query,mode=x['mode'],owner_order=x['edit_order']),
                active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),x['payload_key']))
            key=(x['method'],x['mode'],x['prefix'],x['edit_order'],x['query_id']);assert key not in lookup;lookup[key]=x['payload_key']
        coeff={};groups={}
        for arm in ('C','K',*ARMS,'TT88_W0_RETRO146','balancedit'):
            for prefix in ((1,50,100) if arm in ('C','K',*ARMS) else (100,)):
                for task in ('T0','T1G','T2G','T1L','T2L'):
                    metric,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,tasks[:prefix],prefix)
                    metric.pop('route_ON');n=metric['observations'];metric.update(mode='bank_R0',prefix=prefix,native_edit_count=prefix)
                    metric['micro_bounds']=[100*metric['known_correct']/n,100*(metric['known_correct']+metric['missing'])/n] if n else None
                    if task.endswith('L'):metric['Base_correct_to_wrong_count_bounds']=[n-metric['known_correct']-metric['missing'],n-metric['known_correct']]
                    result['panels'].append(metric);coeff[arm,prefix,task]=co;groups.update(g)
        eligible=[t for t,review in zip(tasks,c.read(R/'private/operator/review.json')['rows']) if review['region_status']=='ELIGIBLE_COARSE_BOX'] if (R/'private/operator/review.json').exists() else []
        if eligible:
            assert len(eligible)==12
            for arm in ARMS:
                for task in ('T0','T1G','T2G','T1L','T2L'):
                    metric,_,_=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,eligible,100)
                    metric.pop('route_ON');metric.update(mode='bank_R0',prefix=100,native_edit_count=12,support='SOURCE_DEFINED_BRAINMRI_COARSE_BOX12')
                    result['panels'].append(metric)
        for aa,bb in (('M-S','M'),('M-E','M-S'),('M-E','M'),('M','K'),('M-S','K'),('M-E','K'),('M-E','C'),('M-E','TT88_W0_RETRO146'),('M-E','balancedit')):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                a,b=coeff[aa,100,task],coeff[bb,100,task];assert set(a)==set(b)
                delta={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
                bound=r.score_bounds(r.combine([{k:v/len(delta) for k,v in row.items()} for row in delta.values()]),scores)
                entry=dict(comparison=aa+'-minus-'+bb+'-prefix100',task=task,delta_macro_bounds_pp=[100*x for x in bound],
                    edits_negative_lower_bound=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),
                    edits_definitely_worse=sum(r.score_bounds(x,scores)[1]<-1e-12 for x in delta.values()))
                if task=='T2G':entry.update(ar.bootstrap(delta,scores,groups))
                result['contrasts'].append(entry)
        for arm in ARMS:
            for task in ('T0','T2G'):
                pairs=[]
                for t in tasks:
                    ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]
                    pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',100,0,q]) for q in dict.fromkeys(ids))
                result['endpoint_transitions'].append(dict(method=arm,task=task,comparison='actual-insertion-to-prefix100',**transitions(pairs,scores)))
            for baseline in ('C','K'):
                for task in ('T1L','T2L'):
                    ids=[q for t in tasks for e in t['events'] if e['task']==task for q in e['all_probe_query_ids'] if ledger['Base_correctness'][q]]
                    active={(t['native']['image_sha256'],t['native']['question']) for t in tasks}
                    ids=[q for q in dict.fromkeys(ids) if (ledger['queries'][q]['image_sha256'],ledger['queries'][q]['question']) not in active]
                    pairs=[(lookup[baseline,'bank_R0',100,0,q],lookup[arm,'bank_R0',100,0,q]) for q in ids]
                    result['endpoint_transitions'].append(dict(method=arm,task=task,comparison=baseline+'-to-'+arm,**transitions(pairs,scores)))
            for task in ('T0','T1G','T2G','T1L','T2L'):
                ids=[q for t in tasks[:50] for e in t['events'] if e['task']==task for q in e['all_probe_query_ids'] if task=='T0' or ledger['Base_correctness'][q]==task.endswith('L')]
                pairs=[(lookup[arm,'bank_R0',50,0,q],lookup[arm,'bank_R0',100,0,q]) for q in dict.fromkeys(ids)]
                result['endpoint_transitions'].append(dict(method=arm,task=task,comparison='same-first50-prefix50-to100',**transitions(pairs,scores)))
            final=[x for x in result['panels'] if x['arm']==arm and x['prefix']==100 and x['native_edit_count']==100]
            native=[scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]] for t in tasks]
            result['gates'][arm]=dict(zero_Base_damage=all(x['known_correct']==x['observations'] for x in final if x['task'] in ('T1L','T2L')),
                T0_insertion_and_final_all_correct=all(x==1 for x in native) and next(x for x in final if x['task']=='T0')['known_correct']==100,
                T1G_full_retention=next(x for x in final if x['task']=='T1G')['known_correct']==394,
                full_required_scoring=all(scores[x['payload_key']] is not None for x in rows if x['method']==arm))
            events=[c.read(p) for p in (R/'private/steps'/arm).glob('*/*.json')];assert len(events)==16000
            assert {(x['owner'],x['slot']) for x in events}=={(i,s) for i in range(1,101) for s in range(1,161)}
            result['diagnostics'][arm]=dict(slots=len(events),accepted_native_writes=sum(x['accepted'] for x in events),
                rejected_or_zero_slots=sum(not x['accepted'] for x in events),proposal_path_length=sum(x['proposal_norm'] for x in events),
                native_path_length=sum(x['actual_native_norm'] for x in events),max_proxy_risk=max(x['risk_after'] for x in events),
                mean_source_CE=sum(x['source_CE'] for x in events)/16000,mean_medical_scope_loss=sum(x['medical_scope_loss'] for x in events)/16000,
                consolidation_nodes=100,no_old_training_samples=True,
                source_evidence_diagnostics=[{k:v for k,v in c.read(p).items() if k!='order'} for p in (R/'private/source_diagnostics'/arm/'final').glob('*.json')])
        result['gates']['M-E']['T2G_per_edit_nonregression']=all(x['edits_negative_lower_bound']==0 for x in result['contrasts'] if x['comparison'] in ('M-E-minus-M-S-prefix100','M-E-minus-K-prefix100') and x['task']=='T2G')
        result['scientific_success']=all(result['gates']['M-E'].values())
    c.write(R/'public/RESULTS.json',result)
    lines=['# 无历史病例重放的连续纯W M/M-S/M-E有限对照','',f'状态：{result["status"]}；完整科学门槛：{result["scientific_success"]}。','',
        '|方法|T0|T1G宏界%|T2G宏界%|T1L Base损伤|T2L Base损伤|','|---|---:|---:|---:|---:|---:|']
    for arm in ('C','K',*ARMS,'TT88_W0_RETRO146','balancedit'):
        ps={x['task']:x for x in result['panels'] if x['arm']==arm and x['prefix']==100 and x['native_edit_count']==100}
        if ps:lines.append(f'|{arm}|{ps["T0"]["known_correct"]}/100|{ps["T1G"]["macro_bounds"]}|{ps["T2G"]["macro_bounds"]}|{ps["T1L"]["Base_correct_to_wrong_count_bounds"]}/2|{ps["T2L"]["Base_correct_to_wrong_count_bounds"]}/34|')
    lines+=['','各原分母、微平均、共享missing精确界、10000次原seed编辑/来源区间及逐题新增损伤/修复见RESULTS.json。',
        '新轨迹记录真实T0/T2G插入到最终；C的T2G插入未测，单例不能替代。完整终态仍不等于科学成功。',
        'M-E对M-S只改变证据函数累计保护；M-S对M加入当前区域范围目标。K含历史重放和61保护源，数据条件与计算量不同。',
        '累计参数统计不保存旧病例或特征，但不自动提供隐私保证。Python读访问守护不是对抗性操作系统沙箱。',
        '粗框只支持12项脑MRI；删除区域不是健康图像，证据指标来自训练源，不是独立临床证实。',
        '曲率采样/FD/局部Taylor/量化都有边界；代理风险不保证真实旧题正确。无临床/SOTA确认。',
        f'新增GPU进程小时：{result["GPU_process_hours"]:.4f}；新增Judge：{cost["Judge_attempts"]}；全部失败成本保留。']
    (R/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n');c.write(R/'private/REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time()))
    print(json.dumps(dict(status=result['status'],scientific_success=result['scientific_success'])))


if __name__=='__main__':main()
