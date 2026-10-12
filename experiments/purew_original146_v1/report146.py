"""Use the original146 aggregation and shared-missing pairing; publish anonymous counts only."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_report as ar
from contract import ARM, METHODS, PREFIXES

R,c,r=ar.RUN,ar.common,ar.r


def main():
    root=R/'private/judge_purew146'
    assert (root/'ALL_WORKERS_COMPLETE.json').exists() and (R/'private/FINAL_CONSUMED.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=list(db.execute('SELECT * FROM consumer'));assert len(rows)==17166
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')}
    assert all(x['status'] in ('FORMAT_VALID','MISSING') for x in payloads.values())
    scores={k:x['correct'] for k,x in payloads.items()}
    from scripts.medtrace.astra_judge_bundle import validate
    cost=c.read(R/'RESOURCE_LEDGER.json')
    attempted=[]
    for batch in cost['Astra_purew146_batches']:
        attempted.extend(batch['keys'])
        evidence=c.read(root/'evidence'/(batch['id']+'.json'))
        assert evidence['batch']['records']==[json.loads(payloads[k]['record']) for k in batch['keys']]
        if batch['status']=='FORMAT_VALID':
            ev=evidence['evidence']
            assert ev['actual_model']=='gpt-6-astra' and ev['reasoning_effort']=='medium'
            assert ev['input_binding']==c.digest(evidence['batch']) and not ev['tool_event_types'] and all(ev['isolation_checks'].values())
            decisions=validate(evidence['batch'],evidence['response'])
            assert [int(x['is_correct']) for x in decisions]==[scores[k] for k in batch['keys']]
        else:
            assert batch['status']=='FAILED_NO_RETRY' and all(scores[k] is None for k in batch['keys'])
    assert len(attempted)==len(set(attempted))==cost['Judge_attempts']
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json')
    records=[]
    for row in rows:
        if row['method']==ARM:
            d=c.read(row['path'])
            assert c.digest(d)==row['output_binding']
            assert d['binding']['phase']['protocol']==c.digest(c.read(R/'private/FROZEN_PROTOCOL.json'))
            assert json.loads(payloads[row['payload_key']]['binding'])['output']==d['R0']
        query=ledger['queries'][row['query_id']]
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:row['prefix']]} if row['mode']=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=row['method'],node=0,prefix=row['prefix']),
            input=query,mode=row['mode'],owner_order=row['edit_order']),
            active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),row['payload_key']))
    panels=[];coeffs={};groups={}
    modes=[('single_R0',1)]+[('bank_R0',n) for n in PREFIXES]
    for mode,prefix in modes:
        for method in METHODS:
            for task in ('T0','T1G','T2G','T1L','T2L'):
                m,coef,g=r.panel(records,scores,'RETRO146',method,0,mode,task,tasks[:prefix] if mode=='bank_R0' else tasks,prefix)
                m.pop('route_ON');m.update(ar.bootstrap(coef,scores,g))
                panels.append(m);coeffs[method,mode,prefix,task]=coef;groups.update(g)
    prior=c.read(Path(c.read(R/'private/FROZEN_PROTOCOL.json')['common_parent'])/'public/RECOVERED_RESULTS.json')['panels']
    for m in panels:
        if m['arm']==ARM:continue
        expected=next(x for x in prior if all(x[k]==m[k] for k in ('arm','mode','prefix','task')))
        for k in ('known_correct','observations','missing','edit_units','macro'):
            assert m[k]==expected[k] or isinstance(m[k],float) and abs(m[k]-expected[k])<1e-9
    contrasts=[]
    for mode,prefix in modes:
        for baseline in METHODS[1:]:
            for task in ('T0','T1G','T2G','T1L','T2L'):
                a,b=coeffs[ARM,mode,prefix,task],coeffs[baseline,mode,prefix,task]
                assert set(a)==set(b)
                delta={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
                bound=r.score_bounds(r.combine([{k:v/len(delta) for k,v in unit.items()} for unit in delta.values()]),scores) if delta else [None,None]
                contrasts.append(dict(baseline=baseline,mode=mode,prefix=prefix,task=task,
                    delta_bounds_pp=[v*100 if v is not None else None for v in bound],**ar.bootstrap(delta,scores,groups)))
    lookup={(x['method'],x['mode'],x['prefix'],x['edit_order'],x['query_id']):x['payload_key'] for x in rows}
    insertion=[]
    for method in METHODS:
        pairs=[]
        for t in tasks:
            i=t['order'];qid=t['edit_id']
            pairs.append((lookup[method,'insertion_R0',i,i,qid],lookup[method,'bank_R0',146,0,qid]))
        transitions=defaultdict(int);unknown=0
        for a,b in pairs:
            if scores[a] is None or scores[b] is None:unknown+=1
            else:transitions[str(scores[a])+'_'+str(scores[b])]+=1
        insertion.append(dict(method=method,native_denominator=146,known_transitions=dict(transitions),missing_pairs=unknown,
            both_correct_bounds=[sum(scores[a]==scores[b]==1 for a,b in pairs),sum(scores[a]!=0 and scores[b]!=0 for a,b in pairs)]))
    diagnostics=[]
    for mode in ('single_R0','bank_R0'):
        events=[c.read(p) for p in (R/'private/steps'/mode).glob('*/*.json')];assert len(events)==23360
        diagnostics.append(dict(mode=mode,steps=len(events),mean_proposal_norm=sum(x['proposal_norm'] for x in events)/len(events),
            mean_actual_native_norm=sum(x['actual_native_norm'] for x in events)/len(events),zero_native_writes=sum(x['changed_elements']==0 for x in events)))
    assert all(x.get('ended_epoch') for x in cost['gpu_sessions'])
    timing=c.read(R/'public/NATIVE_RELOAD_PERFORMANCE.json')
    primary=[x for x in contrasts if x['mode']=='bank_R0' and x['prefix']==146]
    source=next(x for x in panels if x['arm']==ARM and x['mode']=='bank_R0' and x['prefix']==146 and x['task']=='T0')
    full=all(scores[x['payload_key']] is not None for x in rows if x['method']==ARM)
    no_forgetting=next(x for x in insertion if x['method']==ARM)['both_correct_bounds'][0]==146
    per_edit_failures={}
    for baseline in METHODS[1:]:
        per_edit_failures[baseline]={}
        for task in ('T0','T1G','T2G','T1L','T2L'):
            a,b=coeffs[ARM,'bank_R0',146,task],coeffs[baseline,'bank_R0',146,task]
            per_edit_failures[baseline][task]=sum(r.score_bounds(r.combine([a[e],{k:-v for k,v in b[e].items()}]),scores)[0]<0 for e in a)
    gate=full and no_forgetting and source['known_correct']==146 and all(x['delta_bounds_pp'][0]>=0 for x in primary) and not any(n for counts in per_edit_failures.values() for n in counts.values())
    result=dict(status='COMPLETE' if full else 'COMPLETE_WITH_MISSING',panels=panels,contrasts=contrasts,insertion=insertion,
        primary='bank146 edit-macro T2G',DEV_gate=gate,per_edit_negative_bound_counts=per_edit_failures,diagnostics=diagnostics,scoring_statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        added_parameters=0,inference=timing,new_GPU_process_hours=cost['gpu_seconds_used']/3600,new_Judge=cost['Judge_attempts'],
        prior_common_Judge_attempts=3346,old_results_unchanged=True,baseline_scores_reproduced=True,
        failures=len(list((R/'private/failures').glob('*.json')))+len(list((R/'private').glob('engineering_*/failures/*.json'))),owned_weight_peak_bytes=cost['weights_observed_peak_bytes'],
        owned_pt_remaining=len(list((R/'private').rglob('*.pt'))),data_admission=c.read(R/'public/BASELINE_REUSE_ADMISSION.json'),
        limitations=['NativeFP16 and quantization disclosed','Original146 development single-order','T1L/T2L historical diagnostic only','Patient independence UNKNOWN',
                     'Source-only supervision adaptation removes external61 BASIS KL','Method steps/parameters/optimizer differ','Historical baseline matched latency NA','No clinical/SOTA/independent confirmation'])
    c.write(R/'public/RESULTS.json',result)
    lines=['# 原146纯W与TT/BalancEdit完整对比','',f'执行终态：{result["status"]}；冻结DEV门槛：{gate}。','',
           '| 方法 | T0 | T1G Fix | T2G Fix | T1L Retention | T2L Retention |','|---|---:|---:|---:|---:|---:|']
    def cell(x):
        return f'{x["macro"]:.3f}%' if x['macro'] is not None else str(x['macro_bounds'])
    for method in METHODS:
        xs=[next(x for x in panels if x['arm']==method and x['mode']=='bank_R0' and x['prefix']==146 and x['task']==task) for task in ('T0','T1G','T2G','T1L','T2L')]
        lines.append('|'+method+'|'+'|'.join(cell(x) for x in xs)+'|')
    lines+=['','完整single/全部前缀、配对共享缺失界、编辑/来源区间、插入保持和实际FP16量化更新见RESULTS.json。',
            '既有TT/BalancEdit恢复版全部面板原值复现，历史raw/评分/失败不修改。所有新Judge一次尝试，缺分保持分母。',
            '本轮只用原146 Native/FIT和历史Native，未用评价标签；移除旧24编辑额外61 BASIS KL，是明确监督适配。',
            '纯W新增参数0，无路由；旧基线权重已消费，没有重训，配对同硬件时延记NA。原生新进程重载和本轮计时已完成。',
            'T1L仅1编辑2probe、T2L31编辑54probe，是历史局部性诊断，不是临床保护确认。单顺序开发数据，患者独立未知。',
            f'新增GPU进程小时{result["new_GPU_process_hours"]:.4f}；新Judge{result["new_Judge"]}。完整失败成本保留。']
    (R/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    c.write(R/'private/REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time()))
    print(json.dumps(dict(status=result['status'],DEV_gate=gate)))


if __name__=='__main__':
    main()
