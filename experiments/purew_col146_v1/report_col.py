"""Full denominator C/O/L singles, exact shared missing bounds and hard protection."""
from collections import defaultdict
import json
import math
import sqlite3
import time
import astra_report as ar
from validate import protected_counts

R,c,r=ar.RUN,ar.common,ar.r


def main():
    root=R/'private/judge_purew146'
    cost=c.read(R/'RESOURCE_LEDGER.json')
    assert all(x.get('ended_epoch') for x in cost['gpu_sessions']), 'GPU work still active'
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=list(db.execute('SELECT * FROM consumer'))
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')}
    complete=(root/'ALL_WORKERS_COMPLETE.json').exists() and (R/'private/FINAL_CONSUMED.json').exists()
    coverage=dict(db.execute('SELECT method,count(*) FROM consumer GROUP BY method'))
    scores={k:x['correct'] for k,x in payloads.items()}
    assert all(x['status'] in ('FORMAT_VALID','MISSING') for x in payloads.values()) if complete else True
    from scripts.medtrace.astra_judge_bundle import validate
    attempted=[]
    for batch in cost['Astra_purew146_batches']:
        attempted.extend(batch['keys'])
        if batch['status']=='RESERVED':
            assert not complete
            continue
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
    records=[];lookup={}
    for row in rows:
        d=c.read(row['path']);assert c.digest(d)==row['output_binding']
        if row['method']!='C':assert d['binding']['phase']['protocol']==c.digest(c.read(R/'private/FROZEN_PROTOCOL.json'))
        assert json.loads(payloads[row['payload_key']]['binding'])['output']==d['R0']
        assert d['binding']['input']==ledger['queries'][row['query_id']]
        lookup[row['method'],row['edit_order'],row['query_id']]=row['payload_key']
        records.append((dict(binding=dict(phase=dict(arm=row['method'],node=0,prefix=1),input=ledger['queries'][row['query_id']],
            mode='single_R0',owner_order=row['edit_order']),active_target=False,effective_expert=None),row['payload_key']))
    result=dict(status='COMPLETE_WITH_MISSING' if complete else 'PARTIAL_NOT_ALL_CONSUMERS_COMPLETE',
        scientific_success=False,expected_consumers_per_arm=2372,coverage=coverage,
        not_generated={arm:2372-coverage.get(arm,0) for arm in ('C','O','L')},
        scoring_statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        C_permanent_missing=11,primary='single146 edit-macro T2G Fix',new_GPU_process_hours=cost['gpu_seconds_used']/3600,
        new_Judge=cost['Judge_attempts'],new_Judge_failed_payloads=cost.get('Judge_failed_payloads',0),
        GPU_sessions=len(cost['gpu_sessions']),all_GPU_leases_retired=True,owned_weight_peak_bytes=cost['weights_observed_peak_bytes'],
        owned_pt_remaining=sum(1 for p in (R/'private').rglob('*.pt') if not p.is_symlink()),
        failures=len(list((R/'private').rglob('failures/*.json'))),panels=[],contrasts=[],protection={},gates={},diagnostics={},
        limitations=['Conditional direction/layer effects at the fixed C FP32 proposal schedule; not ordinary Adam best performance',
            'Equal FP32 proposal does not establish equal FP16 or functional perturbation',
            'No new continual trajectory; no insertion-to-final T2G retention claim',
            'Development single order, patient independence unknown; T1L 1 edit/2 probes, T2L 31 edits/54 probes'])
    if complete:
        assert coverage=={'C':2372,'L':2372,'O':2372} and len(rows)==7116
        coeff={};groups={}
        for arm in ('C','O','L'):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                panel,co,g=r.panel(records,scores,'RETRO146',arm,0,'single_R0',task,tasks,1)
                panel.pop('route_ON');panel.update(ar.bootstrap(co,scores,g))
                assert panel['observations']==dict(T0=146,T1G=578,T2G=557,T1L=2,T2L=54)[task]
                panel['micro_bounds']=[100*panel['known_correct']/panel['observations'],100*(panel['known_correct']+panel['missing'])/panel['observations']]
                result['panels'].append(panel);coeff[arm,task]=co;groups.update(g)
        for arm in ('O','L'):
            failures={}
            for task in ('T0','T1G','T2G','T1L','T2L'):
                aa,bb=coeff[arm,task],coeff['C',task];assert set(aa)==set(bb)
                delta={e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
                bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in row.items()} for row in delta.values()]),scores)
                failures[task]=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values())
                result['contrasts'].append(dict(comparison=arm+'-C',task=task,delta_bounds_pp=[100*x for x in bounds],
                    per_edit_negative_lower_bounds=failures[task],**ar.bootstrap(delta,scores,groups)))
            protected={}
            for task,expected in (('T1L',2),('T2L',49)):
                pairs=[]
                for t in tasks:
                    ids=list(dict.fromkeys(q for e in t['events'] if e['task']==task for q in e['all_probe_query_ids']))
                    ids=[q for q in ids if ledger['Base_correctness'][q]]
                    pairs.extend((lookup['C',t['order'],q],lookup[arm,t['order'],q]) for q in ids)
                z=protected_counts(pairs,scores);assert z['denominator']==expected
                z.update(Base_correct_denominator=len(pairs),Base_damage_bounds=[sum(scores[b]==0 for a,b in pairs),sum(scores[b]!=1 for a,b in pairs)])
                protected[task]=z
            result['protection'][arm]=protected
            own=[x for x in result['panels'] if x['arm']==arm]
            primary=next(x for x in result['contrasts'] if x['comparison']==arm+'-C' and x['task']=='T2G')
            result['gates'][arm]=dict(T2G_positive_robust_bound=primary['delta_bounds_pp'][0]>0,
                T2G_per_edit_no_regression=failures['T2G']==0,
                T0_T1G_all_retained=all(x['known_correct']==x['observations'] for x in own if x['task'] in ('T0','T1G')),
                protection_no_new_damage=all(x['passed'] for x in protected.values()),
                Base_T2L_damage_at_most_C5=protected['T2L']['Base_damage_bounds'][1]<=5,
                full_required_scoring=all(scores[x['payload_key']] is not None for x in rows if x['method'] in ('C',arm)))
            A=R/'private/arms'/arm;events=[c.read(p) for p in (A/'private/steps/single_R0').glob('*/*.json')]
            assert len(events)==23360 and {(x['owner'],x['step']) for x in events}=={(i,s) for i in range(1,147) for s in range(1,161)}
            assert all(math.isfinite(x[k]) for x in events for k in ('proposal_norm','actual_native_norm','native_master_discrepancy'))
            assert all(abs(x['proposal_norm']-x['target_proposal_norm'])<=max(1e-8,1e-5*x['target_proposal_norm']) for x in events)
            by=defaultdict(list)
            for x in events:by[x['owner']].append(x)
            ratios=[sum(x['actual_native_norm'] for x in xs)/sum(x['C_native_norm'] for x in xs) for xs in by.values()]
            result['diagnostics'][arm]=dict(steps=len(events),zero_native_writes=sum(x['changed_elements']==0 for x in events),
                mean_proposal_norm=sum(x['proposal_norm'] for x in events)/len(events),mean_actual_native_norm=sum(x['actual_native_norm'] for x in events)/len(events),
                mean_native_master_discrepancy=sum(x['native_master_discrepancy'] for x in events)/len(events),
                native_path_ratio_range=[min(ratios),max(ratios)],native_path_confounded_edits=sum(not .9<=x<=1.1 for x in ratios),
                real_VLM=c.read(A/'public/REAL_VLM_ADMISSION.json'),fresh_reload=c.read(A/'public/NATIVE_RELOAD.json'))
            result['gates'][arm]['native_path_attribution']=all(.9<=x<=1.1 for x in ratios)
        result['scientific_success']=all(all(g.values()) for g in result['gates'].values())
    c.write(R/'public/RESULTS.json',result)
    lines=['# 原146 C/O/L单例对照','',f'执行状态：{result["status"]}；完整科学成功门槛：{result["scientific_success"]}。',
        'C原11缺分永久保留。未生成/未尝试与永久missing分别记录；任何缺分不算保护通过。','',
        '| 方法 | T0 | T1G | T2G宏平均上下界% | T1L | T2L |','|---|---:|---:|---:|---:|---:|']
    for arm in ('C','O','L'):
        panels=[x for x in result['panels'] if x['arm']==arm]
        if panels:lines.append('|'+arm+'|'+'|'.join(str(x['macro_bounds']) for x in panels)+'|')
    lines+=['','完整分母、微平均、共享missing配对精确界、10,000次原seed编辑/来源区间、逐题保护和逐编辑退化见RESULTS.json。',
        'O仅改变等FP32 proposal范数下的Adam方向；L改变原W层位。实际FP16路径与功能扰动不保证相同，幅度混杂保留，不删点调参。',
        '本轮没有连续写入实验。原146插入仅测T0；两端正确不证明中间从未遗忘，T2G的完整插入至最终保持不可识别。',
        '49个C正确T2L与2个T1L逐题零新增损伤，同时报告全部54个Base正确T2L和C已有5个损伤。T1L/T2L仅历史局部性诊断，患者独立未知。',
        f'新增GPU进程小时：{result["new_GPU_process_hours"]:.4f}；新Judge尝试：{result["new_Judge"]}；失败成本保留。']
    (R/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    c.write(R/'private/REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time()))
    print(json.dumps(dict(status=result['status'],scientific_success=result['scientific_success'])))


if __name__=='__main__':main()
