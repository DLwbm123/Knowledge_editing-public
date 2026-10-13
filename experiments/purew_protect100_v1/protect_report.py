"""Anonymous C/K comparison with original masks, shared missing and hard gates."""
from collections import defaultdict
import json
import math
import os
import sqlite3
import sys
import time
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_report as ar
from logic import transitions

R, c, r = ar.RUN, ar.common, ar.r


def main():
    root=R/'private/judge_purew146'
    cost=c.read(R/'RESOURCE_LEDGER.json')
    assert not any(not x.get('ended_epoch') for x in cost['gpu_sessions'])
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=list(db.execute('SELECT * FROM consumer'));payloads={x['key']:x for x in db.execute('SELECT * FROM payload')}
    scores={k:x['correct'] for k,x in payloads.items()}
    coverage=dict(db.execute('SELECT method,count(*) FROM consumer GROUP BY method'))
    unresolved=db.execute("SELECT count(*) FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()[0]
    complete=(R/'private/GENERATION_COMPLETE.json').exists() and (root/'ALL_WORKERS_COMPLETE.json').exists() and not unresolved
    if complete: assert coverage.get('K')==2181 and (R/'private/FINAL_CONSUMED.json').exists()
    attempted=[]
    from scripts.medtrace.astra_judge_bundle import validate
    for batch in cost['Astra_purew146_batches']:
        attempted += batch['keys']
        if batch['status']=='RESERVED':
            assert not complete;continue
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
    attempts=[c.read(p) for p in (R/'private/attempts').glob('*.json')]
    counts={k:sum(x.get(k,0) for x in attempts) for k in ('forwards','backwards','committed_updates','generation_attempts')}
    markers=list((R/'private/outputs').glob('*/*/*/COMPLETE.json'))
    generated=sum(c.read(p)['outputs'] for p in markers)
    result=dict(status='COMPLETE_WITH_MISSING' if complete else 'PARTIAL_NOT_ALL_CONSUMERS_COMPLETE',
                scientific_success=False,coverage=coverage,expected_K_consumers=2181,
                generated_K_consumers=generated,not_generated_K=2181-generated,
                unscored_generated_K=max(0,generated-coverage.get('K',0)),
                scoring_statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
                inherited_old_permanent_missing=11,new_Judge_attempts=cost['Judge_attempts'],
                failed_new_Judge_batches=sum(x['status']=='FAILED_NO_RETRY' for x in cost['Astra_purew146_batches']),
                actual_compute_counts=counts,new_GPU_process_hours=cost['gpu_seconds_used']/3600,
                all_GPU_leases_retired=True,owned_weight_peak_bytes=cost['weights_observed_peak_bytes'],
                failures=len(list((R/'private/failures').glob('*.json'))),panels=[],contrasts=[],endpoint_transitions=[],gates={},
                original146='PAUSED_UNMODIFIED',OL='ORIGINAL_DEPENDENCY_UNCHANGED',new_inference_parameters=0,
                limitations=['K-C combines the additional protection objective and61source prompts; not KL-versus-data attribution',
                    'Matched FP32 proposal is not matched native/function perturbation',
                    'C T2G insertion unavailable; independent singles are not insertion measurements',
                    'First100 developmental sequence chosen after seeing old results; patient independence unknown',
                    'Locality support is small; no clinical/SOTA confirmation'])
    if (R/'public/REAL_VLM_ADMISSION.json').exists():result['real_VLM']=c.read(R/'public/REAL_VLM_ADMISSION.json')
    if (R/'public/ADMISSION_REJECTED.json').exists():result['admission_rejected']=c.read(R/'public/ADMISSION_REJECTED.json')
    if complete:
        tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][:100];ledger=c.read(R/'private/EVAL_LEDGER.json')
        records=[];lookup={}
        for x in rows:
            query=ledger['queries'][x['query_id']]
            full=json.loads(payloads[x['payload_key']]['binding'])
            assert full['query_id']==query['query_id'] and all(full[k]==query[k] for k in ('question','reference','image_sha256'))
            if x['method']=='K':
                d=c.read(x['path']);assert c.digest(d)==x['output_binding']
                assert full['output']==d['R0'] and d['binding']['input']==query
                assert d['binding']['phase']['protocol']==c.digest(c.read(R/'private/FROZEN_PROTOCOL.json'))
            active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:x['prefix']]} if x['mode']=='bank_R0' else set()
            records.append((dict(binding=dict(phase=dict(arm=x['method'],node=0,prefix=x['prefix']),input=query,
                            mode=x['mode'],owner_order=x['edit_order']),active_target=(query['image_sha256'],query['question']) in active,
                            effective_expert=None),x['payload_key']))
            key=(x['method'],x['mode'],x['prefix'],x['edit_order'],x['query_id']);assert key not in lookup;lookup[key]=x['payload_key']
        coeff={};groups={}
        scopes=[(a,'bank_R0',n,n) for a in ('C','K') for n in (1,50,100)]
        scopes += [('C','single_R0',1,100)] + [(a,'bank_R0',100,100) for a in ('TT88_W0_RETRO146','balancedit')]
        for arm,mode,prefix,n in scopes:
            for task in ('T0','T1G','T2G','T1L','T2L'):
                metric,co,g=r.panel(records,scores,'RETRO146',arm,0,mode,task,tasks[:n],prefix)
                metric.pop('route_ON');count=metric['observations']
                metric.update(mode=mode,prefix=prefix,native_edit_count=n)
                metric['micro_bounds']=[100*metric['known_correct']/count,100*(metric['known_correct']+metric['missing'])/count] if count else None
                if task.endswith('L'):metric['Base_correct_to_wrong_count_bounds']=[count-metric['known_correct']-metric['missing'],count-metric['known_correct']]
                result['panels'].append(metric);coeff[arm,mode,prefix,n,task]=co;groups.update(g)
        def compare(name,aa,bb,task):
            a=coeff[(*aa,task)];b=coeff[(*bb,task)];assert set(a)==set(b)
            delta={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
            bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in row.items()} for row in delta.values()]),scores)
            item=dict(comparison=name,task=task,delta_macro_bounds_pp=[100*x for x in bounds],
                      edits_negative_lower_bound=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),
                      edits_definitely_worse=sum(r.score_bounds(x,scores)[1]<-1e-12 for x in delta.values()))
            if task=='T2G':item.update(ar.bootstrap(delta,scores,groups))
            result['contrasts'].append(item);return item
        for n in (50,100):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                compare('K-minus-C-prefix'+str(n),('K','bank_R0',n,n),('C','bank_R0',n,n),task)
        for task in ('T2G','T2L'):
            compare('K-prefix100-minus-independent-C-single100',('K','bank_R0',100,100),('C','single_R0',1,100),task)
        for baseline in ('TT88_W0_RETRO146','balancedit'):
            compare('K-minus-'+baseline+'-prefix100',('K','bank_R0',100,100),(baseline,'bank_R0',100,100),'T2G')
        for arm in ('C','K'):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                pairs=[]
                for t in tasks[:50]:
                    ids=list(dict.fromkeys(q for e in t['events'] if e['task']==task for q in e['all_probe_query_ids']))
                    if task!='T0':ids=[q for q in ids if ledger['Base_correctness'][q]==task.endswith('L')]
                    pairs += [(lookup[arm,'bank_R0',50,0,q],lookup[arm,'bank_R0',100,0,q]) for q in ids]
                result['endpoint_transitions'].append(dict(method=arm,task=task,comparison='same-first50-prefix50-to100',**transitions(pairs,scores)))
        endpoints={}
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks:
                ids=[t['edit_id']] if task=='T0' else list(dict.fromkeys(q for e in t['events'] if e['task']=='T2G'
                                for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]))
                pairs += [(lookup['K','insertion_R0',t['order'],t['order'],q],lookup['K','bank_R0',100,0,q]) for q in ids]
            endpoints[task]=transitions(pairs,scores)
            result['endpoint_transitions'].append(dict(method='K',task=task,comparison='actual-insertion-to-prefix100',**endpoints[task]))
        result['C_T2G_insertion']='NA_NOT_MEASURED'
        final=[x for x in result['panels'] if x['arm']=='K' and x['mode']=='bank_R0' and x['prefix']==100]
        primary=next(x for x in result['contrasts'] if x['comparison']=='K-minus-C-prefix100' and x['task']=='T2G')
        primary_rows=[x for x in rows if x['method'] in ('C','K')]
        native_insert=[scores[x['payload_key']] for x in rows if x['method']=='K' and x['mode']=='insertion_R0' and x['query_id']==tasks[x['edit_order']-1]['edit_id']]
        result['gates']=dict(protection_zero_Base_damage=all(x['known_correct']==x['observations'] for x in final if x['task'] in ('T1L','T2L')),
            T0_insert_and_final_all_correct=len(native_insert)==100 and all(x==1 for x in native_insert) and next(x for x in final if x['task']=='T0')['known_correct']==100,
            T2G_paired_nonnegative=primary['delta_macro_bounds_pp'][0]>=0,
            T2G_per_edit_no_regression=primary['edits_negative_lower_bound']==0,
            T2G_no_insertion_correct_loss=endpoints['T2G']['count_bounds']['10'][1]==0 and endpoints['T2G']['unresolved_pairs']==0,
            full_required_scoring=all(scores[x['payload_key']] is not None for x in primary_rows))
        events=[c.read(p) for p in (R/'private/steps/bank_R0').glob('*/*.json')]
        assert len(events)==16000 and {(x['owner'],x['step']) for x in events}=={(i,s) for i in range(1,101) for s in range(1,161)}
        by=defaultdict(list)
        for event in events:
            assert all(math.isfinite(event[k]) for k in ('proposal_norm','actual_native_norm','BASIS_KL','edit_native_net_displacement'))
            assert abs(event['proposal_norm']-event['target_proposal_norm'])<=max(1e-8,1e-5*event['target_proposal_norm'])
            by[event['owner']].append(event)
        ratios=[sum(x['actual_native_norm'] for x in es)/sum(x['C_native_norm'] for x in es) for es in by.values()]
        for es in by.values(): es.sort(key=lambda x:x['step'])
        diagnostics=[x['diagnostics'] for x in events if x['diagnostics']]
        result['diagnostics']=dict(steps=16000,zero_native_writes=sum(x['changed_elements']==0 for x in events),
            mean_BASIS_KL=sum(x['BASIS_KL'] for x in events)/16000,
            proposal_path_length=sum(x['proposal_norm'] for x in events),native_path_length=sum(x['actual_native_norm'] for x in events),
            native_path_ratio_range=[min(ratios),max(ratios)],native_path_confounded_edits=sum(not .9<=x<=1.1 for x in ratios),
            edit_net_displacement_range=[min(es[-1]['edit_native_net_displacement'] for es in by.values()),max(es[-1]['edit_native_net_displacement'] for es in by.values())],
            sampled_gradient_nodes=len(diagnostics),mean_sampled_KL_gradient_norm=sum(x['KL_gradient_norm'] for x in diagnostics)/len(diagnostics),
            mean_native_master_discrepancy=sum(x['native_master_discrepancy'] for x in events)/16000)
        result['diagnostics']['gradient_samples']={key:dict(n=len(values),mean=sum(values)/len(values),minimum=min(values),maximum=max(values))
            for key in ('source_gradient_norm','history_gradient_norm','KL_gradient_norm','source_history_cosine','source_KL_cosine','history_KL_cosine')
            if (values:=[x[key] for x in diagnostics if x[key] is not None])}
        result['gates']['native_path_attribution']=all(.9<=x<=1.1 for x in ratios)
        result['scientific_success']=all(result['gates'].values())
        result['fresh_native_reload']=c.read(R/'public/NATIVE_RELOAD.json')
    c.write(R/'public/RESULTS.json',result)
    lines=['# 连续纯W KL保护对照','',f'状态：{result["status"]}；完整科学门槛：{result["scientific_success"]}。',
           '原146保持暂停，O/L原依赖不变。保护源仅为准入61 BASIS；评价题不训练。','',
           '|方法/设置|T0正确|T2G宏平均上下界%|T2L Base损伤界|','|---|---:|---:|---:|']
    for arm,mode in (('C','bank_R0'),('K','bank_R0'),('C','single_R0')):
        ps=[x for x in result['panels'] if x['arm']==arm and x['mode']==mode and x['native_edit_count']==100]
        if ps:
            t0=next(x for x in ps if x['task']=='T0');tg=next(x for x in ps if x['task']=='T2G');tl=next(x for x in ps if x['task']=='T2L')
            lines.append(f'|{arm}/{mode}|{t0["known_correct"]}/{t0["observations"]}|{tg["macro_bounds"]}|{tl["Base_correct_to_wrong_count_bounds"]}/{tl["observations"]}|')
    lines += ['', '完整分母、共享missing精确界、10,000次原seed编辑/来源区间和逐题转移见RESULTS.json。',
              'C没有T2G插入时输出，记NA；独立单例不作插入测量。K真实插入到最终的遗忘与修复分别列出，平均不补偿损伤。',
              'KL是软保护，零损伤是验收门槛而非优化保证。继承及新增永久缺分不补判；未生成/未尝试不作missing。',
              '等FP32 proposal不保证等native或功能扰动，超路径幅度门的编辑保留。新目标和额外61数据不能分别归因。',
              f'新增GPU小时{result["new_GPU_process_hours"]:.4f}；新Judge{result["new_Judge_attempts"]}，失败与实际计算成本全部保留。',
              '全部证据属于已观察开发单顺序；局部性支持小、患者独立未知，无临床/SOTA确认。']
    (R/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    c.write(R/'private/REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time()))
    print(json.dumps(dict(status=result['status'],scientific_success=result['scientific_success'])))


if __name__=='__main__':main()
