"""Bounded X/Y diagnosis; terminal scores and per-item protection precede any expansion."""
from collections import Counter
import json,sqlite3,time
import astra_report as ar
from panels import protection_ids
from logic import transitions
from report_fs import audit_requests
R,c,r=ar.RUN,ar.common,ar.r

def main():
    cost=c.read(R/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in cost['gpu_sessions'])
    generated={a:sum(c.read(p)['outputs'] for p in (R/'private/outputs'/a).glob('*/*/*/COMPLETE.json')) for a in ('X','Y')}
    full=(R/'private/GENERATION_COMPLETE.json').exists() and (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists()
    diagnostics={}
    for arm in ('X','Y'):
        diagnostics[arm]=[]
        for owner in (1,2,3):
            events=sorted([c.read(p) for p in (R/'private/steps'/arm/str(owner)).glob('*.json')],key=lambda x:x['slot'])
            source=R/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json');source=c.read(source) if source.exists() else None
            reasons=Counter(reason for e in events if not e['accepted'] for reason in set(z for x in e['candidates'] for z in x['reasons']))
            diagnostics[arm].append(dict(source_position=owner,slots=len(events),accepted=sum(x['accepted'] for x in events),
                zero_fraction=sum(not x['accepted'] for x in events)/len(events) if events else None,rejection_flags=dict(reasons),
                source_measurements=source,last_event_risk=events[-1].get('native_risk_bound') if events else None,
                numeric_envelope_dominates_slots=sum(any(z.get('numeric_envelope_dominates') for z in e['candidates']) for e in events),
                gradient_retained_norm_fraction_mean=sum(x.get('gradient_retained_norm_fraction',0) for x in events)/len(events) if events else None))
    result=dict(status='PARTIAL_NOT_COMPLETE',scientific_success=False,diagnostic_only=True,generated=generated,diagnostics=diagnostics,
        new_Judge=cost['Judge_attempts'],GPU_process_hours=cost['gpu_seconds_used']/3600,
        actual_counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ['forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts']},
        failures=len(list((R/'private/failures').glob('*.json'))),followup146_activated=False,
        scalability_146='NOT_ADMITTED_FULL_SUPPORT_CPU_MEMORY_AND_STREAMING_PENDING',historical_results_preserved=True)
    if not full:
        assert list((R/'private/failures').glob('*.json')),'Healthy incomplete trajectory is not terminal'
        result.update(metrics='NA_UNFINISHED',unattempted_not_missing=True)
    else:
        db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
        rows=list(db.execute('SELECT * FROM consumer'));assert len(rows)==322
        payloads={x['key']:x for x in db.execute('SELECT * FROM payload')};scores={k:x['correct'] for k,x in payloads.items()};audit_requests(payloads,cost)
        tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json');records=[];lookup={}
        for x in rows:
            d=c.read(x['path']);assert c.digest(d)==x['output_binding'];bound=json.loads(payloads[x['payload_key']]['binding']);query=ledger['queries'][x['query_id']]
            assert bound['output']==d['R0'] and all(bound[k]==query[k] for k in ['query_id','question','reference','image_sha256'])
            active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:x['prefix']]} if x['mode']=='bank_R0' else set()
            records.append((dict(binding=dict(phase=dict(arm=x['method'],node=0,prefix=x['prefix']),input=query,mode=x['mode'],owner_order=x['edit_order']),active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),x['payload_key']))
            lookup[x['method'],x['mode'],x['prefix'],x['edit_order'],x['query_id']]=x['payload_key']
        panels=[];coefs={};groups={}
        for arm in ('X','Y'):
            for prefix in (1,3):
                for task in ('T0','T1G','T2G','T1L','T2L'):
                    metric,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,tasks[:prefix],prefix);metric.pop('route_ON',None);n=metric['observations']
                    metric['micro_bounds']=[100*metric['known_correct']/n,100*(metric['known_correct']+metric['missing'])/n] if n else None
                    panels.append(metric);coefs[arm,prefix,task]=co;groups.update(g)
        a,b=coefs['Y',3,'T2G'],coefs['X',3,'T2G'];assert set(a)==set(b)
        delta={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a};bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in x.items()} for x in delta.values()]),scores)
        contrast=dict(comparison='Y-minus-X',task='T2G',delta_macro_bounds_pp=[100*x for x in bounds],negative_edit_bounds=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),**ar.bootstrap(delta,scores,groups))
        protection=[];endpoints=[];gates={}
        for arm in ('X','Y'):
            protected=True
            for task in ('T1L','T2L'):
                ids=protection_ids(tasks,ledger,task);assert len(ids)==(2 if task=='T1L' else 34)
                keys=[lookup[arm,'protect100_R0',100,0,q] for q in ids];correct=sum(scores[k]==1 for k in keys);unknown=sum(scores[k] is None for k in keys)
                protection.append(dict(arm=arm,task=task,observations=len(ids),known_correct=correct,missing=unknown,Base_correct_to_wrong_bounds=[len(ids)-correct-unknown,len(ids)-correct]))
                protected &= correct==len(ids)
                if arm=='Y':endpoints.append(dict(task=task,comparison='X-to-Y',**transitions([(lookup['X','protect100_R0',100,0,q],lookup['Y','protect100_R0',100,0,q]) for q in ids],scores)))
            t1=next(x for x in panels if x['arm']==arm and x['prefix']==3 and x['task']=='T1G')
            gates[arm]=dict(zero_individual_Base_damage=protected,
                T0_insertion_and_final=all(scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]]==scores[lookup[arm,'bank_R0',3,0,t['edit_id']]]==1 for t in tasks[:3]),
                T1G_full_retention=t1['known_correct']==t1['observations'],full_scoring=all(scores[x['payload_key']] is not None for x in rows if x['method']==arm),
                all_source_CE_below_point1=all(x['source_measurements']['final']['weighted_all_FIT_CE']<.1 for x in diagnostics[arm]),
                no_numeric_envelope_domination=all(x['numeric_envelope_dominates_slots']==0 for x in diagnostics[arm]))
            for task in ('T0','T2G'):
                pairs=[]
                for t in tasks[:3]:
                    ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]
                    pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',3,0,q]) for q in dict.fromkeys(ids))
                endpoints.append(dict(arm=arm,task=task,comparison='actual-insertion-to-prefix3',**transitions(pairs,scores)))
        unknown=sum(scores[x['payload_key']] is None for x in rows);small_pass=all(gates['Y'].values()) and contrast['negative_edit_bounds']==0
        result.update(status='COMPLETE_WITH_MISSING' if unknown else 'COMPLETE',panels=panels,protection=protection,endpoints=endpoints,gates=gates,T2G_Y_minus_X=contrast,missing_consumers=unknown,
            quick_evidence_satisfactory=small_pass,followup146_condition='SCALABILITY_ADMISSION_REQUIRED' if small_pass else 'NOT_MET')
    c.write(R/'public/RESULTS.json',result)
    text='# ECW修订2有限诊断\n\n'+result['status']+'。完整终态不等于医学成功；三源验证不证明146可扩展。详见RESULTS.json。\n\n'
    text+='X=旧SVD写入/完整native采样风险，Y=全W写入/相同风险；两臂Gamma=I、epsilon=.01、proposal cap=.05、相同当前医学范围/证据资格。旧版R1白化/区域配置不同，其结果仅作历史诊断，不称单因素消融。\n\n|臂|源位置|槽|接受|零更新|末端完整源CE|\n|---|---:|---:|---:|---:|---:|\n'
    for a,ls in diagnostics.items():
        for x in ls:text+='|'+ '|'.join(str(z) for z in [a,x['source_position'],x['slots'],x['accepted'],x['zero_fraction'],x['source_measurements']['final']['weighted_all_FIT_CE'] if x['source_measurements'] else 'NA'])+'|\n'
    text+='\n0.01是冻结线性采样KL代理阈值。数值包络只覆盖声明的FP64运算误差，不覆盖采样/Taylor余项、临床错误或隐私。保护34 T2L+2 T1L逐题零新增损伤，缺分未知不能通过；平均和修复不能抵消损伤。未生成/未评分不是永久missing。原146/M暂停，O/L等待，F/S原GPU6队列保持。\n'
    (R/'public/REPORT_ZH.md').write_text(text);c.write(R/'private/REPORT_COMPLETE.json',dict(status=result['status'],epoch=time.time()));print(result['status'])

if __name__=='__main__':main()
