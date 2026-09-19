"""CPU-only Stage23 export from completed, frozen consumers; never invokes a Judge."""
import json,sys,csv
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs,expected
from scripts.medtrace.stage18_score import query_id,score_key


def run():
    d=Path(__file__).parent;source=ROOT/'reports/medtrace_stage22_20260919';root=source/'private/run';p=root/'private';pub=root/'public'
    assert read(pub/'FINAL_CONTROLLER_STATUS.json')['phase']=='23_COMPLETE_SCORED' and read(pub/'EXIT_23.json')['exit_code']==0
    result=read(pub/'RESULTS_23.json');assert result['complete'] and result['highest_common_scored_N']==45
    stream=read(p/'STREAM.json');lock=read(source/'private/FINAL_STAGE23_LOCK.json');panel=read(source/'private/CONFIRM_QUALIFIED_FREEZE.json');scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores']
    rows=[r for r in outputs(root) if r['arm'].startswith('R_')];base=read(p/'CONFIRM_BASE_23.json')['records']
    identities={(r['arm'],r['mode'],r['prefix'],r['query_id']) for r in rows}
    assert len(rows)==len(identities)==909 and len(base)==5
    assert all(type(scores.get(score_key(r['source'],r[field]))) is bool for r in rows for field in ('Base','output'))
    assert all(type(scores.get(score_key(r['source'],r['output']))) is bool for r in base)
    ledger=read(pub/'GPU_BUDGET_LEDGER.json');judge=read(pub/'BUDGET_LEDGER.json')
    assert all(s.get('exit_code')==0 and 'seconds' in s for s in ledger['sessions'])
    assert sum(s['seconds'] for s in ledger['sessions'])<=28800 and judge['new_judgment_items_dispatched']<=2000
    assert all(x['status']=='COMPLETE' for x in judge['judgment_batches'])
    public_judge={**judge,'judgment_batches':[{k:v for k,v in b.items() if k not in ('thread_id','runtime_warnings')} for b in judge['judgment_batches']]}
    assert '/Users/' not in json.dumps(public_judge) and 'thread_id' not in json.dumps(public_judge)
    for dest,value in [('RESULTS.json',result),('PAIRED_SOURCE_RESULTS.json',read(pub/'PAIRED_SOURCE_23.json')),('FINAL_METHOD_LOCK.json',lock),('GPU_BUDGET_LEDGER.json',ledger),('JUDGE_BUDGET_LEDGER.json',public_judge),('CONFIRM_QUALIFICATION_SUMMARY.json',read(source/'CONFIRM_QUALIFICATION_SUMMARY.json'))]:write(d/dest,value)
    table=[]
    for r in result['results']:
        for name,v in r['metrics'].items():table.append(dict(arm=r['arm'],N=r['N'],panel=r['panel'],metric=name,correct=v['correct'],denominator=v['N'],scored=v['coverage'],QA_accuracy=v['accuracy'],source_macro=v['source_macro'],sources=v['sources'],leave_one_source_out=v['leave_one_source_out']))
    with (d/'RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator="\n");w.writeheader();w.writerows(table)
    order={t['canonical_edit_id']:t['order'] for t in stream['tasks']};routing=[];pairs=[]
    def correct(r):return scores[score_key(r['source'],r['output'])]
    for arm in lock['banks']:
        previous={}
        for n in lock['prefixes']:
            current={r['query_id']:r for r in rows if r['arm']=='R_'+arm and r['prefix']==n and r['mode']=='endpoint'}
            assert set(current)==set(expected(stream,n,final=n==45))
            transitions=Counter();roles=defaultdict(Counter)
            for q in previous.keys()&current.keys():
                a,b=previous[q],current[q];switched=(a['route']['logical_edit_id'],a['route']['activated'])!=(b['route']['logical_edit_id'],b['route']['activated'])
                label=f'{int(correct(a))}->{int(correct(b))}|switch={switched}';transitions[label]+=1;roles[b['source']['role']][label]+=1
            routing.append(dict(arm=arm,N=n,queries=len(current),activated=sum(r['route']['activated'] for r in current.values()),actual_expert_positions=dict(Counter(order.get(r['route']['logical_edit_id'],0) for r in current.values())),nearest_expert_positions=dict(Counter(order.get(r['route']['nearest_logical_edit_id'],0) for r in current.values())),prefix_transitions=dict(transitions),prefix_transitions_by_role=dict(roles),writer_layer=30,router_feature_layer=31))
            values=[]
            for pack in stream['anchor_packages']:
                q=query_id(pack['training']['native'])
                if q not in current:continue
                h=[current[query_id(r)] for r in pack['evaluation'] if r['role']=='H_eval'];values.append(int(correct(current[q]))*mean(correct(r) for r in h))
            pairs.append(dict(arm=arm,N=n,PairCorrect=mean(values),anchor_N=len(values),unit='anchor-weighted mean, not independent H QA',old_H_unique_QA=6,old_H_sources=2))
            previous=current
    write(d/'ROUTING_PREFIX_TRANSITIONS.json',dict(rows=routing,natural_routing=True,anonymous_expert_positions=True,zero_position='Base',conditional_results_do_not_replace_main=True))
    write(d/'ANCHOR_PAIR_CORRECT.json',dict(rows=pairs,caution='Repeated H associations must not be treated as independent QA or sources'))
    def metric(arm,key):return next(r['metrics'][key] for r in result['results'] if r['arm']==arm and r['panel']=='CONFIRM_source_probe')
    candidate=lock['candidate'];noh=lock['NOH'];h1=lock['H1']
    checks=dict(H_source_macro_retention_delta_vs_NOH=metric(candidate,'H_eval_Retention')['source_macro']-metric(noh,'H_eval_Retention')['source_macro'],H_retention_no_worse_than_H1=metric(candidate,'H_eval_Retention')['source_macro']>=metric(h1,'H_eval_Retention')['source_macro'],H_accuracy_no_worse_than_NOH=metric(candidate,'H_eval_accuracy')['source_macro']>=metric(noh,'H_eval_accuracy')['source_macro'],U_strict_retention_no_worse_than_NOH=metric(candidate,'U_eval_strict_Retention')['source_macro']>=metric(noh,'U_eval_strict_Retention')['source_macro'],positive_generalization=None)
    write(d/'CONFIRM_REQUIREMENT_AUDIT.json',dict(official_gate=result['engineering_gate'],observed_checks=checks,observed_H_retention_regression=checks['H_source_macro_retention_delta_vs_NOH']<0,adequate_confirmation_panel=False,note='Insufficient panel does not erase the observed H failure; no post-confirmation tuning or new semantic judgments'))
    usage=Counter()
    for batch in judge['judgment_batches']:
        for k,v in (batch.get('usage') or {}).items():
            if isinstance(v,(int,float)):usage[k]+=v
    write(d/'DELIVERY_STATUS.json',dict(stage23='COMPLETE_GENERATED_SCORED_ANALYZED',all_banks_native_correct_45=result['all_native_45_correct'],formal_regression_outputs=894,CONFIRM_student_outputs=15,CONFIRM_Base_outputs=5,all_output_consumers_scored=True,new_stage23_judgments=judge['23_items_dispatched'],all_follow_on_new_judgments=judge['new_judgment_items_dispatched'],follow_on_GPU_seconds=sum(s['seconds'] for s in ledger['sessions']),remaining_GPU_seconds=ledger['limit_seconds']-sum(s['seconds'] for s in ledger['sessions']),remaining_judgments=judge['limit']-judge['new_judgment_items_dispatched'],consumer_exact_reuses=judge['exact_reuse_count'],exact_reuses_are_not_independent_QA=True,provider_usage_totals=dict(usage),actual_cost=None,cost_status='Provider currency billing not exposed; never zero',Stage20_Stage21_formal_artifacts_untouched=True,patient_study='UNKNOWN',clinical_validation=False))
    c=read(d/'CHECKPOINT_CONSUMERS.json');c['completed_consumers']=list(dict.fromkeys(c['completed_consumers']+['all isolated scoring','CPU source/routing analysis']));c['pending_at_inventory']=['verified public delivery'];write(d/'CHECKPOINT_CONSUMERS.json',c)
    print(json.dumps(dict(scored=len(rows)+len(base),new_judgments=judge['new_judgment_items_dispatched'],GPU_seconds=sum(s['seconds'] for s in ledger['sessions']),gate=result['engineering_gate'],checks=checks)))


if __name__=='__main__':run()
