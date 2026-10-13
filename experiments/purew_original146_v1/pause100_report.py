"""Anonymous partial report: prefix100, matched singles and first50 progression."""
from collections import Counter
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_report as ar
from contract import ARM,METHODS
from prefix50_single_analysis import transitions,selfcheck

R,c,r=ar.RUN,ar.common,ar.r


def main():
    selfcheck()
    pause=c.read(R/'private/PAUSED_USER_REQUEST.json');assert pause['status']=='PAUSED_USER_REQUEST'
    assert not (R/'private/GENERATION_COMPLETE.json').exists() and not (R/'private/FINAL_CONSUMED.json').exists()
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('BEGIN')
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    rows=list(db.execute("SELECT * FROM consumer WHERE (mode='single_R0' AND edit_order<=100) OR (mode='bank_R0' AND prefix IN (50,100))"))
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')};scores={k:x['correct'] for k,x in payloads.items()}
    assert all(payloads[x['payload_key']]['status'] in ('FORMAT_VALID','MISSING') for x in rows)
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][:100];ledger=c.read(R/'private/EVAL_LEDGER.json')
    marker=c.read(R/'private/outputs/bank_R0/0/100/COMPLETE.json')
    assert Counter(x['method'] for x in rows if x['mode']=='bank_R0' and x['prefix']==100)=={m:marker['outputs'] for m in METHODS}
    records=[];lookup={}
    for x in rows:
        if x['method']==ARM and x['mode']=='bank_R0' and x['prefix']==100:
            d=c.read(x['path']);assert c.digest(d)==x['output_binding'] and d['binding']['phase']==marker['phase']
            assert json.loads(payloads[x['payload_key']]['binding'])['output']==d['R0']
        query=ledger['queries'][x['query_id']]
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:x['prefix']]} if x['mode']=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=x['method'],node=0,prefix=x['prefix']),input=query,mode=x['mode'],owner_order=x['edit_order']),active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),x['payload_key']))
        key=(x['method'],x['mode'],x['prefix'],x['edit_order'],x['query_id']);assert key not in lookup;lookup[key]=x['payload_key']
    scopes=dict(single100=('single_R0',1,100),prefix100=('bank_R0',100,100),prefix50_first50=('bank_R0',50,50),prefix100_first50=('bank_R0',100,50))
    panels=[];coefs={};groups={}
    for name,(mode,prefix,n) in scopes.items():
        for method in METHODS:
            for task in ('T0','T1G','T2G','T1L','T2L'):
                metric,coef,g=r.panel(records,scores,'RETRO146',method,0,mode,task,tasks[:n],prefix)
                metric.pop('route_ON');metric['scope']=name;observed=metric['observations'];a=metric['known_correct'];u=metric['missing']
                metric['micro_bounds']=[100*a/observed,100*(a+u)/observed]
                if task.endswith('L'):metric['Base_correct_to_wrong_count_bounds']=[observed-a-u,observed-a]
                panels.append(metric);coefs[name,method,task]=coef;groups.update(g)
    contrasts=[]
    def compare(name,method_a,scope_a,method_b,scope_b,task,pair_n=None):
        a,b=coefs[scope_a,method_a,task],coefs[scope_b,method_b,task];assert set(a)==set(b)
        delta={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
        bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in unit.items()} for unit in delta.values()]),scores)
        directions=dict(worse=sum(r.score_bounds(v,scores)[1]<-1e-12 for v in delta.values()),better=sum(r.score_bounds(v,scores)[0]>1e-12 for v in delta.values()),equal=sum(max(abs(x) for x in r.score_bounds(v,scores))<1e-12 for v in delta.values()))
        directions['uncertain']=len(delta)-sum(directions.values())
        out=dict(comparison=name,method=method_a,baseline=method_b,task=task,delta_macro_bounds_pp=[100*x for x in bounds],edit_direction_counts=directions)
        if task=='T2G':out.update(ar.bootstrap(delta,scores,groups))
        if pair_n:
            pairs=[];ma,pa,_=scopes[scope_a];mb,pb,_=scopes[scope_b]
            for t in tasks[:pair_n]:
                qids=list(dict.fromkeys(q for e in t['events'] if e['task']==task for q in e['all_probe_query_ids']))
                if task!='T0':qids=[q for q in qids if ledger['Base_correctness'][q]==task.endswith('L')]
                for qid in qids:
                    if task.endswith('L'):
                        query=ledger['queries'][qid]
                        assert not any(mode=='bank_R0' and (query['image_sha256'],query['question']) in {(x['native']['image_sha256'],x['native']['question']) for x in tasks[:prefix]} for mode,prefix in ((ma,pa),(mb,pb)))
                    # Transitions run from the earlier/comparator setting to prefix100.
                    pairs.append((lookup[method_b,mb,pb,t['order'] if mb=='single_R0' else 0,qid],lookup[method_a,ma,pa,t['order'] if ma=='single_R0' else 0,qid]))
            out['transitions']=transitions(pairs,scores)
            assert len(pairs)==next(p['observations'] for p in panels if p['scope']==scope_a and p['arm']==method_a and p['task']==task)
        contrasts.append(out)
    for method in METHODS:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            compare('prefix100-minus-matched-single100',method,'prefix100',method,'single100',task,100)
            compare('prefix100-minus-prefix50-on-first50',method,'prefix100_first50',method,'prefix50_first50',task,50)
    for baseline in METHODS[1:]:
        for task in ('T0','T1G','T2G','T1L','T2L'):compare('prefix100-PUREW-minus-baseline',ARM,'prefix100',baseline,'prefix100',task)
    insertion_rows=list(db.execute("SELECT * FROM consumer WHERE mode='insertion_R0' AND edit_order<=100"))
    insertion_lookup={(x['method'],x['edit_order'],x['query_id']):x['payload_key'] for x in insertion_rows}
    insertion=[]
    for method in METHODS:
        pairs=[(insertion_lookup[method,t['order'],t['edit_id']],lookup[method,'bank_R0',100,0,t['edit_id']]) for t in tasks]
        assert all(payloads[k]['status'] in ('FORMAT_VALID','MISSING') for pair in pairs for k in pair)
        insertion.append(dict(method=method,native_insertion_to_prefix100=transitions(pairs,scores),interpretation='Actual T0 insertion/prefix100 endpoints; no claim of uninterrupted intermediate correctness or T2G insertion retention'))
    cost=c.read(R/'RESOURCE_LEDGER.json');keys=[k for batch in cost['Astra_purew146_batches'] for k in batch['keys']]
    assert len(keys)==len(set(keys))==cost['Judge_attempts'] and all(x.get('ended_epoch') for x in cost['gpu_sessions'])
    attempts=[c.read(p) for p in (R/'private/attempts').glob('*.json')]
    result=dict(status='PAUSED_PARTIAL_PREFIX100_TERMINAL_SCORING',checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),panels=panels,contrasts=contrasts,insertion=insertion,pause=pause,
                full_run_complete=False,DEV_final146_gate='NOT_EVALUATED_UNFINISHED',new_inference_parameters=0,base_model_parameters=c.read(R/'public/REAL_VLM_ADMISSION.json')['parameters'],
                actual_compute_counts={k:sum(x.get(k,0) for x in attempts) for k in ('committed_updates','backwards','forwards','generation_attempts')},GPU_process_hours=cost['gpu_seconds_used']/3600,
                Judge_attempts=cost['Judge_attempts'],failed_Judge_batches=sum(x['status']=='FAILED_NO_RETRY' for x in cost['Astra_purew146_batches']),global_scoring_statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
                owned_weight_peak_bytes=cost['weights_observed_peak_bytes'],active_checkpoint_bytes=(R/'private/active/bank_R0.pt').stat().st_size,active_checkpoint_retained=True,bootstrap_replicates=10000,bootstrap_seed=20260912,
                final_native_reload_and_timing='NOT_ATTEMPTED_PAUSED',historical_matched_timing='NA',
                limitations=['User-selected pause following observed prefix50 outcomes; partial development evidence','Single is not insertion-time performance','Full prefix146/final native reload/timing absent; unattempted is not permanent missing','T1L/T2L small-support historical diagnostics; patient independence unknown','Update-norm path length is not net displacement; no gradient-direction causal attribution'])
    c.write(R/'public/PAUSE100_RESULTS.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
