"""Read-only single-BE milestone aggregation; no inference or Judge calls."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics

from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest, lines
from scripts.medtrace.stage17_freeze import accepted
from scripts.medtrace.stage17_student_packets import mode_sources
from scripts.medtrace.stage17_judge import execution_path


def metric(rows, field, selector=lambda r: True):
    selected = [r for r in rows if selector(r)]
    edits = defaultdict(list)
    for r in selected:
        edits[r['edit']].append(int(r[field]))
    values = {e: statistics.mean(v) for e,v in edits.items()}
    if not values:
        return dict(numerator=0,probes=0,edits=0,micro=None,macro=None), values
    return dict(numerator=sum(r[field] for r in selected),probes=len(selected),edits=len(edits),
        micro=sum(r[field] for r in selected)/len(selected),macro=statistics.mean(values.values())), values


def interval(values, groups):
    if not values: return dict(edit_ci95=None,source_cluster_ci95=None,source_groups=0)
    clusters = defaultdict(list)
    for e,v in values.items(): clusters[groups[e]].append(v)
    def sample(units):
        rng = random.Random(20260912); draws=[]
        for _ in range(10000):
            picked = rng.choices(units,k=len(units))
            draws.append(sum(x[0] for x in picked)/sum(x[1] for x in picked))
        draws.sort()
        return [draws[249],draws[9749]]
    return dict(edit_ci95=sample([(v,1) for v in values.values()]),
        source_cluster_ci95=sample([(sum(v),len(v)) for v in clusters.values()]),source_groups=len(clusters))


def report(bundle, destination):
    op = bundle/'operator'; private = bundle/'source/private'
    manifest = read(op/'MANIFEST.json'); ledger=read(private/'COHORT_AND_SUPPORT_LEDGER.json')
    if manifest['freeze_id'] != ledger['freeze_id'] or digest({k:v for k,v in ledger.items() if k!='freeze_id'}) != ledger['freeze_id']:
        raise ValueError('Frozen cohort changed')
    status=read(execution_path(op))
    if status['status'] != 'COMPLETE_FORMAT_AND_COVERAGE_VALIDATED': raise ValueError('Student Judge incomplete')
    bindings=read(op/'BINDINGS.json'); verdicts=lines(op/'VERDICTS_ASTRA.jsonl'); lock=read(op/'JUDGE_LOCK.json')
    scores={v['opaque_query_id']:v for v in verdicts}
    if len(scores)!=len(verdicts) or set(scores)!=set(bindings): raise ValueError('Verdict coverage mismatch')
    for oid,b in bindings.items():
        v=scores[oid]
        if (digest(b)!=oid or b['judge']!=lock or type(v['is_correct']) is not bool or
            v['query_id']!=b['query_id'] or v['protocol']!=lock['protocol'] or
            v['judge_model']!=lock['model'] or v['immutable_snapshot']!=lock['immutable_snapshot']):
            raise ValueError('Student Judge full binding mismatch')
    base=Path(manifest['unchanged_Base_bundle'])/'operator'
    bb=read(base/'BINDINGS.json'); bv=lines(base/'VERDICTS_ASTRA.jsonl')
    if read(base/'JUDGE_LOCK.json')!=lock or accepted(bb,bv)!=ledger['Base_correctness']:
        raise ValueError('Base correctness/config changed')
    c0=ledger['Base_correctness']; tasks={t['edit_id']:t for t in ledger['tasks'] if t['edit_id'] in ledger['main_T0']}
    groups={e:t['native']['source_group'] for e,t in tasks.items()}
    mappings=read(op/'MODE_MAPPING.json'); outputs={}; receipts=[]; expected=set()
    for e,t in tasks.items():
        directory=private/'single_BE'/f"e{t['order']:03d}"
        receipt=read(directory/'COMPLETE.json'); receipts.append(receipt)
        if (directory/'FAILURE.json').exists() or receipt['status']!='GENERATED_NOT_SCORED': raise ValueError('Invalid edit receipt')
        queries=list(dict.fromkeys([e]+[q for event in t['events'] for q in event['all_probe_query_ids']]))
        if receipt['queries']!=len(queries): raise ValueError('Query coverage mismatch')
        for i,q in enumerate(queries):
            expected.add((e,q)); outputs[e,q]=read(directory/f'query_{i:03d}.json')
    if len(mappings)!=len(expected) or {(m['edit_id'],m['query_id']) for m in mappings}!=expected:
        raise ValueError('Mode mapping coverage mismatch')
    decisions={}; agreements={}; routes={}
    norm=lambda s:' '.join(s.lower().split())
    for m in mappings:
        e,q=m['edit_id'],m['query_id']; o=outputs[e,q]; b=bb[o['Base_cache_id']]
        rawbase=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids'])
        sources=mode_sources(o,rawbase)
        if digest(o)!=m['output_binding'] or b['query_id']!=q: raise ValueError('Output binding changed')
        for mode,ref in m['modes'].items():
            if ref['source']!=sources[mode]: raise ValueError('Route source mismatch')
            oid=ref['opaque_query_id']
            if ref['source']=='Base':
                if oid!=o['Base_cache_id']: raise ValueError('OFF Base identity mismatch')
                correct=c0[q]
            else:
                bnd=bindings[oid]
                if bnd['realized_binding']!=o['binding']: raise ValueError('Student realized binding mismatch')
                correct=scores[oid]['is_correct']
            decisions[e,q,mode]=correct
            agreements[e,q,mode]=norm(o[mode]['raw_answer'])==norm(rawbase['raw_answer'])
            routes[e,q,mode]=sources[mode]=='student'
    allrows=[]
    for e,t in tasks.items():
        for event in t['events']:
            for q in event['all_probe_query_ids']:
                for mode in ('R0','RC','FORCED_ON'):
                    allrows.append(dict(edit=e,task=event['task'],mode=mode,base=c0[q],
                        correct=decisions[e,q,mode],agreement=agreements[e,q,mode],on=routes[e,q,mode]))
    panels=[]; paired=[]
    for task in sorted({r['task'] for r in allrows}):
        main_selector=(lambda r:r['base']) if task.endswith('L') else (lambda r:not r['base'])
        name='Retention' if task.endswith('L') else 'Fix'
        macro_by_mode={}
        for mode in ('R0','RC','FORCED_ON'):
            rows=[r for r in allrows if r['task']==task and r['mode']==mode]
            primary,values=metric(rows,'correct',main_selector); primary.update(interval(values,groups)); macro_by_mode[mode]=values
            if task.endswith('L'): primary['c2w_micro']=None if primary['micro'] is None else 1-primary['micro']
            panels.append(dict(task=task,mode=mode,primary_metric=name,primary=primary,
                post_accuracy=metric(rows,'correct')[0],base_accuracy=metric(rows,'base')[0],
                output_agreement=metric(rows,'agreement')[0],route_on=metric(rows,'on')[0]))
        for mode in ('RC','FORCED_ON'):
            delta={e:macro_by_mode[mode][e]-v for e,v in macro_by_mode['R0'].items()}
            paired.append(dict(task=task,contrast=mode+' minus R0',edits=len(delta),
                improved=sum(v>0 for v in delta.values()),worsened=sum(v<0 for v in delta.values()),
                tied=sum(v==0 for v in delta.values()),macro_delta=statistics.mean(delta.values()) if delta else None,
                **interval(delta,groups)))
    result=dict(status='REPORTED_SINGLE_BE_MILESTONE_STAGE17_INCOMPLETE',N=len(tasks),
        student_judgments=len(scores),judge_model=lock['model'],reasoning_effort=lock['reasoning_effort'],
        immutable_snapshot=lock['immutable_snapshot'],config_id=lock['config_sha256'],freeze_id=ledger['freeze_id'],
        scope='single BE on main T0 cohort and its attached supported tasks only',
        cohort=ledger['summary'],panels=panels,paired_route_contrasts=paired,
        statistics=dict(bootstrap=10000,seed=20260912,confidence=.95,unit='edit',
            sensitivity='native source-group cluster resampling, edit-weighted mean; not full connected-component clustering'),
        cost=dict(edits_completed=len(receipts),edit_failures=0,steps=sum(r['training']['steps'] for r in receipts),
            summed_edit_seconds=sum(r['seconds'] for r in receipts),
            timing_scope='sum of per-edit training/checkpoint and generation sections; excludes model load, between-edit cleanup and Judge time',
            max_allocated_bytes=max(r['peak_allocated_bytes'] for r in receipts),
            max_reserved_bytes=max(r['peak_reserved_bytes'] for r in receipts),
            checkpoint_bytes=sum(r['checkpoint_bytes'] for r in receipts)),
        incomplete=['C_NO_H','LoRA-Perf-v1','GRACE','BELoRA','task-specific queues outside main T0',
            'sequential','Qwen sidecar final comparison'],
        unsupported_main_T0=['C_FACT: H support=0','C_EXTRA_QA: H/G support=0','PairCorrect: H-eval support=0','T5: not authorized'],
        no_training_generation_or_judge_calls=True)
    destination.mkdir()
    write_new(destination/'SINGLE_BE_AGGREGATES.json',result)
    with (destination/'SINGLE_RESULTS.csv').open('x',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['task','mode','metric','edits','numerator','probes','edit_macro','probe_micro','ci_low','ci_high','post_acc_micro','route_on_micro'])
        for p in panels:
            m=p['primary']; ci=m['edit_ci95'] or [None,None]
            writer.writerow([p['task'],p['mode'],p['primary_metric'],m['edits'],m['numerator'],m['probes'],m['macro'],m['micro'],*ci,p['post_accuracy']['micro'],p['route_on']['micro']])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('bundle',type=Path); parser.add_argument('destination',type=Path)
    args=parser.parse_args(); report(args.bundle,args.destination)
