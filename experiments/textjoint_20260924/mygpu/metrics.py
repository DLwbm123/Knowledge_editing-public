"""Offline edit-macro metrics with fixed Base masks and complete paired comparisons."""
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean


def summarize(rows,scores):
    groups=defaultdict(list)
    for r in rows:groups[(r['arm'],r['mode'],r['prefix'],r['task'])].append(r)
    result=[]
    for (arm,mode,prefix,task),rs in sorted(groups.items()):
        eligible=[];unknown=0
        for r in rs:
            b=scores.get(r['base_judge_key']);v=scores.get(r['judge_key'])
            if b is None:unknown+=1
            elif b is task.endswith('L'):eligible.append((r,v))
        per_edit=defaultdict(list);sources=set()
        for r,v in eligible:per_edit[r['edit']].append(v);sources.add(r['source_group'])
        missing=sum(v is None for _,v in eligible);num=sum(v is True for _,v in eligible);den=len(eligible)
        complete=not unknown and not missing
        known_means=[mean(vs) for vs in per_edit.values() if all(v is not None for v in vs)]
        result.append(dict(arm=arm,mode=mode,prefix=prefix,task=task,primary='Retention' if task.endswith('L') else 'Fix',
            numerator=num,denominator=den if not unknown else None,known_eligible=den,missing=missing,unknown_Base=unknown,
            edit_macro=mean(known_means) if complete and known_means else None,micro=num/den if complete and den else None,
            edits=len(per_edit),source_groups=len(sources),expected=len(rs),scored=sum(scores.get(r['judge_key']) is not None for r in rs),
            accuracy=sum(scores.get(r['judge_key']) is True for r in rs)/len(rs) if all(scores.get(r['judge_key']) is not None for r in rs) else None,
            exact_Base_token_consistency=mean(r['exact_Base_token_consistency'] for r in rs),
            status='UNKNOWN_BASE_MASK' if unknown else 'MISSING' if missing else 'EMPTY_DENOMINATOR' if not den else 'COMPLETE'))
    return result


def paired(rows,scores,arm,mode='single',prefix=1):
    selected=[r for r in rows if r['mode']==mode and r['prefix']==prefix]
    def identity(r):return r['edit'],r['task'],r['query_id']
    a={identity(r):r for r in selected if r['arm']=='B0'};b={identity(r):r for r in selected if r['arm']==arm}
    if set(a)!=set(b):return dict(status='UNPAIRED_COHORT',arm=arm)
    per=defaultdict(lambda:defaultdict(list));changes=[];missing=0
    for key,r in a.items():
        t=b[key]
        if r['base_judge_key']!=t['base_judge_key']:raise ValueError('Base mask differs across arms')
        base=scores.get(r['base_judge_key']);x=scores.get(r['judge_key']);y=scores.get(t['judge_key'])
        if base is None or x is None or y is None:missing+=1;continue
        if base is not r['task'].endswith('L'):continue
        per[r['task']][r['edit']].append(int(y)-int(x))
        changes.append(dict(edit=r['edit'],task=r['task'],query_id=r['query_id'],before=x,after=y))
    stats={};rng=random.Random(20260924)
    for task,edits in per.items():
        vals=[mean(v) for v in edits.values()];boot=sorted(mean(rng.choices(vals,k=len(vals))) for _ in range(2000))
        stats[task]=dict(delta_edit_macro=mean(vals) if not missing else None,paired_edits=len(vals),
            confidence_interval_95=[boot[49],boot[1949]] if not missing else None,
            right_to_wrong=sum(c['task']==task and c['before'] and not c['after'] for c in changes),
            wrong_to_right=sum(c['task']==task and not c['before'] and c['after'] for c in changes))
    return dict(status='COMPLETE' if not missing else 'MISSING',arm=arm,missing=missing,metrics=stats,changes=changes)


def job_report(root):
    root=Path(root);scores={p.stem:json.loads(p.read_text())['is_correct'] for p in (root/'private/judge/scores').glob('*.json')}
    reports={}
    for job in sorted((root/'jobs').glob('*')):
        if not (job/'STATUS.json').exists():continue
        status=json.loads((job/'STATUS.json').read_text());rows=[]
        for p in job.glob('*/p*/CONSUMERS.json'):rows+=json.loads(p.read_text())
        if status['job'].get('baseline_job'):
            baseline=root/'jobs'/status['job']['baseline_job']
            previous=json.loads((baseline/'STATUS.json').read_text())['job']
            if previous['N']!=status['job']['N'] or previous['mode']!=status['job']['mode']:raise ValueError('Unmatched baseline reuse')
            for p in baseline.glob('B0/p*/CONSUMERS.json'):rows+=json.loads(p.read_text())
        summary=summarize(rows,scores)
        comparisons=[paired(rows,scores,a,mode=status['job']['mode'],prefix=status['job']['N'] if status['job']['mode']=='sequential' else 1)
            for a in status['job'].get('arms',[]) if a!='B0']
        reports[job.name]=dict(status=status,summary=summary,comparisons=comparisons,consumers=len(rows))
    return reports


def self_check():
    r=dict(arm='B0',mode='single',prefix=1,task='T2L',edit='e',query_id='q',source_group='s',base_judge_key='b',judge_key='x',exact_Base_token_consistency=False)
    assert summarize([r],{'b':True})[0]['edit_macro'] is None
    assert summarize([r],{'b':False,'x':True})[0]['denominator']==0
    assert summarize([r],{'b':True,'x':False})[0]['edit_macro']==0
    other=dict(r,arm='E',judge_key='y')
    p=paired([r,other],{'b':True,'x':False,'y':True},'E')
    assert p['metrics']['T2L']['delta_edit_macro']==1
    assert p['metrics']['T2L']['confidence_interval_95']==[1,1]

if __name__=='__main__':self_check();print('PASS: missing, fixed masks, paired edit macro')
