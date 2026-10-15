"""Locked paired edit-macro analysis with missing bounds and no score repair."""
import collections
import json
import os
from pathlib import Path
import re
import sqlite3
import numpy as np
import wording_pair as protocol

SEED = 20261015
REPLICATES = 10000
MARKERS = re.compile(r"(?:cannot|can't|unable to) (?:determine|identify|see|provide)|insufficient (?:information|detail)|not enough information|无法(?:判断|确定|识别|看)|信息不足", re.I)


def contrast(values, selected, rows):
    owners = sorted({rows[i]['owner'] for i in selected})
    means = np.array([np.mean([values[i] for i in selected if rows[i]['owner'] == o]) for o in owners])
    rng = np.random.default_rng(SEED)
    boot = means[rng.integers(0, len(means), size=(REPLICATES, len(means)))].mean(axis=1)
    return dict(edit_macro_pp=float(means.mean()*100), CI99_pp=(np.quantile(boot, [.005, .995])*100).tolist(), edits=len(owners), probes=len(selected))


def analyze(pairs, old, new):
    primary = list(range(protocol.N)); sensitivity = [i for i,p in enumerate(pairs) if p['index'] not in protocol.SUSPECT]
    methods = {}; differences = {}; complete = all(new[m][i]['correct'] in (0,1) for m in protocol.METHODS for i in primary)
    for m in protocol.METHODS:
        scores = [new[m][i]['correct'] for i in primary]
        delta = [None if s is None else s-old[m][i]['correct'] for i,s in enumerate(scores)]
        lo = [(0 if s is None else s)-old[m][i]['correct'] for i,s in enumerate(scores)]
        hi = [(1 if s is None else s)-old[m][i]['correct'] for i,s in enumerate(scores)]
        lower = contrast(lo, primary, pairs)['edit_macro_pp']; upper = contrast(hi, primary, pairs)['edit_macro_pp']
        transitions = collections.Counter(f"{old[m][i]['correct']}->{s if s is not None else 'MISSING'}" for i,s in enumerate(scores))
        route_changes = sum(new[m][i]['route'] != old[m][i]['route'] for i in primary)
        methods[m] = dict(valid=sum(s is not None for s in scores), missing=sum(s is None for s in scores),
            old_correct=sum(v['correct'] for v in old[m]), new_correct=sum(s for s in scores if s is not None),
            paired_transitions=dict(transitions), delta_missing_bounds_pp=[lower,upper],
            old_nonanswer_marker=sum(bool(MARKERS.search(v['answer'])) for v in old[m]),
            new_nonanswer_marker=sum(bool(MARKERS.search(v['answer'])) for v in new[m]),
            activation_or_expert_changes=route_changes,
            primary=contrast(delta, primary, pairs) if all(s is not None for s in scores) else None,
            sensitivity75=contrast(delta, sensitivity, pairs) if all(s is not None for s in scores) else None)
        if methods[m]['primary']:
            p=methods[m]['primary']; s=methods[m]['sensitivity75']
            methods[m]['material_wording_signal'] = p['edit_macro_pp'] >= 10 and p['CI99_pp'][0] > 0 and s['CI99_pp'][0] > 0
        else: methods[m]['material_wording_signal'] = None
        differences[m] = delta
    extras = {}
    if complete:
        for m in protocol.METHODS[1:]:
            values = [differences[m][i]-differences['BASE'][i] for i in primary]
            extras[m+'_minus_BASE_change'] = contrast(values,primary,pairs)
    return dict(status='COMPLETE_VALID' if complete else 'COMPLETE_WITH_MISSING', methods=methods,
        change_contrasts=extras, main_denominator=78, main_edit_denominator=20,
        sensitivity_denominator=75, family='3 within-method changes and 2 changes versus BASE',
        interval='Paired edit bootstrap 10000, seed20261015; 99% marginal intervals, Bonferroni nominal family95%; small DEV, not exact coverage',
        meaningful_gain_threshold_pp=10, verdict='EVALUABLE_DESCRIPTIVE_DIAGNOSTIC' if complete else 'INSUFFICIENT_MISSING_SCORES',
        semantic_review='Operator exposed to prior results; three reference-scope concerns retained',
        Base_selection='Original mask selected Base failures; do not generalize Base gain or difference-in-change to an unselected population',
        causal_claim=False, original_scores_modified=False, original_generation_repeated=False,
        new_training=0, new_M1_M3=False)


def main():
    run=Path(os.environ['RUN_ROOT']); read=lambda p:json.loads(p.read_text())
    pairs=read(run/'private/WORDING_PAIRS.json'); old=read(run/'private/ORIGINAL_SCORES.json')
    new={m:[] for m in protocol.METHODS}
    with sqlite3.connect('file:'+str(run/'private'/protocol.QUEUE/'queue.sqlite')+'?mode=ro',uri=True) as db:
        assert db.execute("SELECT count(*) FROM consumer").fetchone()[0]==protocol.CONSUMERS
        assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
        for m in protocol.METHODS:
            for pair in pairs:
                qid=f"wording-pair-v1-{pair['index']:03d}"
                rows=db.execute('SELECT p.correct,p.status,c.path,p.record FROM consumer c JOIN payload p ON p.key=c.payload_key WHERE c.method=? AND c.query_id=?',(m,qid)).fetchall()
                assert len(rows)==1
                score,status,path,record=rows[0]; output=read(Path(path)); rec=json.loads(record)
                route=output.get('route') or {}
                new[m].append(dict(correct=score if status=='FORMAT_VALID' else None,answer=rec['raw_base_answer'],
                    route=[route.get('activated'),output.get('effective_expert')]))
    result=analyze(pairs,old,new)
    result['new_Judge_attempts']=read(run/'RESOURCE_LEDGER.json')['Judge_attempts']-read(run/'private/INHERITED_COST.json')['Judge_attempts']
    result['generation']=protocol.CONSUMERS
    result['new_GPU_seconds']=read(run/'RESOURCE_LEDGER.json')['gpu_seconds_used']-read(run/'private/INHERITED_COST.json')['gpu_seconds_used']
    prior=read(run/'private/WORDING_ADMISSION1.json').get('prior_failed_cost',{})
    result['physical_generation_attempts_including_failure']=protocol.CONSUMERS+prior.get('generation_attempts',0)
    result['prior_failed_GPU_seconds']=prior.get('GPU_seconds',0)
    result['total_diagnostic_GPU_seconds']=result['new_GPU_seconds']+prior.get('GPU_seconds',0)
    out=run/'public';out.mkdir(exist_ok=True)
    (out/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 问法配对诊断结果','',result['status'],'','原78条与原评分保留；新问法自由生成，0训练。仅DEV事后提出的固定诊断，无独立确认。','',
           '|方法|原正确/78|新正确/有效|缺失|宏平均变化pp|99%配对CI|','|---|---:|---:|---:|---:|---|']
    for m,v in result['methods'].items():
        p=v['primary']; lines.append(f"|{m}|{v['old_correct']}|{v['new_correct']}/{v['valid']}|{v['missing']}|{p['edit_macro_pp'] if p else 'NA'}|{p['CI99_pp'] if p else v['delta_missing_bounds_pp']}|")
    lines+=['','五个预定对比采用99%边际bootstrap区间；材料性信号需主变化至少10pp且主/75条敏感性下界均大于0。不是临床或唯一因果结论。',
            '原mask选择Base错误题，Base变化与方法变化之差有选择效应。路由激活变化须另报，不能将其解释成纯专家生成改善。',
            '非回答词标记只是固定字符串描述，不是重新评分。原图/标签3项疑虑未获得独立临床确认。M1/M3不自动启动。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    (run/'private/WORDING_REPORT_COMPLETE1.json').write_text(json.dumps(dict(status=result['status'],consumers=protocol.CONSUMERS))+'\n')


if __name__=='__main__': main()
