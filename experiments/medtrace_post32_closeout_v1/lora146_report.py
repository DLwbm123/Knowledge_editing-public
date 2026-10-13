"""Paired full-scale structural comparison using the frozen task masks."""
import json
import os
from pathlib import Path
import sqlite3
import time

import astra_report as ar
import lora146_queue as queue

RUN, r, common = ar.RUN, ar.r, ar.common
TT, LORA = 'TT88_W0_RETRO146', queue.ARM


def connect(path):
    db = sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    return db


def main():
    root = queue.q.ROOT
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    parents = queue.parent_rows()
    scores = {k:x['correct'] for k,(x,_) in parents.items()}
    old = connect(RUN/'private/judge_astra_medium/queue.sqlite')
    new = connect(root/'queue.sqlite')
    payloads = list(new.execute('SELECT * FROM payload'))
    for row in payloads:
        assert row['status'] in ('FORMAT_VALID','MISSING')
        assert common.digest(json.loads(row['binding']))==row['key']
        if row['key'] in scores: assert scores[row['key']]==row['correct']
        scores[row['key']] = row['correct']
    previous = list(old.execute('SELECT * FROM consumer WHERE method=?',(TT,)))
    current = list(new.execute('SELECT * FROM consumer'))
    assert len(previous)==len(current)==5722
    projection = {c['consumer_id']:c for c in new.execute('SELECT * FROM projection')}
    tasks = common.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks']
    ledger = common.read(RUN/'private/EVAL_LEDGER.json')
    records = []
    for c in previous+current:
        q = ledger['queries'][c['query_id']]
        active = {(t['native']['image_sha256'],t['native']['question']) for t in tasks[:c['prefix']]} if c['mode']=='bank_R0' else set()
        is_active = (q['image_sha256'],q['question']) in active
        if c['method']==LORA: assert bool(projection[c['id']]['active_target'])==is_active
        records.append((dict(binding=dict(phase=dict(arm=c['method'],node=0,prefix=c['prefix']),
            input=q,mode=c['mode'],owner_order=c['edit_order']),active_target=is_active,effective_expert=None),c['payload_key']))
    prior = common.read(RUN/'public/ASTRA_MEDIUM_RECOVERED_RESULTS.json')['panels']['benchmark146']
    panels, coeffs, groups = [], {}, {}
    for mode,prefix in [('single_R0',1)]+[('bank_R0',n) for n in (1,50,100,146)]:
        ts = tasks[:prefix] if mode=='bank_R0' else tasks
        for arm in (TT,LORA):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                m,c,g = r.panel(records,scores,'RETRO146',arm,0,mode,task,ts,prefix)
                m.pop('route_ON')
                if arm==TT:
                    old_m = next(x for x in prior if (x['mode'],x['prefix'],x['task'])==(mode,prefix,task))
                    for k in ('known_correct','observations','missing','edit_units'):
                        assert m[k]==old_m[k],(mode,prefix,task,k)
                    assert m['macro']==old_m['macro'] or abs(m['macro']-old_m['macro'])<1e-9
                m.update(ar.bootstrap(c,scores,g))
                panels.append(m)
                coeffs[(arm,mode,prefix,task)] = c
                groups.update(g)
    contrasts = []
    for mode,prefix in [('single_R0',1)]+[('bank_R0',n) for n in (1,50,100,146)]:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            aa,bb = coeffs[(TT,mode,prefix,task)],coeffs[(LORA,mode,prefix,task)]
            assert set(aa)==set(bb)
            cc = {e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
            bound = r.score_bounds(r.combine([{k:v/len(cc) for k,v in c.items()} for c in cc.values()]),scores) if cc else [None,None]
            contrasts.append(dict(comparison='TT88 minus normalized LoRA r1',mode=mode,prefix=prefix,task=task,
                paired_edits=len(cc),delta_bounds_pp=[x*100 if x is not None else None for x in bound],**ar.bootstrap(cc,scores,groups)))
    resource = common.read(RUN/'RESOURCE_LEDGER.json')
    missing = sum(x['status']=='MISSING' for x in payloads)
    result = dict(status='COMPLETE_NO_MISSING' if not missing else 'TERMINAL_WITH_MISSING',panels=panels,contrasts=contrasts,
        primary_endpoint='T2G edit macro, TT88 minus LoRA, final bank146',new_LoRA_consumers=5722,
        LoRA_payloads=len(payloads),LoRA_missing_payloads=missing,
        inherited_Astra_payloads=new.execute('SELECT count(*) FROM inherited').fetchone()[0],
        TT_existing_25_panels_reproduced=True,TT_no_retraining_or_rejudging=True,
        TT_parameters=7168,LoRA_parameters=18432,parameter_matched=False,
        LoRA_normalization='Input RMS matched to TT',LoRA_rank=1,not_standard_unmodified_LoRA=True,
        independent_confirmation=False,medical_protection='NA_SCOPE_NOT_QUALIFIED',
        cohort='Original146 exposed retrospective development, one frozen order',
        Judge_attempts=resource['Judge_attempts'],Judge_limit_enabled=False,GPU_hours=common.used()/3600,
        bootstrap_draws=10000,bootstrap_seed=20260912,missing_denominators_preserved=True)
    common.write(RUN/'public/LORA146_RESULTS.json',result)
    def value(m):
        return f"{m['macro']:.3f}" if m['macro'] is not None else 'NA' if m['macro_bounds'][0] is None else f"[{m['macro_bounds'][0]:.3f}, {m['macro_bounds'][1]:.3f}]"
    text = '# TT88—归一化LoRA：完整146结构对照\n\n沿用原146、原顺序、Base资格掩码、冻结Base/层/R0与生成设置。LoRA独立初始化、rank1、输入RMS归一化、native140/A2_80/W0_320；无U续训。TT全部沿用既有输出与完整Astra评分。所有数值为编辑宏平均百分比；缺失区间保留完整分母，不是置信区间。\n\n'
    text += '|模式|前缀|结构|T0|T1G|T2G|T1L诊断|T2L诊断|\n|---|---:|---|---:|---:|---:|---:|---:|\n'
    for mode,prefix in [('single_R0',1)]+[('bank_R0',n) for n in (1,50,100,146)]:
        for arm,label in [(TT,'TT88'),(LORA,'归一化LoRA r1')]:
            ms = [next(x for x in panels if (x['arm'],x['mode'],x['prefix'],x['task'])==(arm,mode,prefix,t)) for t in ('T0','T1G','T2G','T1L','T2L')]
            text += '|'+ '|'.join([mode,str(prefix),label]+[value(m) for m in ms])+'|\n'
    endpoint = next(x for x in contrasts if (x['mode'],x['prefix'],x['task'])==('bank_R0',146,'T2G'))
    text += '\n预登记主比较TT−LoRA的T2G差值界：'+str(endpoint['delta_bounds_pp'])+'个百分点；95%编辑配对区间'+str(endpoint['edit_ci'])+'，来源配对区间'+str(endpoint['source_ci'])+'。完整25项对比与50面板见LORA146_RESULTS.json。\n\n'
    text += '参数为7168与18432，非等参数对照，也不是未经修改的标准LoRA。已触达开发队列、仅一个编辑顺序，患者独立性未确认；T1L仅1编辑/2探针，T2L仅31编辑/54探针，不能替代合格医学保护评估。结果无论正负均保留，不据此挑checkpoint、seed或声称独立SOTA确认。\n\n'
    text += f"LoRA评分载荷{len(payloads)}份，缺失{missing}份；累计Judge尝试{resource['Judge_attempts']}次，已取消次数上限，失败仍计入。\n"
    (RUN/'public/LORA146_REPORT_ZH.md').write_text(text)
    common.write(root/'REPORT_COMPLETE.json',dict(status='AGGREGATES_COMPLETE_PUBLICATION_PENDING',epoch=time.time(),missing=missing))
    common.write(RUN/'public/LORA146_PROGRESS.json',dict(status='RESULTS_COMPLETE_PUBLICATION_PENDING',training_complete=True,generation_complete=True,
        scoring_terminal=True,scoring_complete_without_missing=missing==0,publication_complete=False,independent_confirmation_complete=False))
    print(json.dumps(dict(status=result['status'],primary_endpoint=endpoint,Judge_attempts=resource['Judge_attempts'])))


if __name__=='__main__':
    main()
