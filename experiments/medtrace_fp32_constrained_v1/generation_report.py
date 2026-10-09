"""Report paired semantic transitions without hiding baseline precision damage."""
import os
import json
import time
import sqlite3
from pathlib import Path
import common as c
import probe_report as stats
RUN=Path(os.environ['RUN_ROOT']);ARMS=('RAW','BOUNDED','MATCHED_RAW')


def transition(base,now):
    if now is None:return 'missing'
    return ('retained' if now else 'new_damage') if base else ('recovered' if now else 'still_wrong')


def summary(rows):
    assert rows
    return dict(observations=len(rows),correct=sum(r['correct']==1 for r in rows),missing=sum(r['correct'] is None for r in rows),
        against_FP32={x:sum(transition(r['base32'],r['correct'])==x for r in rows) for x in ('retained','new_damage','recovered','still_wrong','missing')},
        against_original={x:sum(transition(r['base16'],r['correct'])==x for r in rows) for x in ('retained','new_damage','recovered','still_wrong','missing')})


def selfcheck():
    assert [transition(b,n) for b,n in [(1,1),(1,0),(0,1),(0,0),(1,None)]]==['retained','new_damage','recovered','still_wrong','missing']
    s=summary([dict(correct=0,base32=1,base16=1),dict(correct=1,base32=0,base16=0)])
    assert s['correct']==1 and s['against_FP32']['new_damage']==s['against_FP32']['recovered']==1


def main():
    selfcheck();root=RUN/'private/judge_generation_astra_medium';assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores=list(db.execute('SELECT c.path,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'))
    assert len(scores)==2464 and all(r['status'] in ('FORMAT_VALID','MISSING') for r in scores)
    qualification=c.read(Path(os.environ['FP32_BASELINE_PARENT'])/'private/FP32_SEMANTIC_QUALIFICATION.json')['rows']
    bases={r['row']['query_id']:r for r in qualification if r['role']=='HELDOUT'};assert len(bases)==96
    held=[];edit=[]
    for r in scores:
        d=c.read(r['path']);row=d['binding']['input'];correct=r['correct'] if r['status']=='FORMAT_VALID' else None
        v=dict(arm=d['arm'],order=d['expert_order'],qid=row['query_id'],correct=correct,raw_answer=d['R0']['raw_answer'])
        if d['role']=='HELDOUT':
            b=bases[row['query_id']];v.update(base32=b['correct'],base16=int(b['previously_correct']),group=row['source_group'],path=r['path']);held.append(v)
        else:edit.append(v)
    assert len(held)==2304 and len(edit)==160
    panels={}
    for name,rs in [('PRIMARY63',[r for r in held if r['base16']]),('ALL96',held)]:
        panels[name]={a:summary([r for r in rs if r['arm']==a]) for a in ARMS}
    paired=[]
    for order in sorted({r['order'] for r in held}):
        rs=[r for r in held if r['order']==order and r['base16']];maps={a:{r['qid']:r for r in rs if r['arm']==a} for a in ARMS}
        assert all(len(v)==63 for v in maps.values())
        paired.append(dict(order=order,panels={a:summary(list(maps[a].values())) for a in ARMS}))
    complete=all(r['correct'] is not None for r in held+edit)
    contrast=None;decision='INCOMPLETE_SEMANTIC_EVIDENCE'
    if complete:
        primary=[r for r in held if r['base16']];mapping={(r['order'],r['qid'],r['arm']):r['correct'] for r in primary}
        delta=lambda r:mapping[r['order'],r['qid'],'BOUNDED']-mapping[r['order'],r['qid'],'MATCHED_RAW']
        unique=[r for r in primary if r['arm']=='BOUNDED']
        contrast=dict(mean=stats.mean([delta(r) for r in unique]),
            expert_CI=stats.interval([stats.mean([delta(r) for r in unique if r['order']==o]) for o in sorted({r['order'] for r in unique})]),
            source_CI=stats.interval([stats.mean([delta(r) for r in unique if r['group']==g]) for g in sorted({r['group'] for r in unique})]))
        edit_correct={a:sum(r['correct']==1 for r in edit if r['arm']==a) for a in ('BASE',)+ARMS}
        same={a:all(r['correct']==next(z['correct'] for z in held if z['order']==r['order'] and z['qid']==r['qid'] and z['arm']=='MATCHED_RAW') for r in held if r['arm']==a) for a in ('RAW','BOUNDED')}
        decision='NO_SEMANTIC_SEPARATION' if all(same.values()) else 'NO_ESTABLISHED_SEMANTIC_PROTECTION_GAIN'
        if contrast['expert_CI'][0]>0 and contrast['source_CI'][0]>0 and edit_correct['BOUNDED']>=edit_correct['RAW']:
            decision='DEVELOPMENT_SEMANTIC_SIGNAL_ONLY'
    else:edit_correct={a:sum(r['correct']==1 for r in edit if r['arm']==a) for a in ('BASE',)+ARMS}
    c.write(RUN/'private/GEN_SEMANTIC_TRANSITIONS.json',dict(heldout=held,edit=edit))
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');before=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==6 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    result=dict(status='COMPLETE',decision=decision,panels=panels,per_expert=paired,primary_contrast=contrast,
        edit_correct=edit_correct,edit_missing={a:sum(r['correct'] is None for r in edit if r['arm']==a) for a in ('BASE',)+ARMS},
        edit_denominator=40,edit_correct_counts_accepted_verdicts_only=True,payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        generation=c.read(RUN/'private/GENERATION_COMPLETE.json'),queue=c.read(root/'READY.json'),
        baseline=dict(original_correct=63,FP32_correct=62,primary_denominator=63),
        independent_confirmation=False,clinical_protection=False,full_training_evaluated=False,
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,
            cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']))
    c.write(RUN/'public/GEN_RESULTS.json',result)
    (RUN/'public/GEN_REPORT_ZH.md').write_text('# 冻结候选自由生成验证\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n固定原63主面板；语义相同不证明完整训练或临床保护。零差异可能来自单步更新太小，不反推局部KL机制无效。\n')
    c.write(RUN/'private/GEN_REPORT_COMPLETE.json',dict(status='COMPLETE',decision=decision,epoch=time.time()))


if __name__=='__main__':main()
