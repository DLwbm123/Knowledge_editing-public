"""Audit semantic support and token-boundary aggregates without releasing answers."""
import os
import json
import sqlite3
from pathlib import Path
from collections import Counter

RUN=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def supported(panels):
    return all(panels[role]['all']['correct_sources']>=8 and
        all(panels[role][kind]['correct']>0 for kind in ('yes','no','open')) for role in ('BASIS','HELDOUT'))


def selfcheck():
    panels={r:{k:dict(correct=1,correct_sources=8) for k in ('all','yes','no','open')} for r in ('BASIS','HELDOUT')}
    assert supported(panels)
    panels['HELDOUT']['all']['correct_sources']=7;assert not supported(panels)
    panels['HELDOUT']['all']['correct_sources']=8;panels['BASIS']['open']['correct']=0;assert not supported(panels)


def main():
    selfcheck()
    root=RUN/'private/judge_answer_astra_medium'
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=[]
    for rec in db.execute('SELECT c.*,p.status,p.correct,p.batch FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        d=read(Path(rec['path']));row=d['binding']['input']
        assert rec['status'] in ('FORMAT_VALID','MISSING')
        rows.append(dict(row=row,audit=d['audit'],correct=rec['correct'] if rec['status']=='FORMAT_VALID' else None,
                         status=rec['status'],path=rec['path'],payload=rec['payload_key']))
    assert len(rows)==192
    panels={}
    for role in ('BASIS','HELDOUT'):
        panels[role]={}
        for kind in ('all','yes','no','open'):
            rs=[r for r in rows if r['row']['audit_role']==role and (kind=='all' or r['row']['answer_kind']==kind)]
            correct=sum(r['correct']==1 for r in rs);missing=sum(r['correct'] is None for r in rs)
            panels[role][kind]=dict(queries=len(rs),correct=correct,wrong=len(rs)-correct-missing,missing=missing,
                accuracy_bounds=[correct/len(rs),(correct+missing)/len(rs)],
                correct_sources=len({r['row']['source_group'] for r in rs if r['correct']==1}),
                semantic_correct_but_content_not_exact=sum(r['correct']==1 and not r['audit']['content_all_correct'] for r in rs),
                strict_content_correct=sum(r['audit']['content_all_correct'] for r in rs))
    audits=[r['audit'] for r in rows]
    flags=('prefix_exact','strict_all_correct','content_all_correct','EOS_correct','first_target_correct',
        'first_teacher_equals_first_generated','decoded_target_matches_annotation',
        'generated_roundtrip_prefix_exact','generated_roundtrip_tokens_exact','at_generation_cap')
    boundary={k:sum(a[k] for a in audits) for k in flags}
    boundary['target_leading_empty_token_histogram']=dict(Counter(a['target_leading_empty_tokens'] for a in audits))
    decision='SEMANTIC_SUPPORT_AVAILABLE' if supported(panels) else 'DATA_SUPPORT_NOT_ESTABLISHED'
    ledger=read(RUN/'RESOURCE_LEDGER.json');before=read(RUN/'private/INHERITED_COST.json')
    assert all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,
        cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],
        cumulative_Judge=ledger['Judge_attempts'],new_generations=192,annotation_forwards=192,updates=0,new_checkpoints=0)
    assert resource['new_Judge']==192
    write(RUN/'private/SEMANTIC_QUALIFICATION.json',dict(rows=rows,independent_confirmation=False))
    result=dict(status='COMPLETE',decision=decision,panels=panels,boundary=boundary,resource=resource,
        payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        annotation_source_only=True,independent_confirmation=False,new_method_training_registered=False)
    write(RUN/'public/RESULTS.json',result)
    (RUN/'public/REPORT_ZH.md').write_text(f'''# Base源回答资格核验\n\n结论：{decision}。BASIS正确{panels['BASIS']['all']['correct']}/96，缺失{panels['BASIS']['all']['missing']}；HELDOUT正确{panels['HELDOUT']['all']['correct']}/96，缺失{panels['HELDOUT']['all']['missing']}。具体三题型、来源支持和缺失界见RESULTS。\n\n严格内容全token正确{boundary['content_all_correct']}/192，EOS正确{boundary['EOS_correct']}/192。原标注与生成prompt前缀一致{boundary['prefix_exact']}/192；首teacher预测与实际首生成token一致{boundary['first_teacher_equals_first_generated']}/192；生成文本往返token相同{boundary['generated_roundtrip_tokens_exact']}/192。上述诊断不能单独证明医学正确或实现错误。\n\n全部192Base生成和192标注前向完成，0更新/权重，原隔离Astra medium每payload一次，不隐式补判缺失。新增GPU进程小时{resource['new_GPU_process_hours']:.6f}，累计{resource['cumulative_GPU_process_hours']:.6f}；新增Judge{resource['new_Judge']}，累计{resource['cumulative_Judge']}。原图、文本、tokens和逐题关联仅私有保存。尚未登记新的投影训练。\n''')
    write(RUN/'private/REPORT_COMPLETE.json',dict(status='COMPLETE',decision=decision))


if __name__=='__main__':main()
