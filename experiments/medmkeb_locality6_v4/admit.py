"""Stage the fixed shared consumers only after complete Base/source/implementation qualification."""
import json,time,collections
from pathlib import Path

def stage(private,remote):
    read=lambda p:json.loads(Path(p).read_text())
    write=lambda n,v:(private/n).write_text(json.dumps(v,indent=2)+'\n')
    m=read(private/'CANDIDATES_FROZEN.json');review=read(private/'BASE_VERDICTS.json')['rows'];labels={v['query_id']:v['label'] for v in review}
    assert len(labels)==len(review)==len(m['rows']) and set(labels)=={v['query_id'] for v in m['rows']}
    assert set(labels.values())<={'CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY'}
    assert read(private/'remote/IMPLEMENTATION_ADMISSION.json')['status']=='PASS'
    assert read(private/'SOURCE_ADMISSION.json')['status']=='PASS' and not m['visual_review_pending']
    for i in (0,1):assert read(private/'remote'/str(i)/'COMPLETE.json')['status']=='BASE_QUALIFICATION_COMPLETE'
    train=[c for c in m['candidates'] if c['split']=='train'];evaluation=[c for c in m['candidates'] if c['split']=='eval'];correct_train=[c for c in train if labels[c['row']['query_id']]=='CORRECT'];correct_eval=[c for c in evaluation if labels[c['row']['query_id']]=='CORRECT']
    groups={c['article'] for c in correct_eval};counts={s:dict(collections.Counter(v['label'] for v in review if v['split']==s)) for s in ('train','eval')}
    result=dict(status='DATA_GATE_NOT_MET',base_review=counts,train_correct=len(correct_train),eval_correct=len(groups),required_eval_correct=4,fisher_source_count=len(train),time=time.time(),six_arm_started=False,balancedit_started=False)
    if len(train)<8 or not correct_train or len(groups)<4:
        result['missing']=[n for yes,n in [(len(train)>=8,'Fewer than8 source-qualified Fisher rows'),(bool(correct_train),'No clearly correct legal training construct'),(len(groups)>=4,'Fewer than4 independent correct eval sources')] if not yes];write('DATA_ADMISSION.json',result);return result
    previous=private.parent/'medmkeb_semantic6_v1_20261018';old=read(previous/'CANDIDATES_FROZEN.json');oldlabels={v['query_id']:v['label'] for v in read(previous/'BASE_VERDICTS.json')['rows']}
    edits=[c for c in old['candidates'] if all(oldlabels[v['query_id']]=='INCORRECT' for v in c['rows'])][:2];assert [c['index'] for c in edits]==[2,3]
    oldroot=read(previous/'STATE.json')['remote_root'];retention=correct_train[0];tasks=[];reuse={r['query_id']:remote+'/'+str(i%2)+'/outputs/BASE/'+r['query_id']+'.json' for i,r in enumerate(m['rows'])};eval_rows=[c['row'] for c in evaluation]
    for i,c in enumerate(edits):
        key='E'+str(i+1).zfill(2);source=c['rows'][0];rephrase=dict(source,query_id=key+'_rephrase',question=c['original']['rephrase'],audit_role='TEXT_GENERALIZATION')
        tasks.append(dict(key=key,source=source,rephrase=rephrase,retention=retention['row'],retention_correct=retention['row']['reference'],retention_wrong=retention['wrong_choice'],source_component=c['component']))
        for row in c['rows']:
            reuse[row['query_id']]=oldroot+'/'+str((c['index']-1)%2)+'/outputs/BASE/'+row['query_id']+'.json';eval_rows.append(row)
    assert len(eval_rows)==len(evaluation)+4 and len({r['query_id'] for r in eval_rows})==len(eval_rows)
    assert not {v['article'] for v in train}&{v['article'] for v in evaluation}
    result.update(status='PASS',tasks=tasks,fisher_rows=[c['row'] for c in train[:8]],retention_query=retention['row']['query_id'],fixed_eval_rows=len(eval_rows),heldout_used_for_training=False,independent_clinical_confirmation=False)
    write('BASE_REUSE.json',reuse);write('EVAL_ROWS.json',eval_rows);write('DATA_ADMISSION.json',result);return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path);p.add_argument('--remote',required=True);a=p.parse_args();d=stage(a.private,a.remote);print(json.dumps({k:d[k] for k in ('status','train_correct','eval_correct','fisher_source_count')}))
