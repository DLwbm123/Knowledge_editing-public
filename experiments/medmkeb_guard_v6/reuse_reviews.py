"""Reuse frozen exact question/reference/answer verdicts without reading arm mappings."""
import argparse,json,collections
from pathlib import Path

def reuse(packet,prior_packet,prior_verdicts):
    labels={r['id']:r for r in prior_verdicts['rows']};known=collections.defaultdict(list)
    assert len(labels)==len(prior_verdicts['rows']) and set(labels)=={r['id'] for r in prior_packet}
    identity=lambda r:(r['question'],r['reference'],r['answer'])
    for row in prior_packet:known[identity(row)].append(labels[row['id']])
    reused=[];pending=[]
    for row in packet:
        matches=known.get(identity(row),[])
        if matches and len({v['label'] for v in matches})==1:
            reused.append(dict(id=row['id'],label=matches[0]['label'],reason=matches[0].get('reason','Frozen exact-input/answer verdict'),provenance='Exact question/reference/answer match to prior frozen review'))
        else:pending.append(row)
    assert len(reused)+len(pending)==len(packet)
    return dict(rows=reused),pending

def selfcheck():
    prior=[dict(id='old',question='q',reference='r',answer='a')];verdicts=dict(rows=[dict(id='old',label='CORRECT')]);current=[dict(prior[0],id='new'),dict(prior[0],id='changed',answer='b')]
    reused,pending=reuse(current,prior,verdicts);assert reused['rows'][0]['id']=='new' and pending[0]['id']=='changed';return dict(status='PASS',exact_only=True,changed_answer_not_inherited=True,no_model_calls=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--selfcheck',action='store_true');a=p.parse_args()
    if a.selfcheck:print(json.dumps(selfcheck()))
    else:
        assert a.private and a.prior
        read=lambda root,name:json.loads((root/name).read_text());reused,pending=reuse(read(a.private,'BLIND_PACKET.json'),read(a.prior,'BLIND_PACKET.json'),read(a.prior,'BLIND_VERDICTS.json'))
        (a.private/'BLIND_VERDICTS_REUSED.json').write_text(json.dumps(reused,indent=2)+'\n');(a.private/'BLIND_PENDING.json').write_text(json.dumps(pending,indent=2)+'\n');print(json.dumps(dict(reused=len(reused['rows']),pending=len(pending))))
