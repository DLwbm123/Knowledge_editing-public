"""All64 method-hidden answers, exact historical semantic verdict reuse only."""
import json,random
from pathlib import Path

def build(private):
    read=lambda p:json.loads(p.read_text());previous=private.parent/'medmkeb_guard_v6_20261019';oldpacket={x['id']:x for x in read(previous/'BLIND_PACKET.json')};oldlabels={x['id']:x['label'] for x in read(previous/'BLIND_VERDICTS.json')['rows']};old={};base={}
    for m in read(previous/'BLIND_MAPPING.json'):
        x=oldpacket[m['id']];identity=(m['query_id'],x['question'],x['reference'],x['answer']);label=oldlabels[m['id']];assert identity not in old or old[identity]==label;old[identity]=label
        if m['arm']=='BASE':assert m['query_id'] not in base or base[m['query_id']]==label;base[m['query_id']]=label
    admission=read(private/'DATA_ADMISSION.json');rows=read(private/'EVAL_ROWS.json');bindings={x['query_id']:x for x in rows}
    for t in admission['tasks']:bindings[t['rephrase']['query_id']]=t['rephrase']
    items={};mapping=[]
    for i,t in enumerate(admission['tasks']):
        root=private/'remote'/str(i);assert read(root/'EXIT.json')['code']==0 and read(root/'STUDY_COMPLETE.json')['answers']==32
        expected={x['query_id'] for x in rows}|{t['rephrase']['query_id']};files={x.stem:x for x in (root/'outputs').glob('*.json')};assert set(files)==expected
        for q,p in sorted(files.items()):
            v=read(p);r=bindings[q];identity=(q,r['question'],r['reference'],v['raw_answer']);items.setdefault(identity,dict(question=r['question'],reference=r['reference'],answer=v['raw_answer']));mapping.append(dict(task=t['key'],query_id=q,identity=identity,EOS=v['EOS'],at_cap=v['at_cap']))
    order=list(items);random.Random(20261019).shuffle(order);ids={k:'R'+str(i+1).zfill(4) for i,k in enumerate(order)};packet=[dict(id=ids[k],**items[k]) for k in order];reused=[dict(id=ids[k],label=old[k],source='EXACT_FROZEN_IDENTITY') for k in order if k in old]
    for m in mapping:m['id']=ids[m.pop('identity')]
    assert len(mapping)==64
    for name,value in [('BLIND_PACKET.json',packet),('BLIND_MAPPING.json',mapping),('BASE_LABELS_FROZEN.json',base),('BLIND_VERDICTS_REUSED.json',dict(rows=reused)),('BLIND_PENDING.json',[x for x in packet if x['id'] not in {v['id'] for v in reused}])]:
        (private/name).write_text(json.dumps(value,indent=2)+'\n')
    return dict(consumers=64,unique_reviews=len(packet),reused=len(reused),pending=len(packet)-len(reused))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);print(build(p.parse_args().private))
