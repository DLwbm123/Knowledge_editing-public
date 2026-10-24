"""Build complete method-hidden review material only after all registered consumers finish."""
import argparse,json,random
from pathlib import Path

def build(private):
    remote=private/'remote';admission=json.loads((private/'DATA_ADMISSION.json').read_text());assert admission['status']=='PASS'
    tasks=admission['tasks'];rows=json.loads((private/'EVAL_ROWS.json').read_text());bindings={r['query_id']:r for r in json.loads((private/'CANDIDATES_FROZEN.json').read_text())['rows']+rows}
    for t in tasks:bindings[t['rephrase']['query_id']]=t['rephrase']
    for i in (0,1):assert json.loads((remote/str(i)/'STUDY_COMPLETE.json').read_text())['status']=='COMPLETE'
    consumers=[]
    for i in (0,1):
        for f in sorted((remote/str(i)/'outputs/BASE').glob('*.json')):consumers.append(dict(arm='BASE',task=None,query_id=f.stem,file=str(f)))
        prefix='OFF' if i==0 else 'ON'
        for task in tasks:
            expected={r['query_id'] for r in rows}|{task['rephrase']['query_id']}
            for mode in ('NONE','CE','SEM','BASELINE'):
                arm=(('BALANCEDIT' if i==0 else 'BALANCEDIT_CE') if mode=='BASELINE' else prefix+'_'+mode);folder=remote/str(i)/task['key']/'outputs'/arm;files=sorted(folder.glob('*.json'));assert {f.stem for f in files}==expected
                for f in files:consumers.append(dict(arm=arm,task=task['key'],query_id=f.stem,file=str(f)))
    # All four immutable historical edit Base outputs are reviewable, without regeneration.
    previous=private.parent/'medmkeb_semantic6_v1_20261018'
    reuse=json.loads((private/'BASE_REUSE.json').read_text())
    oldroot=json.loads((previous/'STATE.json').read_text())['remote_root']
    for row in rows:
        if row['query_id'].startswith('C'):
            local=previous/'remote'/Path(reuse[row['query_id']]).relative_to(oldroot)
            consumers.append(dict(arm='BASE',task=None,query_id=row['query_id'],file=str(local)))
    distinct={};items=[];mapping=[]
    for c in consumers:
        x=json.loads(Path(c['file']).read_text());r=bindings[c['query_id']];identity=(c['query_id'],r['question'],r['reference'],x['raw_answer'])
        if identity not in distinct:
            distinct[identity]=len(items);items.append(dict(question=r['question'],reference=r['reference'],answer=x['raw_answer']))
        mapping.append(dict(c,unique_index=distinct[identity],EOS=x['EOS'],at_cap=x['at_cap']))
    order=list(range(len(items)));random.Random(20261018).shuffle(order);opaque={old:f'R{index+1:04d}' for index,old in enumerate(order)}
    packet=[dict(id=opaque[index],**items[index]) for index in order]
    for c in mapping:c['id']=opaque[c.pop('unique_index')]
    (private/'BLIND_PACKET.json').write_text(json.dumps(packet,indent=2)+'\n');(private/'BLIND_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    print(json.dumps(dict(consumers=len(consumers),unique_reviews=len(packet),all_methods_finished=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path);a=p.parse_args();build(a.private)
