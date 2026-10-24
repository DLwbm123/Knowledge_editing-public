"""Compare scoring versions only after the new unified judgments are frozen."""
import collections,json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def bridge(new,old):
 assert (new/'VERDICTS_FROZEN.json').exists() and (old/'VERDICTS_FROZEN.json').exists()
 nv=read(new/'BLIND_VERDICTS.json');ov=read(old/'BLIND_VERDICTS.json')
 nm={(r['arm'],r['owner'],r['query']):r['blind_id'] for r in read(new/'BLIND_MAPPING.json')}
 om={(r['arm'],r['owner'],r['query']):r['blind_id'] for r in read(old/'BLIND_MAPPING.json')}
 np={r['id']:r for r in read(new/'BLIND_PACKET.json')};op={r['id']:r for r in read(old/'BLIND_PACKET.json')}
 result={}
 for new_arm,old_arm in [('BASE','BASE'),('TT_FIXED160','ALL_TOKENS')]:
  tr=collections.defaultdict(collections.Counter)
  for t in read(new/'DATA_ADMISSION.json')['tasks']:
   owner=None if new_arm=='BASE' else t['order']
   for r in t['eval_rows']:
    ni=nm[(new_arm,owner,r['query_id'])];oi=om[(old_arm,owner,r['query_id'])]
    assert all(np[ni][k]==op[oi][k] for k in ['question','reference','answer'])
    for role in r['roles']:tr[role][ov[oi]['label']+'->'+nv[ni]['label']]+=1
  result[new_arm]={role:dict(total=sum(v.values()),transitions=dict(v),old_correct=sum(c for k,c in v.items() if k.split('->')[0]=='CORRECT'),new_correct=sum(c for k,c in v.items() if k.split('->')[1]=='CORRECT')) for role,v in tr.items()}
 return dict(status='SAME_OUTPUTS_DIFFERENT_SCORING_VERSION',old_scores_unchanged=True,new_labels_frozen_before_old_label_comparison=True,results=result)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('new',type=Path);p.add_argument('old',type=Path);a=p.parse_args();r=bridge(a.new,a.old);(a.new/'SCORING_VERSION_BRIDGE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
