import json, os, subprocess, sys
from pathlib import Path
w=Path(os.environ['WORKTREE']); root=Path(os.environ['LOCAL_SCORE_ROOT'])
sys.path[:0]=[str(w/'experiments/med_mdu_boundary_average_v1')]
import score_driver as prior
rows,decisions,conflicts,roles=prior.baseline()
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],input="from pathlib import Path\nprint(Path('/data/bmw/Knowledge_editing/outputs/med-mdu-boundary-v1-20261105/run/MDU/SCORE_PACKET_PRIVATE.json').read_text())\n",text=True,capture_output=True,timeout=60,check=True)
packet=json.loads(p.stdout)
if packet['status']!='SCORE_PACKET_FROZEN' or len(packet['consumers'])!=784: raise ValueError('packet missing')
history={(x['arm'],x['consumer_id']):x for x in rows}; added=[]
for x in packet['consumers']:
 old=history['MED',x['consumer_id']]
 if any(x[k]!=old[k] for k in ('consumer_id','query_id','owner','role','route')): raise ValueError('identity mismatch')
 if x['arm']=='MED':
  if x['key']!=old['key']:raise ValueError('MED key mismatch')
 elif x['arm']=='CANDIDATE': added.append(dict(x,arm='BOUNDARY'))
 else: raise ValueError('unexpected arm')
if len(added)!=392:raise ValueError('coverage mismatch')
rows+=added
old=json.loads((w/'reports/med_mdu_boundary_average_v1_20261105/RESULTS.json').read_text())
if prior.aggregate(rows,decisions,roles)!=old['panels'] or prior.aggregate(rows,{k:v for k,v in decisions.items() if k not in conflicts},roles)!=old['conflicts_unresolved']:raise ValueError('prior result mismatch')
keys={x['key'] for x in added if x['key'] not in decisions}
if len(keys)!=8 or not keys <= set(packet['payloads']):raise ValueError('target mismatch')
payloads={k:packet['payloads'][k]['record'] for k in keys}
if any(k!=v['opaque_query_id'] for k,v in payloads.items()):raise ValueError('payload mismatch')
def write(name,value):
 p=root/name
 with p.open('x') as f: json.dump(value,f,ensure_ascii=False)
 p.chmod(0o600)
write('BASELINE.private.json',{'rows':rows,'decisions':decisions,'conflicts':list(conflicts),'roles':roles})
write('SUPPLEMENT_INPUT.private.json',{'status':'SUPPLEMENT_IDENTITIES_FROZEN','payloads':payloads,'categories':{'history_unverified':sorted(keys),'conflict_unknown':[],'known_unknown':[]}})
write('PREPARE_COMPLETE.json',{'status':'BASELINE_REPRODUCED','targets':8,'history_classification':'not fully verified; explicit user authorization covers all eight'})
