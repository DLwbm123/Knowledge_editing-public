"""Reuse verified judgments after the complete 150-expert bank freezes."""
import json,os,sys,time,subprocess,traceback
from pathlib import Path
root=Path(os.environ['LOCAL_SCORE_ROOT']);w=Path(os.environ['WORKTREE']);remote=Path(os.environ['REMOTE_ROOT']);deadline=float(os.environ['EXPERIMENT_DEADLINE'])
sys.path[:0]=[str(w/'experiments'/x) for x in ('med_mdu150_v1','med_medinit_101_v1','med_medinit_loss_v1','med_full_d150_v1','med_proposition_generalization_v1','med_fixed_depth_v1','med_source_teacher_v1','med150_source_review_v1')]
import summary,score_reuse,ledger
FIELDS=('core_correct','full_supported','unsupported_expansion','extra_contradiction')
def read(p):return json.loads(Path(p).read_text())
def write(n,obj):
 p=root/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False));t.chmod(0o600);t.replace(p)
def ssh(script,timeout=60):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],input=script,text=True,capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError('remote dependency failed: '+p.stderr[-400:])
 if len(p.stdout.encode())>45000000:raise ValueError('packet limit')
 return json.loads(p.stdout)
def main():
 baseline=read(os.environ['BASELINE']);routing=baseline['routing'];roles=read(os.environ['ROLES']);prior=baseline['consumers'];base=dict(baseline['decisions'])
 if len(prior)!=6444 or routing.get('status')!='ROUTING_FROZEN':raise ValueError('baseline incomplete')
 for path in (os.environ['SCORE_97'],os.environ['SCORE_8']):
  part=read(path)
  if part['unknown']:raise ValueError('prior supplement incomplete')
  for k,v in part['decisions'].items():
   if k in base and any(base[k][f]!=v[f] for f in FIELDS):raise ValueError('prior judgment conflict')
   base[k]=v
 prior_map={(x['arm'],x['query_id']):x for x in prior}
 if len(prior_map)!=6444:raise ValueError('prior identity collision')
 selected=[x['consumer_id'] for x in routing['records'] if x['group']=='med49']
 if len(selected)!=635 or len(set(selected))!=635:raise ValueError('new route count')
 probe="""import json\nfrom pathlib import Path\nr=Path(%r);f=r/'EVAL_GUARD_EXIT.json';bad=any((r/n).exists() for n in ('TRAIN_GUARD_FAILURE.json','TRAIN_STOP.json','EVAL_GUARD_FAILURE.json','EVAL_STOP.json'))\nprint(json.dumps({'bad':bad,'done':f.exists(),'code':json.loads(f.read_text())['code'] if f.exists() else None}))\n"""
 while True:
  if time.time()>=deadline-180:raise TimeoutError('original deadline')
  state=ssh(probe % str(remote))
  if state['bad'] or (state['done'] and state['code']!=0):raise RuntimeError('training or generation dependency failed')
  if state['done']:break
  write('DEPENDENCY_WAIT.json',{'time':time.time()});time.sleep(20)
 packet_script="""import json,sys\nfrom pathlib import Path\nr=Path(%r); old=Path(%r);sys.path[:0]=[str(r/'tools'),str(Path(%r)/'tools'),str(Path(%r)/'tools')]\nimport packet\npacket.build(r,old,%r)\nprint((r/'SCORE_PACKET_PRIVATE.json').read_text())\n""" % (str(remote),os.environ['ADDITIONS_ROOT'],os.environ['FULL_D_ROOT'],os.environ['FIXED_ROOT'],selected)
 packet=ssh(packet_script,timeout=300);write('PACKET.private.json',packet)
 history=read(os.environ['HISTORY_INDEX'])
 if history.get('status')!='AVAILABLE_HISTORY_INDEX_FROZEN':raise ValueError('history missing')
 for k,v in base.items():history['semantic_valid'].setdefault(k,[]).append(v)
 legacy={k for v in packet['payloads'].values() for k in v['legacy_keys']}
 for raw in os.environ.get('LEGACY_DECISION_PATHS','').split(os.pathsep):
  p=Path(raw)
  if raw and p.is_file():
   for k,v in ledger.selected_decisions(p,legacy).items():
    if k in history['valid'] and any(history['valid'][k][f]!=v[f] for f in FIELDS):history['conflicts'].append(k)
    else:history['valid'][k]=v
 part=score_reuse.resolve(packet,history); blocked=set(part['conflict_unknown']); decisions=dict(base)
 for k,v in part['known_valid'].items():
  if k in decisions and any(decisions[k][f]!=v[f] for f in FIELDS):blocked.add(k)
  else:decisions[k]=v
 for k in blocked:decisions.pop(k,None)
 consumers=[]
 for x in prior:
  if x['arm']=='MED':consumers.append(dict(x))
  elif x['arm']=='MDU':
   consumers.append(dict(x,arm='MDU101'))
   if x['consumer_id'] not in selected:consumers.append(dict(x,arm='MDU150'))
 for x in packet['consumers']:
  ref=prior_map[('MED',x['query_id'])]
  if any(x[k]!=ref[k] for k in ('consumer_id','query_id','owner','role','route')):raise ValueError('new consumer identity differs')
  if x['arm']=='MED' and x['key']!=ref['key']:raise ValueError('MED payload differs')
  if x['arm']=='CANDIDATE':consumers.append(dict(x,arm='MDU150'))
 be=read(os.environ['BE_PACKET']);bed=read(os.environ['BE_DECISIONS'])
 if bed['unknown']:raise ValueError('BE decisions incomplete')
 be_rows=[x for x in be['consumers'] if x['mode']=='bank' and x['prefix']==150]
 if len(be_rows)!=2148:raise ValueError('BE bank count')
 for k,v in bed['decisions'].items():
  if k in decisions and any(decisions[k][f]!=v[f] for f in FIELDS):blocked.add(k)
  else:decisions[k]=v
 for k in blocked:decisions.pop(k,None)
 for x in be_rows:
  ref=prior_map[('MED',x['query_id'])]
  if x['owner']!=ref['owner']:raise ValueError('BE owner mismatch')
  consumers.append({'arm':'BE','query_id':x['query_id'],'key':x['key']})
 result=summary.compare(consumers,decisions,roles,routing)
 conflicts=set()
 for path in (os.environ['INPUT_97'],os.environ['INPUT_8']):conflicts.update(read(path)['categories']['conflict_unknown'])
 sensitive=summary.compare(consumers,{k:v for k,v in decisions.items() if k not in conflicts},roles,routing)
 result.update(scoring={k:len(v) for k,v in part.items()},history_coverage_complete=False,new_judge_attempts=0,conflicts_sensitivity=sensitive['groups'])
 write('RESULTS.json',result);write('COMPLETE.json',{'status':'SCORING_REUSE_COMPLETE','events':2148,'new_judge_attempts':0,'unknown_unique':sum(len(part[k]) for k in ('known_unknown','conflict_unknown','history_unverified'))})
if __name__=='__main__':
 code=0
 try:main()
 except BaseException:
  code=1;write('DRIVER_FAILURE.private.json',{'traceback':traceback.format_exc()});raise
 finally:write('DRIVER_EXIT.json',{'code':code,'time':time.time()})
