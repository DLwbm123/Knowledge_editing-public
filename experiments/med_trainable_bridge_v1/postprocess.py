"""Reaggregate frozen decisions with later existing same-contract evidence, without new judge calls."""
import json,os,resource,signal,subprocess,sys
from pathlib import Path
w=Path(os.environ['WORKTREE']);root=Path('/private/tmp/.r.20261103.01')
resource.setrlimit(resource.RLIMIT_FSIZE,(50_000_000,50_000_000));signal.alarm(180)
if json.loads((root/'GUARD_EXIT.json').read_text())['code']!=0 or json.loads((root/'DRIVER_EXIT.json').read_text())['code']!=0:raise ValueError('scoring not complete')
sys.path[:0]=[str(w/'experiments'/x) for x in ('med_full_d150_v1','med_proposition_generalization_v1','med_fixed_depth_v1','med_source_teacher_v1','med150_source_review_v1')]
import ledger,partition,report
packet=json.loads((root/'PACKET_PRIVATE.json').read_text());original=json.loads((root/'DECISIONS.json').read_text());h=json.loads(Path('/private/tmp/.r.20261102.03/HISTORY_INDEX_PRIVATE.json').read_text());valid=h['valid'];conflicts=set(h['conflicts']);legacy={k for v in packet['payloads'].values() for k in v['legacy_keys']};semantic=set(packet['payloads'])
for f in (w/'.private/med146_single_sequential_v133_20261027/scoring_recovery1/MERGED_DECISIONS.json',w/'.private/med150_exact_switch_v1_20261028/DECISIONS.json',w/'.private/med150_balancedit_v1_20261028/scoring/DECISIONS.json',w/'.private/med150_memit_extension_v1_20261028/scoring/DECISIONS.json',w/'.private/med150_grace_v1_20261028/scoring/DECISIONS.json'):
 if not f.is_file():raise ValueError('original evidence missing')
 for k,v in ledger.selected_decisions(f,legacy).items():
  if k in valid and any(valid[k][j]!=v[j] for j in partition.FIELDS):conflicts.add(k)
  else:valid[k]=v
paths=[Path('/private/tmp/.r.20261102.'+s+'/DECISIONS.json') for s in ('01','02','03','05','06','08')]+[Path('/private/tmp/.r.20261101.09/DECISIONS.json')]
for f in paths:
 if f.is_file():
  for k,v in ledger.selected_decisions(f,semantic).items():h['semantic_valid'].setdefault(k,[]).append(v)
parts=partition.partition(packet,valid,h['attempts'],h['semantic_valid'],h['semantic_attempts'],conflicts)
reused=parts['known_valid'];unknown=parts['known_unknown']+parts['conflict_unknown']+parts['history_unverified']
changed=sum(k in reused and any(v[j]!=reused[k][j] for j in partition.FIELDS) for k,v in original['decisions'].items())
if changed:raise ValueError('existing verdict modified')
roles=json.loads((w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
script="import json;from pathlib import Path;p=Path('/data/bmw/Knowledge_editing/outputs/med-trainable-bridge-v1-20261103/run');print(json.dumps(json.loads((p/'ROLE_ADMISSION.json').read_text())['owners']))"
x=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],input=script,text=True,capture_output=True,timeout=30)
if x.returncode:raise RuntimeError('cohort identities unavailable')
ids=json.loads(x.stdout);assert len(ids)==101 and len(set(ids))==101
info={'original_reused':len(original['decisions']),'reused_now':len(reused),'remaining_unknown':len(unknown),'new_judge':0,'original_valid_now_conflicting':len(set(original['decisions'])-set(reused)),'known_unknown':len(parts['known_unknown']),'conflict_unknown':len(parts['conflict_unknown']),'history_unverified':len(parts['history_unverified']),'history_coverage_complete':False}
result=report.run(packet,reused,roles,info,cohorts={'admitted101':ids,'unresolved49':sorted(set(range(1,151))-set(ids))});result['candidate_experts']=101;result['fallback_experts']=49
with (root/'REAGG_DECISIONS.private.json').open('x') as f:json.dump({'decisions':reused,'unknown':unknown},f)
(root/'REAGG_DECISIONS.private.json').chmod(0o600)
dest=w/'reports/med_trainable_bridge_v1_20261103'
with (dest/'RESULTS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':result['status'],'info':info,'T2G':result['panels']['all']['T2G']['core_correct'],'T2L':result['panels']['all']['T2L']['core_correct']}))
