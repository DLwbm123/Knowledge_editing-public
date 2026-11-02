import json,sys,os
from pathlib import Path
w=Path(os.environ['WORKTREE']);root=Path(os.environ['SUPPLEMENT_ROOT']);old=Path('/private/tmp/.r.20261102.05')
sys.path[:0]=[str(w/'experiments'/x) for x in ('med_full_d150_v1','med_proposition_generalization_v1','med_fixed_depth_v1','med_source_teacher_v1','med150_source_review_v1')]
import ledger,partition
packet=json.loads((old/'PACKET_PRIVATE.json').read_text());prior=json.loads((old/'DECISIONS.json').read_text());complete=json.loads((old/'COMPLETE.json').read_text())
assert complete['new_attempts']==0 and complete['unknown']==161 and len(packet['consumers'])==4296
assert json.loads((old/'GUARD_EXIT.json').read_text())['code']==0
roles=json.loads((w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
queries={q for a in roles['roles'].values() for q in a['T2G']}
target={x['key'] for x in packet['consumers'] if x['query_id'] in queries}&set(prior['unknown'])
h=json.loads(Path('/private/tmp/.r.20261102.03/HISTORY_INDEX_PRIVATE.json').read_text());assert h['history_coverage_complete'] is False
legacy={k for v in packet['payloads'].values() for k in v['legacy_keys']};semantic=set(packet['payloads']);valid=h['valid'];conflicts=set(h['conflicts'])
for f in (w/'.private/med146_single_sequential_v133_20261027/scoring_recovery1/MERGED_DECISIONS.json',w/'.private/med150_exact_switch_v1_20261028/DECISIONS.json',w/'.private/med150_balancedit_v1_20261028/scoring/DECISIONS.json',w/'.private/med150_memit_extension_v1_20261028/scoring/DECISIONS.json',w/'.private/med150_grace_v1_20261028/scoring/DECISIONS.json'):
 if not f.is_file():raise ValueError('missing history evidence')
 for k,v in ledger.selected_decisions(f,legacy).items():
  if k in valid and any(valid[k][j]!=v[j] for j in partition.FIELDS):conflicts.add(k)
  else:valid[k]=v
for f in (Path('/private/tmp/.r.20261102.01/DECISIONS.json'),Path('/private/tmp/.r.20261102.02/DECISIONS.json'),Path('/private/tmp/.r.20261102.03/DECISIONS.json'),Path('/private/tmp/.r.20261101.09/DECISIONS.json')):
 if f.is_file():
  for k,v in ledger.selected_decisions(f,semantic).items():h['semantic_valid'].setdefault(k,[]).append(v)
parts=partition.partition(packet,valid,h['attempts'],h['semantic_valid'],h['semantic_attempts'],conflicts)
assert set(parts['known_valid'])==set(prior['decisions']) and set(parts['known_unknown']+parts['conflict_unknown']+parts['history_unverified'])==set(prior['unknown'])
cats={k:sorted(target & set(parts[k])) for k in ('known_unknown','conflict_unknown','history_unverified')}
assert set.union(*(set(v) for v in cats.values()))==target
root.mkdir(mode=0o700,parents=True,exist_ok=False)
payloads={k:packet['payloads'][k]['record'] for n in ('history_unverified','conflict_unknown') for k in cats[n]}
assert all(v['opaque_query_id']==k for k,v in payloads.items())
f=root/'SUPPLEMENT_INPUT.private.json';f.write_text(json.dumps({'status':'SUPPLEMENT_IDENTITIES_FROZEN','categories':cats,'payloads':payloads},ensure_ascii=False));f.chmod(0o600)
print(json.dumps({'targets':len(target),'categories':{k:len(v) for k,v in cats.items()},'maximum_attempts':len(payloads),'role_events':len(queries)}))
