"""Freeze the explicitly authorized 35 unknown T2G judgments; preserve old results."""
import json
import os
from pathlib import Path
import sys

w = Path(os.environ['WORKTREE'])
root = Path(os.environ['LOCAL_SCORE_ROOT'])
sys.path[:0] = [str(w/'experiments'/x) for x in (
    'med_mdu150_v1', 'med_medinit_101_v1', 'med_medinit_loss_v1',
    'med_full_d150_v1', 'med_proposition_generalization_v1',
    'med_fixed_depth_v1', 'med_source_teacher_v1', 'med150_source_review_v1')]
import score_reuse
import summary

def read(p):
    return json.loads(Path(p).read_text())

def write(name, value):
    p = root/name
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False)
    p.chmod(0o600)

packet = read('/private/tmp/.r.20261105.150/PACKET.private.json')
base = read('/private/tmp/.r.20261104.102prep/BASELINE.private.json')
roles = read(w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json')
keys = set(packet['payloads'])
legacy = {k for v in packet['payloads'].values() for k in v['legacy_keys']}
history = dict(valid={}, conflicts=[], attempts=[], semantic_valid={}, semantic_attempts=[])
paths = sorted(Path('/private/tmp').glob('.r.2026110[2345]*/DECISIONS.json')) + [
    w/'.private/med_full_d150_v1_20261102/scoring/DECISIONS.json',
    w/'.private/med146_single_sequential_v133_20261027/scoring_recovery1/MERGED_DECISIONS.json',
    w/'.private/med150_balancedit_v1_20261028/scoring/DECISIONS.json']
for d in [base['decisions']] + [read(p).get('decisions', {}) for p in paths]:
    for k, v in d.items():
        if k in keys:
            history['semantic_valid'].setdefault(k, []).append(v)
        if k in legacy:
            if k in history['valid'] and any(history['valid'][k][f] != v[f] for f in score_reuse.FIELDS):
                history['conflicts'].append(k)
            else:
                history['valid'][k] = v
part = score_reuse.resolve(packet, history)
if {k:len(v) for k,v in part.items()} != read(w/'reports/med_mdu150_v1_20261105/EXECUTION_AUDIT.json')['scoring_partition']:
    raise ValueError('Original scoring partition does not reproduce')
decisions = dict(base['decisions'])
for suffix in ('102', '103'):
    new = read('/private/tmp/.r.20261104.'+suffix+'/DECISIONS.json')
    if new['unknown']:
        raise ValueError('Old supplement incomplete')
    for k,v in new['decisions'].items():
        if k in decisions and any(decisions[k][f] != v[f] for f in score_reuse.FIELDS):
            raise ValueError('Old supplement conflict')
        decisions[k] = v
blocked = set(part['conflict_unknown'])
for k,v in part['known_valid'].items():
    if k in decisions and any(decisions[k][f] != v[f] for f in score_reuse.FIELDS):
        blocked.add(k)
    else:
        decisions[k] = v
selected = {r['consumer_id'] for r in base['routing']['records'] if r['group']=='med49'}
consumers = []
for r in base['consumers']:
    if r['arm']=='MED':
        consumers.append(dict(r))
    elif r['arm']=='MDU':
        consumers.append(dict(r, arm='MDU101'))
        if r['consumer_id'] not in selected:
            consumers.append(dict(r, arm='MDU150'))
consumers += [dict(r, arm='MDU150') for r in packet['consumers'] if r['arm']=='CANDIDATE']
be_root = w/'.private/med150_balancedit_v1_20261028/scoring'
be = read(be_root/'PACKET_PRIVATE.json')
bed = read(be_root/'DECISIONS.json')
if bed['unknown']:
    raise ValueError('BE judgments incomplete')
for k,v in bed['decisions'].items():
    if k in decisions and any(decisions[k][f] != v[f] for f in score_reuse.FIELDS):
        blocked.add(k)
    else:
        decisions[k] = v
for k in blocked:
    decisions.pop(k, None)
consumers += [dict(arm='BE', query_id=r['query_id'], key=r['key'])
              for r in be['consumers'] if r['mode']=='bank' and r['prefix']==150]
old = read(w/'reports/med_mdu150_v1_20261105/RESULTS.json')
reproduced = summary.compare(consumers, decisions, roles, base['routing'])
if reproduced['groups'] != old['groups']:
    differences = [(g,role,field) for g in old['groups'] for role in summary.ROLES
                   for field in summary.FIELDS
                   if reproduced['groups'][g][role][field] != old['groups'][g][role][field]]
    raise ValueError('Original panels do not reproduce: '+repr(differences))
t2g = {q for owner in roles['roles'].values() for q in owner['T2G']}
events = [r for r in consumers if r['arm']=='MDU150' and r['query_id'] in t2g and r['key'] not in decisions]
targets = {r['key'] for r in events}
if len(events)!=35 or len(targets)!=35 or not targets <= set(part['history_unverified']):
    raise ValueError('Authorized unknown T2G cohort differs')
payloads = {k:packet['payloads'][k]['record'] for k in sorted(targets)}
write('BASELINE.private.json', dict(consumers=consumers, decisions=decisions, roles=roles,
                                  routing=base['routing'], target_events=events))
write('SUPPLEMENT_INPUT.private.json', dict(status='SUPPLEMENT_IDENTITIES_FROZEN', payloads=payloads,
      categories=dict(history_unverified=sorted(targets), conflict_unknown=[], known_unknown=[])))
write('PREPARE_COMPLETE.json', dict(status='BASELINE_REPRODUCED', target_events=35,
      unique_payloads=35, history_unverified=35, known_failed=0, conflicts=0,
      original_all_groups_reproduced=True, training=0, generation=0, new_judge_attempts=0))
print('BASELINE_REPRODUCED; 35 unknown T2G payloads frozen')
