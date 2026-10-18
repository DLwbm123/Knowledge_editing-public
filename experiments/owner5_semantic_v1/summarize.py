"""Aggregate frozen receipts and already-written semantic verdicts; no inference."""
import collections
import json
import sys
from pathlib import Path

run, destination = map(Path, sys.argv[1:])
read = lambda name: json.loads((run/name).read_text())
assert read('EXIT.json')['code'] == 0
complete = read('COMPLETE.json')
assert complete['status'] == 'COMPLETE' and complete['counts']['updates'] == 320
mapping = read('BLIND_MAPPING.json')
verdicts = read('BLIND_VERDICTS.json')['verdicts']
assert len(verdicts) == len(mapping) == 27
assert {row['id'] for row in verdicts} == set(mapping)
allowed = {'CORRECT', 'INCORRECT', 'UNCERTAIN', 'CONTRADICTORY'}
roles = {'edit_5_0':'native', 'positive_construct':'positive_construct',
         'negative_construct':'negative_construct', 'positive_check':'positive_check',
         'negative_check':'negative_check'}
roles.update({f'edit_5_{i}':'fit' for i in range(1,5)})
semantic = {}; by_query = {}
for row in verdicts:
    assert row['verdict'] in allowed
    info = mapping[row['id']]; arm = info['arm']; query = info['query']; role = roles[query]
    semantic.setdefault(arm, {}).setdefault(role, collections.Counter())[row['verdict']] += 1
    by_query.setdefault(arm, {})[query] = row['verdict']
assert all(len(rows) == 9 for rows in by_query.values())
proxy = {arm:read(arm+'_SEMANTIC.json') for arm in ('BASE','CONTROL','PROTECTED')}
training = {}
for arm in ('CONTROL','PROTECTED'):
    record = read(arm+'_TRAIN.json'); curve = record['curve']
    assert record['steps'] == len(curve) == 160 and record['save_load_exact']
    scores = [v['guard']['score'] for v in curve]
    training[arm] = dict(steps=160, active_protection_steps=[v['step'] for v in curve if v['guard']['applied']],
        hinge_positive_steps=sum(v['guard']['hinge']>0 for v in curve),
        pre_update_score_min=min(scores), pre_update_score_max=max(scores),
        final_score=proxy[arm]['score'], change_from_base=proxy[arm]['score']-proxy['BASE']['score'])
    assert len(training[arm]['active_protection_steps']) <= 160
same = sum(read('outputs/CONTROL/'+q+'.json')['raw_token_ids'] ==
           read('outputs/PROTECTED/'+q+'.json')['raw_token_ids'] for q in by_query['CONTROL'])
outputs = [read('outputs/'+arm+'/'+q+'.json') for arm, rows in by_query.items() for q in rows]
assert same == 9 and all(v == 'CORRECT' for arm in ('CONTROL','PROTECTED') for v in by_query[arm].values())
result = dict(status='NO_INCREMENTAL_SEMANTIC_PROTECTION_BENEFIT_OBSERVED',
    owner_count=1, seed_count=1, lambda_protection=1.0, semantic_verdicts=semantic,
    proxy=proxy, training=training, identical_control_protected_token_sequences=same,
    outputs=len(outputs), eos_terminated=sum(v['EOS'] for v in outputs), at_token_cap=sum(v['at_cap'] for v in outputs),
    execution=complete, adjudication='Assistant semantic review against fixed references with arm labels hidden before verdicts; no independent clinical adjudication',
    limitations=['One owner and one independent negative check', 'Control has no semantic damage on either negative',
                 'Positive check already correct at Base', 'Training proxy is not semantic outcome'],
    automatic_next_run=False)
destination.mkdir(parents=True,exist_ok=True)
(destination/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(semantic=semantic,training=training,identical_sequences=same),indent=2))
