"""Paired, aggregate-only audit of frozen MED development consumers."""
import collections
import json
import os
from pathlib import Path
import statistics

root = Path(os.environ['AUDIT_SOURCE'])
out = Path(os.environ['AUDIT_OUTPUT'])
def read(name):
    return json.loads((root / name).read_text())

contract = read('SCORING_CONTRACT.json')
assert contract['reasoning_effort'] == 'medium'
packet = read('PACKET_METADATA_B.json')
consumers = read('CONSUMERS_B_FROZEN.json')
decisions = read('MERGED_DECISIONS.json')['decisions']
assert consumers['protocol'] == contract['protocol']
records = {(v['arm'], v['mode'], v['prefix'], v['owner'], v['query_id']): v for v in consumers['consumers']}
assert len(records) == len(consumers['consumers'])
tasks = packet['tasks']
assert len(tasks) == 146 and {t['order'] for t in tasks} == set(range(1, 147))
assert len({t['edit_id'] for t in tasks}) == 146
pairs = []
for task in tasks:
    owner = task['order']
    qs = list(dict.fromkeys(q for e in task['events'] if e['task'] == 'T2G' for q in e['all_probe_query_ids']))
    for q in qs:
        if packet['base_mask'][q] is not False:
            continue
        one = records[('MED', 'single', 1, owner, q)]
        bank = records[('MED', 'bank', 146, 0, q)]
        assert all(one[k] == bank[k] for k in ('query_id', 'question', 'reference', 'image_sha256'))
        assert one['payload_key'] in decisions and bank['payload_key'] in decisions
        assert one['effective_expert'] in (None, task['edit_id'])
        selected = bank['effective_expert']
        assert selected is None or selected in {t['edit_id'] for t in tasks}
        group = 'OFF' if selected is None else 'same_owner' if selected == task['edit_id'] else 'different_owner'
        pairs.append((owner, group, one, bank))
assert len(pairs) == 557 and len({x[0] for x in pairs}) == 146
result = {'status':'COMPLETE', 'n_events':len(pairs), 'support_edits':146,
          'input_identity_checked':['query_id','question','reference','image_sha256'],
          'contract':'same frozen Astra medium scoring package',
          'runtime_causal_attribution':'not established by consumer packet alone',
          'independent_confirmation':False, 'panels':{}}
for field in ('core_correct','full_supported'):
    counts = {g:collections.Counter() for g in ('all','same_owner','different_owner','OFF')}
    by_owner = collections.defaultdict(lambda: [[], []])
    for owner, group, one, bank in pairs:
        a = decisions[one['payload_key']].get(field)
        b = decisions[bank['payload_key']].get(field)
        assert a in (True,False,None) and b in (True,False,None)
        by_owner[owner][0].append(a); by_owner[owner][1].append(b)
        for name in ('all',group):
            c=counts[name]; c['n']+=1; c['same_payload'] += one['payload_key']==bank['payload_key']
            c['single_off'] += one['effective_expert'] is None
            if a is None or b is None: c['unknown']+=1
            elif a and b: c['both_correct']+=1
            elif a: c['single_correct_bank_wrong']+=1
            elif b: c['single_wrong_bank_correct']+=1
            else: c['both_wrong']+=1
    assert all(sum(c[k] for k in ('unknown','both_correct','single_correct_bank_wrong','single_wrong_bank_correct','both_wrong'))==c['n'] for c in counts.values())
    complete=all(v is not None for rows in by_owner.values() for arm in rows for v in arm)
    macro = [100*statistics.mean(statistics.mean(arms[i]) for arms in by_owner.values()) for i in (0,1)] if complete else None
    result['panels'][field]={'groups':{k:dict(v) for k,v in counts.items()},'single_bank_macro_percent':macro}
assert result['panels']['core_correct']['groups']['all'].get('unknown',0)==0
assert round(result['panels']['core_correct']['single_bank_macro_percent'][0],4)==78.9384
assert round(result['panels']['core_correct']['single_bank_macro_percent'][1],4)==77.5114
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
