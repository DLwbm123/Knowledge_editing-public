"""Reconstruct the prior frozen comparison and select only its T1G unknowns."""
import json
import os
from pathlib import Path
import runpy

w = Path(os.environ['WORKTREE'])
root = Path(os.environ['LOCAL_SCORE_ROOT'])
source = (w / 'experiments/med_medinit_loss_v1/score_driver.py').read_text()
start = source.index('    remote = os.environ[\'EXPERIMENT_ROOT\']')
end = source.index('    legacy = {key for p in outputs.values()', start)
source = source[:start] + '''    outputs = {arm: json.loads((Path(os.environ['PREVIOUS_SCORE_ROOT']) / ('PACKET_' + arm + '.private.json')).read_text()) for arm in ('MCE', 'MDU')}
    outputs['HCE'] = json.loads(Path(os.environ['OLD_CE_PACKET']).read_text())
''' + source[end:]
needle = '    result = summary.compare(consumers, decisions, roles)'
insertion = '''    result = summary.compare(consumers, decisions, roles)
    prior = json.loads((Path(os.environ['PREVIOUS_SCORE_ROOT']) / 'RESULTS.json').read_text())
    if result['panels'] != prior['panels']:
        raise ValueError('reconstructed panels differ from original')
    recovered = json.loads((Path(os.environ['T2_RECOVERY_ROOT']) / 'DECISIONS.json').read_text())
    if len(recovered['decisions']) != 43 or recovered['unknown'] or set(recovered['decisions']) & set(decisions):
        raise ValueError('T2 recovery decisions differ')
    decisions.update(recovered['decisions'])
    recovered_summary = summary.compare(consumers, decisions, roles)
    expected = json.loads((worktree / 'reports/med_medinit_t2_recovery_v1_20261104/RESULTS.json').read_text())
    if recovered_summary['panels'] != expected['panels']:
        raise ValueError('T2-recovered baseline differs')
    t2 = {q for owner in range(1, 151) for role in ('T1G',) for q in roles['roles'][str(owner)][role]}
    targets = {row['key'] for row in consumers if row['arm'] in ('MCE', 'MDU') and row['query_id'] in t2 and row['key'] not in decisions}
    categories = {'history_unverified': [], 'conflict_unknown': [], 'known_unknown': []}
    payloads = {}
    for arm in ('MCE', 'MDU'):
        packet = outputs[arm]
        part = score_reuse.resolve(packet, history)
        for category in categories:
            for key in set(part[category]) & targets:
                if key in payloads and payloads[key] != packet['payloads'][key]['record']:
                    raise ValueError('shared payload differs')
                payloads[key] = packet['payloads'][key]['record']
                categories[category].append(key)
    categories = {k: sorted(set(v)) for k, v in categories.items()}
    if set().union(*(set(v) for v in categories.values())) != targets:
        raise ValueError('unclassified target')
    if sum(len(v) for v in categories.values()) != len(targets):
        raise ValueError('conflicting target classifications')
    if any(payloads[k]['opaque_query_id'] != k for k in payloads):
        raise ValueError('payload identity mismatch')
    write('BASELINE.private.json', {'consumers': consumers, 'decisions': decisions})
    write('SUPPLEMENT_INPUT.private.json', {'status': 'SUPPLEMENT_IDENTITIES_FROZEN', 'categories': categories, 'payloads': payloads})
    write('PREPARE_COMPLETE.json', {'counts': {k: len(v) for k, v in categories.items()}, 'total': len(targets)})
    return
'''
if source.count(needle) != 1:
    raise ValueError('unexpected source')
source = source.replace(needle, insertion)
exec(compile(source, '<frozen-comparison-reconstruction>', 'exec'), {'__name__': '__main__'})
