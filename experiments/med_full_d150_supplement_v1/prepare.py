"""Freeze the user's 53 unknown payloads by their existing history category."""
import json
from pathlib import Path
import sys

w=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(w/'experiments/med_full_d150_v1'),str(w/'experiments/med_proposition_generalization_v1'),
              str(w/'experiments/med_fixed_depth_v1'),str(w/'experiments/med_source_teacher_v1'),
              str(w/'experiments/med150_source_review_v1')]
import ledger
import partition


def build(destination):
    destination=Path(destination)
    old=w/'.private/med_full_d150_v1_20261102/scoring'
    packet=json.loads((old/'PACKET_PRIVATE.json').read_text())
    prior=json.loads((old/'DECISIONS.json').read_text())
    complete=json.loads((old/'COMPLETE.json').read_text())
    if complete.get('unknown')!=53 or complete.get('new_attempts')!=0 or len(packet.get('consumers',[]))!=4296:
        raise ValueError('original completed scoring receipt differs')
    if json.loads((old/'GUARD_EXIT.json').read_text()).get('code')!=0:
        raise ValueError('original score guard did not exit successfully')
    h=json.loads(Path('/private/tmp/.r.20261102.03/HISTORY_INDEX_PRIVATE.json').read_text())
    if h.get('status')!='AVAILABLE_HISTORY_INDEX_FROZEN' or h.get('history_coverage_complete') is not False:
        raise ValueError('available history index missing')
    legacy={k for v in packet['payloads'].values() for k in v['legacy_keys']}
    semantic=set(packet['payloads']);valid=h['valid'];conflicts=set(h['conflicts'])
    files=[w/'.private/med146_single_sequential_v133_20261027/scoring_recovery1/MERGED_DECISIONS.json',
           w/'.private/med150_exact_switch_v1_20261028/DECISIONS.json',
           w/'.private/med150_balancedit_v1_20261028/scoring/DECISIONS.json',
           w/'.private/med150_memit_extension_v1_20261028/scoring/DECISIONS.json',
           w/'.private/med150_grace_v1_20261028/scoring/DECISIONS.json']
    if not all(f.is_file() for f in files):raise ValueError('original scoring evidence unavailable')
    for f in files:
        for key,value in ledger.selected_decisions(f,legacy).items():
            if key in valid and any(valid[key][k]!=value[k] for k in partition.FIELDS):conflicts.add(key)
            else:valid[key]=value
    for f in (Path('/private/tmp/.r.20261102.01/DECISIONS.json'),Path('/private/tmp/.r.20261102.02/DECISIONS.json'),
              Path('/private/tmp/.r.20261102.03/DECISIONS.json'),Path('/private/tmp/.r.20261101.09/DECISIONS.json')):
        if f.is_file():
            for key,value in ledger.selected_decisions(f,semantic).items():h['semantic_valid'].setdefault(key,[]).append(value)
    parts=partition.partition(packet,valid,h['attempts'],h['semantic_valid'],h['semantic_attempts'],conflicts)
    if len(parts['known_unknown'])!=0 or len(parts['conflict_unknown'])!=5 or len(parts['history_unverified'])!=48:
        raise ValueError('frozen unknown partition changed')
    if set(parts['known_valid'])!=set(prior['decisions']) or set(parts['conflict_unknown']+parts['history_unverified'])!=set(prior['unknown']):
        raise ValueError('original decision identities differ')
    ordered={k:sorted(parts[k]) for k in ('history_unverified','conflict_unknown')}
    payloads={k:packet['payloads'][k]['record'] for values in ordered.values() for k in values}
    if len(payloads)!=53 or any(payloads[k]['opaque_query_id']!=k for k in payloads):
        raise ValueError('supplement payload identity mismatch')
    destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    with (destination/'SUPPLEMENT_INPUT.private.json').open('x') as f:
        json.dump({'status':'SUPPLEMENT_IDENTITIES_FROZEN','categories':ordered,'payloads':payloads},f,ensure_ascii=False)
    (destination/'SUPPLEMENT_INPUT.private.json').chmod(0o600)
    return {'status':'SUPPLEMENT_IDENTITIES_FROZEN','history_unverified':48,'conflict_adjudication':5,
            'known_failed':0,'original_valid_reused':1376,'maximum_attempts':53}

if __name__=='__main__':
    import os
    print(json.dumps(build(os.environ['SUPPLEMENT_ROOT'])))
