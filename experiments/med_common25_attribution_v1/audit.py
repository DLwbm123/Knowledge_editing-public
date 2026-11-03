"""Select the fixed routed cohort without decoding question or answer payloads."""
import json, resource, signal, sys
from pathlib import Path

W = Path.cwd()
sys.path[:0] = [str(W / 'experiments' / p) for p in ('med_fixed_depth_v1', 'med150_source_review_v1', 'med_full_d150_v1', 'med_proposition_generalization_v1', 'med_source_teacher_v1')]
import prior_binding as pb
import stream_extract as s
import ledger
import importlib.util
spec = importlib.util.spec_from_file_location('routed_report', W / 'experiments/med_full_d150_v1/report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('audit time limit')))
signal.alarm(60)


def consumers(path):
    out = {}
    with path.open('rb') as f:
        if s.nonspace(f) != b'{': raise ValueError('packet object expected')
        while True:
            t = s.nonspace(f)
            if t == b'}': break
            if t != b'"': raise ValueError('packet field expected')
            name = s.string(f, True)
            if s.nonspace(f) != b':': raise ValueError('packet colon expected')
            t = s.nonspace(f)
            if name != 'consumers': pb.skip(f, t)
            else:
                if t != b'[': raise ValueError('consumers array expected')
                while True:
                    t = s.nonspace(f)
                    if t == b']': break
                    row = pb.select(f, t, {k: True for k in ('consumer_id', 'query_id', 'arm', 'key', 'owner', 'role', 'route', 'candidate_expert')})
                    k = (row['consumer_id'], row['arm'])
                    if k in out: raise ValueError('duplicate consumer')
                    out[k] = row
                    t = s.nonspace(f)
                    if t == b']': break
                    if t != b',': raise ValueError('consumer delimiter')
            t = s.nonspace(f)
            if t == b'}': break
            if t != b',': raise ValueError('packet delimiter')
    if len(out) != 4296: raise ValueError('packet consumer count')
    return out


def main():
    d = consumers(Path('/private/tmp/.r.20261102.06/PACKET_PRIVATE.json'))
    u = consumers(Path('/private/tmp/.r.20261102.05/PACKET_PRIVATE.json'))
    if set(d) != set(u): raise ValueError('consumer sets differ')
    selected = []
    for k, a in d.items():
        b = u[k]
        if any(a[f] != b[f] for f in ('consumer_id', 'query_id', 'arm', 'owner', 'role', 'route')):
            raise ValueError('identity or routing mismatch')
        if a['arm'] == 'MED' and a['key'] != b['key']: raise ValueError('MED semantic binding mismatch')
        if a['candidate_expert'] != d[(a['consumer_id'], 'MED')]['candidate_expert']:
            raise ValueError('arm membership mismatch')
        if a['candidate_expert'] and not b['candidate_expert']: raise ValueError('D candidate not in DU')
        if a['arm'] == 'MED_D' and a['candidate_expert']: selected.append(a['consumer_id'])
    if len(selected) != 392 or len(set(selected)) != 392: raise ValueError('candidate route coverage mismatch')
    by_query = {}
    for cid in selected:
        a, b, med = d[(cid, 'MED_D')], u[(cid, 'MED_D')], d[(cid, 'MED')]
        if a['query_id'] in by_query: raise ValueError('duplicate query')
        by_query[a['query_id']] = {'D': a['key'], 'DU': b['key'], 'MED': med['key']}
    needed = {key for row in by_query.values() for key in row.values()}
    ds = ledger.selected_decisions(Path('/private/tmp/.r.20261102.06/DECISIONS.json'), needed)
    us = ledger.selected_decisions(Path('/private/tmp/.r.20261102.05/DECISIONS.json'), needed)
    extra = ledger.selected_decisions(Path('/private/tmp/.r.20261102.08/DECISIONS.json'), needed)
    for key, val in extra.items():
        if key in us and us[key] != val: raise ValueError('supplement decision conflict')
        us[key] = val
    roles = json.loads((W / '.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
    if roles.get('status') != 'EVALUATION_ROLES_FROZEN' or roles.get('owners') != 150: raise ValueError('roles not frozen')
    result = {}
    for role in report.ROLES:
        entries = [(o,q) for o in range(1,151) for q in roles['roles'][str(o)][role] if q in by_query]
        result[role] = {}
        for field in report.FIELDS:
            vals = {a: [] for a in ('MED','D','DU')}
            paired = {'MED_to_D':[0,0,0], 'MED_to_DU':[0,0,0], 'D_to_DU':[0,0,0]}
            for owner, q in entries:
                keys = by_query[q]
                v = {'MED': ds.get(keys['MED'], {}).get(field), 'D': ds.get(keys['D'], {}).get(field), 'DU': us.get(keys['DU'], {}).get(field)}
                for arm in vals: vals[arm].append((owner,v[arm]))
                for name, left, right in (('MED_to_D','MED','D'),('MED_to_DU','MED','DU'),('D_to_DU','D','DU')):
                    x,y=v[left],v[right]
                    if x is None or y is None: paired[name][2] += 1
                    elif not x and y: paired[name][0] += 1
                    elif x and not y: paired[name][1] += 1
            result[role][field]={'arms':{arm:report.summarize(v) for arm,v in vals.items()},'pairs_gain_loss_unknown':paired}
    out = {'status':'COMPLETE','route_selected_events':392,'total_events':2148,'panels':result,'supplement':'14 prior authorized valid decisions exact-key matched; all other unknown unchanged','interpretation':'exposed DEV, selected by routed expert not owner; not causal','new_training_generation_judge':0}
    path=W/'reports/med_common25_attribution_v1_20261103/RESULTS.json'
    with path.open('x') as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'COMPLETE','events':392}))

if __name__=='__main__': main()
