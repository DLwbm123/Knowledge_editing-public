"""Aggregate paired fixed-panel decisions without decoding QA or answers."""
import json
import os
from collections import Counter
from pathlib import Path
import statistics
import sys

worktree = Path(os.environ['PAIR_WORKTREE'])
sys.path[:0] = [str(worktree/'experiments/med150_source_review_v1'),
                str(worktree/'experiments/med_fixed_depth_v1')]
import stream_extract as s
import score_index


def consumer_ids(path, wanted):
    result = {}
    def consumers(f):
        if s.nonspace(f) != b'[': raise ValueError('consumers array expected')
        while True:
            t = s.nonspace(f)
            if t == b']': break
            if t != b'{': raise ValueError('consumer object expected')
            row = score_index.fields(f, {'owner','query_id','arm','key'})
            key = (row.get('owner'),row.get('query_id'),row.get('arm'))
            if key[:2] in wanted and key[2] in ('C21','CDEEP'):
                if key in result or not row.get('key'): raise ValueError('duplicate or missing identity')
                result[key] = row['key']
            t=s.nonspace(f)
            if t == b']': break
            if t != b',': raise ValueError('array delimiter')
    with path.open('rb') as f:
        if s.nonspace(f) != b'{': raise ValueError('root object expected')
        found=False
        while True:
            t=s.nonspace(f)
            if t == b'}': break
            if t != b'"': raise ValueError('root field')
            field=s.string(f,True)
            if s.nonspace(f)!=b':': raise ValueError('root colon')
            if field=='consumers':
                if found: raise ValueError('duplicate consumers')
                found=True; consumers(f)
            else: score_index.skip(f,s.nonspace(f))
            t=s.nonspace(f)
            if t==b'}': break
            if t!=b',': raise ValueError('root delimiter')
    if not found or set(result)!={(o,q,a) for o,q in wanted for a in ('C21','CDEEP')}:
        raise ValueError('fixed event or arm missing')
    return result


def main():
    score_root=Path(os.environ['PAIR_SCORE_ROOT'])
    roles=json.loads((worktree/'.private/med_fixed_depth_v1_20261030/EVALUATION_ROLES.json').read_text())
    if len(roles)!=16: raise ValueError('owner coverage')
    panel={role:{(int(o),q) for o,g in roles.items() for q in g[role]} for role in ('T2G','T2L')}
    if (len(panel['T2G']),len(panel['T2L']))!=(64,30) or panel['T2G']&panel['T2L']:
        raise ValueError('role coverage')
    keys=consumer_ids(score_root/'PACKET_PRIVATE.json',panel['T2G']|panel['T2L'])
    decisions=score_index.decisions(score_root/'DECISIONS.json',set(keys.values()))
    result={}; by_owner={int(o):{} for o in roles}
    for role,events in panel.items():
        pairs=[]
        for o,q in events:
            a=decisions[keys[o,q,'C21']]['core_correct']
            b=decisions[keys[o,q,'CDEEP']]['core_correct']
            pairs.append((o,a,b)); by_owner[o].setdefault(role,[]).append((a,b))
        bins=Counter(('correct' if a else 'wrong')+'_to_'+('correct' if b else 'wrong') for _,a,b in pairs)
        groups={o:[int(b)-int(a) for oo,a,b in pairs if oo==o] for o in roles_as_int(roles)}
        groups={o:x for o,x in groups.items() if x}
        result[role]={'events':len(events),'edits':len(groups),'transitions':dict(sorted(bins.items())),
                      'macro_delta_pp':100*statistics.mean(statistics.mean(x) for x in groups.values())}
    with_gain_with_damage=sum(any(not a and b for a,b in g.get('T2G',[])) and any(a and not b for a,b in g.get('T2L',[])) for g in by_owner.values())
    result['edit_overlap']={'t2g_repair_and_t2l_damage':with_gain_with_damage,'edits':16}
    result['limits']=['post hoc exposed DEV; no causal or clinical claim','does not establish patient independence','T1L unsupported']
    out=worktree/'reports/med_depth_pairwise_v1_20261031/RESULTS.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


def roles_as_int(roles): return [int(o) for o in roles]

if __name__=='__main__': main()
