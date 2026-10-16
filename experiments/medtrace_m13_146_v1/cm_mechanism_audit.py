"""Fixed, post-result descriptive audit; no new inference or selected subsets."""
import collections
import json
import os
from pathlib import Path


def compare(a,b):
    assert set(a)==set(b)
    counts=collections.Counter();pairs=collections.Counter()
    for owner in a:
        left,right=a[owner],b[owner];assert len(left)==len(right)
        assert all(x in (0,1) for x in left+right)
        if not left:counts['no_support']+=1;continue
        delta=sum(right)-sum(left)
        counts['improved' if delta>0 else 'worse' if delta<0 else 'equal']+=1
        pairs.update(str(x)+'_to_'+str(y) for x,y in zip(left,right))
    return dict(owner_counts=dict(counts),query_transitions=dict(pairs),observations=sum(pairs.values()))


def main():
    assert compare({'a':[1,0],'b':[]},{'a':[0,1],'b':[]})==dict(owner_counts={'equal':1,'no_support':1},query_transitions={'1_to_0':1,'0_to_1':1},observations=2)
    run=Path(os.environ['RUN_ROOT']);data=json.loads((run/'private/CM_RESULTS_WITH_OWNER_AUDIT.json').read_text());assert data['complete_no_missing']
    roles=data['owner_panels']['role_panels'];result={'scope':'Post-result fixed DEV diagnostic; not independent causal confirmation','new_GPU_seconds':0,'new_Judge':0}
    result['bank_M13_vs_C']={role:compare(roles['C_bank_146'][role],roles['M13_bank_146'][role]) for role in ('FIT','T2G','T2L')}
    result['T2G_single_to_bank']={m:compare(roles[m+'_single_1']['T2G'],roles[m+'_bank_146']['T2G']) for m in ('C','M13')}
    assert all(v['observations']==557 for v in result['T2G_single_to_bank'].values())
    counts=collections.Counter();damage=data['owner_panels']['damage']
    for owner,a in damage['C_bank_146'].items():
        b=damage['M13_bank_146'][owner];assert not a['missing'] and not b['missing'] and a['fresh_Base_correct']==b['fresh_Base_correct']
        if not a['fresh_Base_correct']:counts['no_support']+=1;continue
        counts['more_damage' if b['new_damage']>a['new_damage'] else 'less_damage' if b['new_damage']<a['new_damage'] else 'equal']+=1
    result['damage_owner_counts']=dict(counts)
    p=data['panels'];single=p['M13']['single_1']['T2G']['macro']-p['C']['single_1']['T2G']['macro'];bank=p['M13']['bank_146']['T2G']['macro']-p['C']['bank_146']['T2G']['macro']
    result['T2G_macro_gap_pp']=dict(single=single,bank146=bank,bank_minus_single=bank-single)
    (run/'public/CM_MECHANISM_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
