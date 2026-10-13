"""Anonymous, paired comparison from existing records; never loads a model."""
import json
from pathlib import Path
import statistics


def summary(values):
    return dict(n=len(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values)) if values else dict(n=0)


def read_parent(parent):
    parent=Path(parent)
    rows=[json.loads(p.read_text()) for p in sorted((parent/'private/results_joint').glob('*.json'))]
    assert len(rows)==8 and {r['owner'] for r in rows}==set(range(1,9))
    assert all(r['status']=='COMPLETE' and r['save_load_exact'] for r in rows)
    return rows


def paired(rows):
    groups=[]
    for fraction in (1.,.5,.25,.125):
        all_old=[]; pairs=[]
        for row in rows:
            old=next(t for t in row['old_decision']['checks'] if t['fraction']==fraction)
            new=next(t for t in row['fractions'] if t['fraction']==fraction)
            assert old['pure_source_gain']>0 and not old['accepted']
            all_old.append(old['source_gain']/old['pure_source_gain'])
            if new['solver']['status']=='LOCAL_KKT_PASS':
                assert abs(new['pure_gain']-old['pure_source_gain'])<1e-5
                pairs.append(dict(old_gain_ratio=old['source_gain']/new['pure_gain'],
                                  joint_gain_ratio=new['actual_source_gain']/new['pure_gain'],
                                  gain_ratio_change=(new['actual_source_gain']-old['source_gain'])/new['pure_gain'],
                                  old_step_ratio=old['operator_norm']/old['pure_norm'],
                                  joint_step_ratio=new['step_ratio']))
        groups.append(dict(fraction=fraction,original_projection_all_eight=summary(all_old),
                           paired_candidates=len(pairs),
                           paired={key:summary([p[key] for p in pairs]) for key in
                                   ('old_gain_ratio','joint_gain_ratio','gain_ratio_change','old_step_ratio','joint_step_ratio')}))
    return dict(status='COMPLETE',fixed_states=8,local_problems=32,paired_candidates=6,paired_unique_states=5,
                new_model_calls=0,comparison='Original rejected projection versus joint candidate at identical state/fraction',
                per_fraction=groups,basis_pair_available=False,
                limitations='Conditional on six prior local passes; no all-state causal or generalization claim. Original rejected projection BASIS metrics were not recorded.')


if __name__=='__main__':
    import os
    print(json.dumps(paired(read_parent(os.environ['JOINT_PARENT'])),ensure_ascii=False,indent=2))
