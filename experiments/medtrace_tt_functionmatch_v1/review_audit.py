"""Read back registered schedules and matching precision from anonymous reports."""
import json
import statistics as s
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]/'reports/medtrace_tt_functionmatch_v1_20261009'
    r=json.loads((root/'RESULTS.json').read_text())
    schedules=json.loads((root/'FUNCTION_SCHEDULES.json').read_text())
    fixed={(x['expert_order'],x['mode']):x['norms'] for x in schedules}
    assert len(fixed)==16 and all(len(x)==320 for x in fixed.values())
    assert len(r['training'])==16 and r['scoring']['valid']==1855 and r['scoring']['missing']==0
    rows=[]
    for arm,mode in [('RAW_A_SCHEDULE','ADAM'),('ADAM_R_SCHEDULE','RAW')]:
        selected=[x for x in r['training'] if x['arm']==arm]
        assert len(selected)==8
        values=[]
        for x in selected:
            assert len(x['curve'])==320 and x['forwards']==640 and x['backwards']==640
            for row,target in zip(x['curve'],fixed[x['expert_order'],mode]):
                m=row['function_match'];assert m['target']==target and m['actual']==row['actual_map_norm']
                assert abs(m['actual']-target)<=1e-10+1e-3*target and 0<=m['scale']<=16
                values.append(m)
        rows.append(dict(arm=arm,steps=len(values),maximum_relative_error=max(x['relative_error'] for x in values),
            scale_min=min(x['scale'] for x in values),scale_median=s.median(x['scale'] for x in values),scale_max=max(x['scale'] for x in values),
            final_native_FIT_all_correct=all(z['all_argmax_correct'] for x in selected for z in x['final'])))
    ratios=[b/a for order in range(1,9) for a,b in zip(fixed[order,'ADAM'],fixed[order,'RAW'])]
    assert len(ratios)==2560
    assert not r['primary_screen_pass'] and not r['reverse_direction_screen_pass'] and r['generation_cap_pass']
    result=dict(status='PASS',same_expert_step_schedule_binding=True,reference_endpoints_exact=16,matched_training_steps=5120,
        arms=rows,reference_R_over_A=dict(n=len(ratios),minimum=min(ratios),median=s.median(ratios),maximum=max(ratios),above_one=sum(x>1 for x in ratios)),
        original_decisions_preserved=True,posthoc_descriptive_only=True)
    (root/'REVIEW_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
