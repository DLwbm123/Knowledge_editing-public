"""Audit receipts and summarize backend/precision contrasts without choosing a candidate."""
import json
import math
import os
from pathlib import Path

RUN=Path(os.environ['RUN_ROOT'])
read=lambda path:json.loads(path.read_text())
result=read(RUN/'public/RESULTS.json')
completion=read(RUN/'public/COMPLETION_AUDIT.json')
lifecycle=read(RUN/'public/LIFECYCLE.json')
assert (RUN/'private/REPORT_COMPLETE.json').is_file()
assert result['decision']=='CALIBRATION_ONLY_NO_METHOD_PROMOTION'
assert completion['counts']['parent_response_checks']==1058
assert completion['counts']['JVP_zero_checks']==2116
assert completion['counts']['zero_Base_checks']==completion['counts']['repeat_checks']==27
assert lifecycle['temporary_candidate_packages_deleted']==16
assert not list((RUN/'private').glob('FAILURE*'))
data=[read(p) for p in sorted((RUN/'private/results').glob('*.json'))]
assert len(data)==8
for item in data:
    assert item['state_restored_exact'] and item['original_dtypes_restored']
    assert len(item['rows'])==1188
    for row in item['rows']:
        for field in ('KL','observed_logit_quadratic','logit_RMS','baseline_KL_from_native'):
            assert math.isfinite(row[field]) and row[field]>=0
        assert (row['JVP_quadratic'] is None)==(row['precision']=='ORIGINAL_FP16')
        if row['JVP_quadratic'] is not None:assert math.isfinite(row['JVP_quadratic']) and row['JVP_quadratic']>=0
cells={(x['precision'],x['arm'],x['scale']):x for x in result['panels']}
assert len(cells)==18
summary=[]
for precision in ('ORIGINAL_FP16','MATH_FP16','MATH_FP32'):
    raw,bounded=[cells[precision,arm,1.] for arm in ('RAW','BOUNDED')]
    summary.append(dict(precision=precision,RAW_source_KL=raw['KL'],BOUNDED_source_KL=bounded['KL'],
        source_KL_relative_reduction=1-bounded['KL']/raw['KL'] if raw['KL']>0 else None,
        mean_edit_retention=bounded['edit_progress']/raw['edit_progress'] if raw['edit_progress']>0 else None,
        RAW_actual_over_JVP=raw['actual_over_JVP'],BOUNDED_actual_over_JVP=bounded['actual_over_JVP'],
        RAW_response_relative_error=raw['response_relative_error'],BOUNDED_response_relative_error=bounded['response_relative_error'],
        RAW_KL_over_scale_squared={str(scale):cells[precision,'RAW',scale]['KL']/scale**2 for scale in (.25,.5,1.)},
        BOUNDED_KL_over_scale_squared={str(scale):cells[precision,'BOUNDED',scale]['KL']/scale**2 for scale in (.25,.5,1.)},
        baseline_KL_from_native=raw['baseline_KL_from_native']))
ledger=read(RUN/'RESOURCE_LEDGER.json')
assert len(ledger['gpu_sessions'])==7 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
audit=dict(status='PASS',summary=summary,full_experiment_forwards=13827,
    full_experiment_completed_backwards=19905,failed_backward_attempts=1,
    original_failed_call_count_inferred_from_control_flow=True,
    full_experiment_new_GPU_hours=result['resource']['new_GPU_process_hours']+result['inherited_engineering_failure']['original_GPU_process_hours'],
    cumulative_GPU_hours=result['resource']['cumulative_GPU_process_hours'],cumulative_Judge=12406,
    preserved_original_failure=True,candidate_selection=False,heldout_promotion=False)
(RUN/'public/REVIEW_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
