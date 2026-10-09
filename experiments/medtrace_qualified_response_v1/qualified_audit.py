"""Post-run aggregate review; never changes the preregistered decision."""
import json
import os
from pathlib import Path
import statistics

r = Path(os.environ['RUN_ROOT'])
read = lambda p: json.loads(p.read_text())
data = [read(p) for p in sorted((r/'private/results').glob('*.json'))]
x = read(r/'public/RESULTS.json')
assert len(data) == 8 and x['status'] == 'COMPLETE'
assert not list((r/'private').glob('FAILURE*'))
assert sum(d['forward_calls'] for d in data) == 4552
assert sum(d['backward_calls'] for d in data) == 504
assert sum(d['zero_expert_Base_checks'] for d in data) == 808
primary = [q for d in data for q in d['diagnostics'] if q['role']=='HELDOUT_FIT' and q['semantic_correct']]
assert len(primary) == 504 and len({q['group'] for q in primary}) == 29
arms = ('RAW','PROJECTED','MATCHED_RAW')
result = dict(status='PASS', decision_unchanged=x['decision'], posthoc_descriptive_only=True,
    primary_observations=504, primary_queries=63, primary_sources=29,
    primary_baseline_all_argmax_correct=sum(q['baseline']['all_argmax_correct'] for q in primary),
    primary_baseline_correct_tokens=sum(q['baseline']['correct_tokens'] for q in primary),
    primary_tokens=sum(q['baseline']['tokens'] for q in primary),
    primary_content_and_EOS={part:{a:dict(
        KL=statistics.mean(q['candidates'][a][part]['KL'] for q in primary),
        lost_correct_tokens=sum(q['candidates'][a][part]['lost_correct_tokens'] for q in primary))
        for a in arms} for part in ('content','EOS')},
    KL_relative_reduction=1-x['protection']['projected_KL']/x['protection']['matched_KL'],
    edit_progress_retention=x['edit_NLL_progress']['PROJECTED']/x['edit_NLL_progress']['MATCHED_RAW'],
    experts_with_lower_KL=sum(e['heldout_KL']['PROJECTED']<e['heldout_KL']['MATCHED_RAW'] for e in x['per_expert']),
    constraint_ranks=[d['geometry']['rank'] for d in data],
    maximum_matching_relative_error=max(d['function_matching']['relative_error'] for d in data),
    normalized_linear_response_ratios=[d['geometry']['normalized_predicted_after']/d['geometry']['normalized_predicted_before'] for d in data],
    actual_linear_response_ratios=[d['geometry']['actual_predicted_norm']/d['geometry']['raw_predicted_norm'] for d in data],
    sessions_ended=all(s.get('ended_epoch') for s in read(r/'RESOURCE_LEDGER.json')['gpu_sessions']),
    public_raw_inputs=False, clinical_correctness='ASTRA_AGAINST_EXISTING_ANNOTATIONS_ONLY',
    heldout_previously_exposed=True, independent_confirmation=False)
assert result['sessions_ended'] and all(d['state_restored_exact'] and d['baseline_repeat_exact'] for d in data)
(r/'public/REVIEW_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
