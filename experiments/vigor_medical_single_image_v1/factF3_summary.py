"""Offline audit of frozen F results and completed, cache-aware blind scores."""
import json, math
from pathlib import Path


def main():
    report = Path(__file__).resolve().parents[2] / 'reports/vigor_factF3_v1_20261020'
    out = report / 'completion'
    result = json.loads((out / 'RESULT.json').read_text())
    ended = json.loads((out / 'ENDED.json').read_text())
    scores = json.loads((out / 'SCORES.json').read_text())
    checks = []

    def check(field, condition):
        checks.append({'field': field, 'passed': bool(condition)})

    check('terminal', result['status'] == ended['status'] == scores['status'] == 'COMPLETE')
    for x in result['checks']:
        check('runtime.' + x['field'], x['passed'] if 'passed' in x else math.isfinite(x['absolute_difference']) and x['absolute_difference'] <= x['tolerance'])
    for k, v in result['restoration'].items():
        check('restore.' + k, v is True)
    rows = []
    baseline = {'B': [False, False, True], 'N': [False, False, True]}
    for ci, case in enumerate(result['cases']):
        for arm in case['arms']:
            name = arm['name'].replace('_REBUILD', '')
            key = case['sample'] + '_' + name
            check(key + '.budget', arm['editing_forwards'] <= 512)
            check(key + '.restoration', arm['base_restored'] is True)
            accepted = 0
            for step in arm['steps']:
                if name == 'F':
                    selected = [t for t in step['trials'] if t['status'] == 'ACCEPTED']
                    check(key + '.unique_accept.' + str(step['step']), len(selected) == int(step['accepted']))
                    if not selected:
                        continue
                    trial = selected[0]
                    accepted += 1
                    check(key + '.full_prediction', trial['linear'][0] < -1e-5)
                    check(key + '.weighted_prediction', trial['weighted_linear'] < -1e-5)
                    for stage in ['actual_response', 'final_response']:
                        response = trial[stage]
                        check(key + '.' + stage, response['passed'] and response['NLL_gain'] > 1e-5 and response['weighted_gain'] > 1e-5 and response['rho_NLL'] >= .1 and response['rho_weighted'] >= .1)
                    gate = trial['final_gate']
                    fm = trial['final_margin']
                    check(key + '.final_full_native_binding', abs(fm['mean_nll'] - gate['native_nll'][0]) <= 1e-5)
                    check(key + '.final_weighted_identity', abs(fm['weighted_nll'] - .5*(fm['mean_nll'] + fm['fact_nll'])) <= 1e-6)
                else:
                    accepted += 1
                    gate = step['final_gate']
                check(key + '.original_gates', gate['accepted'] and all(gate['gates'].values()))
                check(key + '.original_null', abs(gate['cumulative_null_change']) <= .01)
                check(key + '.original_trust', gate['combined_native_fisher_cost'] <= .001)
            check(key + '.accepted_count', accepted == arm['accepted_steps'])
            events = [e for e in result['evaluation'] if e['sample'] == case['sample'] and e['arm'] == arm['name']]
            check(key + '.panel_count', len(events) == 6)
            values = [scores['scores'].get(e['alias']) for e in events]
            damage = sum(v is False for v in values)
            missing = sum(v is None for v in values)
            t0 = scores['scores'].get(case['sample'] + '_F_T0') if name == 'F' else baseline[name][ci]
            last = arm['last_accepted']
            rows.append(dict(case=case['sample'], arm=name, semantic_T0=t0, semantic_origin='new_F_output_cached_or_once_blind' if name == 'F' else 'previous_BC3_N3_exact_endpoint_binding_no_new_T0', damage=damage, locality_denominator=6, locality_missing=missing, damaged_panel=[e['panel_id'] for e,v in zip(events,values) if v is False], accepted_steps=arm['accepted_steps'], editing_forwards=arm['editing_forwards'], backward_calls=arm['backward_calls'], secant_calls=arm['secant_calls'], native_NLL=last['native_nll'][0], fact_NLL=last['margin']['fact_nll'], weighted_NLL=last['margin']['weighted_nll'], margin=last['margin']['margin'], original_base_null_drift=last['native_nll'][1]-case['base']['native_nll'][1], endpoint_F0_cost=last['cumulative_F0_cost'], original_gates_all_accepted=True, per_arm_seconds=None))
    summary = {}
    for name in ['B', 'N', 'F']:
        rr = [x for x in rows if x['arm'] == name]
        summary[name] = dict(semantic_correct=sum(x['semantic_T0'] is True for x in rr), semantic_denominator=3, semantic_missing=sum(x['semantic_T0'] is None for x in rr), damage=sum(x['damage'] for x in rr), locality_denominator=18, locality_missing=sum(x['locality_missing'] for x in rr), near_support=0, unrelated_distinct_queries=6, editing_forwards=sum(x['editing_forwards'] for x in rr), backward_calls=sum(x['backward_calls'] for x in rr), secant_calls=sum(x['secant_calls'] for x in rr), costs_are_reconstruction=name != 'F')
    edit = sum(x['editing_forwards'] for x in rows)
    locality_fw = sum(e['generation_forwards'] for e in result['evaluation'])
    t0_fw = sum(a['generation_forwards_total'] for c in result['cases'] for a in c['arms'] if a['name'] == 'F')
    costs = dict(main_gpu_seconds=ended['resident_seconds'], main_forwards=ended['model_forward_calls'], F_editing=summary['F']['editing_forwards'], B_reconstruction=summary['B']['editing_forwards'], N_reconstruction=summary['N']['editing_forwards'], locality_generation=locality_fw, F_T0_generation=t0_fw, control_restore=ended['model_forward_calls']-edit-locality_fw-t0_fw, preparation_failure_seconds=21.052013397216797, preparation_failure_forwards=4, Base_qualification_seconds=38.401888, Base_qualification_forwards=451, total_gpu_seconds=ended['resident_seconds']+21.052013397216797+38.401888, total_forwards=ended['model_forward_calls']+4+451, post_new_payloads=scores['new_payloads'], peak_allocated_bytes=ended['peak_allocated_bytes'], checkpoint_count=0)
    # New payload count is a unique-input count, not an endpoint count.
    costs['post_cache_endpoint_count'] = sum(v != 'new_blind_score' for v in scores['provenance'].values())
    costs['post_newly_scored_endpoint_count'] = sum(v == 'new_blind_score' for v in scores['provenance'].values())
    check('score_count', len(scores['scores']) == 57 and not scores['missing'])
    check('generation_count', len(result['evaluation']) == 54)
    check('forward_accounting', costs['control_restore'] >= 0 and costs['total_forwards'] == 5981)
    check('main_budget', ended['resident_seconds'] < 7200 and ended['model_forward_calls'] <= 20000)
    audit = dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL', checks=checks, check_count=len(checks), failed=[x for x in checks if not x['passed']], runtime_check_count=len(result['checks']))
    analysis = dict(rows=rows, summary=summary, original_B_algorithm_editing_forwards=661, original_N_algorithm_editing_forwards=1350, original_B_backward=237, original_N_backward=515, costs=costs, near_locality='NA: zero support', scope='Three exposed development edits; six shared unrelated Base-correct questions; no T1G/T2G or near-boundary evaluation', decision='Stop fixed 0.5/0.5 weighting branch: F loses E03 and adds no semantic repairs; do not tune weights')
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    print(json.dumps(dict(audit=audit['status'],checks=len(checks),failed=audit['failed'],summary=summary,costs=costs),indent=2))
    if audit['status'] != 'PASS':
        raise SystemExit('Independent audit failed; do not declare delivery complete')


if __name__ == '__main__':
    main()
