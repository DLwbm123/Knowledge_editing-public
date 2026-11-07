"""Recompute frozen T4 results and audit selected inputs without model/Judge calls.

Private inputs are collected from the completed run. Public output contains only
counts; the immutable method results and accepted labels are never rewritten.
Question-form strata are descriptive heuristics, not clinical labels or a new
selection rule.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text())


def question_form(question):
    q = question.strip().casefold().removeprefix('question: ').strip()
    if q.startswith('where') or 'what part' in q:
        return 'location'
    if 'color' in q or 'colour' in q:
        return 'color'
    if re.match(r'^(is|are|does|do|has|have|can)\b', q):
        return 'presence'
    if 'disease' in q or 'abnormality' in q:
        return 'diagnosis'
    return 'other'


def audit(worktree):
    w = Path(worktree)
    private = w / '.private/med_t4_core_v1_20261107'
    material = read(private / 'audit_20261107/INPUTS_AND_SOURCE.private.json')
    data = material['TASK_INPUTS.private.json']
    tasks = {t['edit_id']: t for t in data['tasks']}
    by_owner = {t['order']: t for t in data['tasks']}
    roles = {e['edit_query_id']: e['task'] for e in data['events']}
    qualification = read(private / 'qualification/BASE_MASK_FROZEN.json')
    base = {b['query_id']: b for b in read(private / 'qualification/BASE_BINDINGS.private.json')['bindings'].values()}
    outputs = read(private / 'GENERATION_OUTPUTS.private.json')
    packet = read(private / 'evaluation/PACKET_PRIVATE.json')
    decisions = read(private / 'evaluation/DECISIONS.json')
    consumers = {c['consumer_id']: c for c in packet['consumers']}
    receipts = read(private / 'audit_20261107/STATE_RECEIPTS.private.json')
    source_metadata = read(private / 'audit_20261107/SELECTED_COMPOSITION_METADATA.private.json')
    published = read(w / 'reports/med_t4_core_v1_20261107/RESULTS.json')
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        assert condition, name

    check('all_outputs_and_consumers_unique', len(outputs) == len(consumers) == 944
          and len({x['consumer_id'] for x in outputs}) == 944)
    check('complete_known_scoring', not decisions['unknown'])
    qualified = [e for e in qualification['events'] if e['status'] == 'BASE_ELIGIBLE']
    check('qualified_edit_set_unchanged', set(tasks) == {e['edit_query_id'] for e in qualified})
    for e in qualified:
        check('Base_native_and_probe_contract', e['base_native_core'] is False and all(
            qualification['query_core'][q] is (e['task'] == 'T4L') for q in e['eligible_probe_query_ids']))
    plans = {(p['edit_id'], p['role'], p['query']['query_id']): p['query'] for p in data['plan']}
    check('no_duplicate_plan', len(plans) == len(data['plan']) == 472)
    state_hashes = receipts['state_hashes']
    for x in outputs:
        query = plans[(x['edit_id'], x['role'], x['query_id'])]
        check('plan_to_output_binding', all(query[k] == x['judge_input'][k]
              for k in ('question', 'reference', 'image_sha256')))
        if x['role'] != 'FIT':
            check('Base_to_method_binding', all(base[x['query_id']][k] == x['judge_input'][k]
                  for k in ('question', 'reference', 'image_sha256', 'prompt_ids', 'attention_mask')))
        chosen = x['effective_expert']
        check('effective_state_binding', x['state_hash'] == (state_hashes[chosen] if chosen else 'UNEDITED_BASE'))
        check('router_activation_binding', bool(chosen) == x['route']['activated'])
        if x['mode'] == 'single' and chosen:
            check('single_expert_identity', chosen == x['edit_id'])
        c = consumers[x['consumer_id']]
        check('scoring_consumer_binding', all(c[k] == x[k] for k in ('mode', 'owner', 'role', 'query_id')))
        payload = packet['payloads'][c['key']]
        check('judge_visible_payload_binding', payload['question'] == x['judge_input']['question']
              and payload['gold_answer'] == x['judge_input']['reference']
              and payload['raw_base_answer'] == x['answer']['raw_answer'])
        if x['role'] in ('T4G', 'T4L') and not chosen:
            check('OFF_output_tokens_match_frozen_Base', x['answer']['raw_token_ids']
                  == base[x['query_id']]['output']['raw_generated_token_ids'])
    for entry in material['TRAIN_SOURCE_BOUND_PRIVATE.json']:
        task, rows = entry['task'], entry['source_rows']
        check('training_prior_only_five_views', len(rows) == 5 and [q['audit_role'] for q in rows] == ['NATIVE'] + ['FIT'] * 4
              and [q['question'] for q in rows] == [task['native']['question']] + task['fit_questions']
              and all(q['image_sha256'] == task['native']['image_sha256']
                      and q['reference'] == task['native']['reference'] for q in rows))
    tokens = material['TOKEN_LOCK.private.json']['records']
    check('all_training_token_bound', {x['order'] for x in tokens if x['status'] == 'TOKEN_BOUND'} == set(by_owner))
    check('training_completed_without_failure', all(s['TRAIN_WORKER_EXIT.json']['code'] == 0
          and s['TRAIN_COMPLETE.json']['status'] == 'T4_C_MED_MDU_SHARD_COMPLETE' for s in receipts['shards'].values()))
    check('fixed_update_total', sum(s['TRAIN_COMPLETE.json']['counts']['updates'] for s in receipts['shards'].values()) == 480 * 67)
    pair_metadata = {(r['single_image'], r['multi_image']): r for r in source_metadata}
    check('composition_labels', all(r['single_lesion'].strip().casefold() in
          [s.strip().casefold() for s in r['multi_lesions'].split(',')]
          and len(r['multi_lesions'].split(',')) >= 2 for r in source_metadata))
    input_refs = defaultdict(set)
    for x in outputs:
        input_refs[(x['judge_input']['image_sha256'], x['judge_input']['question'])].add(x['judge_input']['reference'])
    check('no_identical_input_reference_conflict', all(len(v) == 1 for v in input_refs.values()))
    panels, detail = {}, {}
    for mode in ('single', 'bank'):
        panels[mode] = {}
        for role in ('T0', 'FIT', 'T4G', 'T4L'):
            rows = [x for x in outputs if x['mode'] == mode and x['role'] == role]
            groups = defaultdict(list)
            for x in rows:
                groups[x['owner']].append(int(decisions['decisions'][consumers[x['consumer_id']]['key']]['core_correct']))
            exact = sum((Fraction(sum(v), len(v)) for v in groups.values()), Fraction()) / len(groups)
            check('reported_edit_macro_reproduced', abs(100 * float(exact) - published['panels'][mode][role]['lower_pct']) < 1e-10)
            panels[mode][role] = dict(edits=len(groups), events=len(rows), correct_events=sum(map(sum, groups.values())),
                edit_macro_pct=100 * float(exact), edit_macro_fraction=str(exact),
                event_micro_pct=100 * sum(map(sum, groups.values())) / len(rows))
        rows = [x for x in outputs if x['mode'] == mode and x['role'] == 'T4G']
        strata = defaultdict(lambda: dict(events=0, correct=0))
        chosen_roles = Counter()
        for x in rows:
            native = by_owner[x['owner']]['native']
            pair = (native['source_group'].split(':', 1)[1], plans[(x['edit_id'], x['role'], x['query_id'])]['source_group'].split(':', 1)[1])
            check('all_selected_pairs_in_original_composition_metadata', pair in pair_metadata)
            check('T4G_distinct_source_and_probe_images', native['image_sha256'] != x['judge_input']['image_sha256'])
            key = question_form(native['question']) + '_to_' + question_form(x['judge_input']['question'])
            strata[key]['events'] += 1
            strata[key]['correct'] += decisions['decisions'][consumers[x['consumer_id']]['key']]['core_correct']
            if x['effective_expert']:
                chosen_roles[roles[x['effective_expert']]] += 1
        detail[mode] = dict(events=len(rows), route_ON=sum(x['route']['activated'] for x in rows),
            own_expert=sum(x['effective_expert'] == x['edit_id'] for x in rows),
            other_expert=sum(x['effective_expert'] not in (None, x['edit_id']) for x in rows),
            selected_expert_origin=dict(chosen_roles), at_cap=sum(x['at_cap'] for x in rows),
            unique_probe_queries=len({x['query_id'] for x in rows}),
            unique_probe_images=len({x['judge_input']['image_sha256'] for x in rows}),
            unique_anchor_images=len({by_owner[x['owner']]['native']['image_sha256'] for x in rows}),
            question_form_strata=dict(sorted(strata.items())),
            changed_reference_events=sum(by_owner[x['owner']]['native']['reference'].strip().casefold()
                != x['judge_input']['reference'].strip().casefold() for x in rows),
            selected_expert_native_same_probe_image=sum(
                x['effective_expert'] is not None and tasks[x['effective_expert']]['native']['image_sha256']
                == x['judge_input']['image_sha256'] for x in rows))
    return dict(status='COMPLETED_READ_ONLY_AUDIT', checks=checks, recomputed_panels=panels, T4G_diagnostics=detail,
        original_source_pairs=len(pair_metadata), original_source_target_lesions=dict(Counter(r['single_lesion'] for r in source_metadata)),
        model_calls=0, Judge_calls=0, new_hashes=0, changed_labels=0, changed_method_results=False,
        diagnosis='No score aggregation/input/state-binding defect found. Mixed question attributes and joint T4G/T4L R0 bank confound a narrow interpretation as pure compositional transfer.',
        limitations=['Question-form strata are descriptive string heuristics, not post-hoc cohort eligibility.',
            'Other-expert routing is an observation of the frozen R0 rule, not proof that forcing the owner would improve answers.',
            'Composition metadata is source annotation, not independent clinical verification.',
            'The original negative result remains unchanged; no tuning or new claim of superiority.'])


if __name__ == '__main__':
    import os
    result = audit(os.environ['AUDIT_WORKTREE'])
    out = Path(os.environ['AUDIT_WORKTREE']) / 'reports/med_t4_core_v1_20261107/T4G_AUDIT.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), all_passed=all(result['checks'].values()))))
