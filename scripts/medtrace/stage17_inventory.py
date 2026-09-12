"""Reproduce a metadata-only handoff; never open formal QA, labels, or images."""
import argparse
import json
from pathlib import Path


def summarize(snapshot):
    data = {k: v['content'] for k, v in snapshot.items()}
    cohort = data['cohort']
    tasks = {}
    for task, row in cohort['tasks'].items():
        tasks[task] = dict(historical_candidate_edits=row['candidate_edit_count'],
            historical_eligible_edits=row['eligible_edit_count'],
            historical_eligible_probes=row['eligible_probe_count'],
            manifest_unique_images=row['unique_image_count'],
            image_count_domain='as reported by historical task builder; not eligible source-group count',
            eligible_source_groups=None, stage17_supported_edits=None, stage17_supported_probes=None)
    measured = set().union(*(data[k]['event_ids'] for k in ('dev', 'qual8', 'qual16')))
    return dict(status='METADATA_HANDOFF_COMPLETE__FORMAL_COHORT_AND_SUPPORT_NOT_FROZEN',
        historical=dict(candidate_occurrences=tasks['T0']['historical_candidate_edits'], eligible_T0=cohort['final_t0_n'],
            removed_Base_correct=cohort['tasks']['T0']['candidate_edit_count']-cohort['final_t0_n'],
            unique_native_images=tasks['T0']['manifest_unique_images'],
            duplicate_underlying_facts=None, duplicate_occurrences=None,
            previous_diagnostic_200_and_amended189_not_adopted=True,
            new_Judge_may_change_membership=True),
        actual_stage17_N=None, actual_stage17_S_H=None, actual_stage17_S_G=None,
        tasks=tasks,
        exposure=dict(DEV16_selected=16, QUAL8_selected=8, QUAL16_selected=16,
            SEQ16_selected_but_historical_run_not_started=16,
            distinct_events_in_DEV16_QUAL8_QUAL16_manifests=len(measured),
            DEV16_QUAL16_overlap=len(set(data['dev']['event_ids']) & set(data['qual16']['event_ids'])),
            source_level_development_seen=None, source_level_evaluation_seen=None,
            source_level_not_development_seen=None,
            reason='event selections are not a complete source exposure ledger; no fresh claim'),
        support_masks={k: dict(membership=None, count=None, status='PENDING_GLOBAL_SOURCE_ROLE_JOIN')
            for k in ('E_U', 'E_H', 'E_HG', 'E_HG_eval')},
        missing_roles={
            'U_fit': 'existing train-only QA may be reused; entire formal source-group exclusion join and own-edit teacher binding missing',
            'H_fit': 'same verified proposition, conflicting original answer, different image and out-of-scope evidence; global source exclusion join missing',
            'G': 'different proposition QA matched to H slot, excludes U and evaluation sources; Stage14 selected supports are candidates, not Stage17 proof',
            'G_positive': 'legal in-scope positive relation/source proof; never substitute ordinary G',
            'H_eval': 'frozen untrained relation labels and source independence from H-fit; official task support requires access authority',
            'U_eval': 'independent evaluation QA/source binding; reused U-fit cannot count as repeated independent evidence'},
        T5=dict(status='NA_NOT_AUTHORIZED', sealed_or_PadChest_access=False),
        compatibility=dict(old_raw='CONDITIONAL_EXACT_FULL_INPUT_RUNTIME_DECODE_MATCH_ONLY',
            old_verdicts='INCOMPATIBLE_WITH_NEW_JUDGE_PROTOCOL',
            old_checkpoints='NOT_REUSABLE_BY_METHOD_NAME; require own native, fit, support, init, seed and training binding; no future information',
            old_RC='HISTORICAL_DIAGNOSTIC_ONLY'),
        new_training=0, new_Judge_calls=0, data_downloads=0,
        raw_formal_inputs_read_by_this_inventory=0,
        reason_N_unknown='new formal-data/Judge authorization and pre-student Base masks not yet available; 179 is historical, not a freshly eligible cohort')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    value = summarize(json.loads(args.snapshot.read_text()))
    with args.out.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
