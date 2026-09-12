"""Score-independent role join on existing manifests; no model or Judge calls."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage13r_sources import canonical
from scripts.medtrace.stage17_prepare import group, lines


def identity_closure(groups, identities):
    """Close existing image digests and canonical source identities to a fixed point."""
    result = set(groups)
    by_hash = defaultdict(set)
    for row in identities:
        if row.get('image_sha256'):
            by_hash[row['image_sha256']].add(group(row))
    while True:
        expanded = result | set().union(*(g for g in by_hash.values() if g & result), set())
        if expanded == result:
            return result
        result = expanded


def filter_events(events, identities, excluded):
    lookup = {r['query_id']: r for r in identities}
    kept, removed = [], []
    for event in events:
        if group(lookup[event['edit_query_id']]) in excluded:
            removed.append(dict(event_id=event['event_id'], reason='PRIOR_PROJECT_EVALUATION_SOURCE'))
        else:
            kept.append(event)
    return kept, removed


def run(config):
    outputs = Path(config['storage_root'])/'Knowledge_editing/outputs'
    formal = Path(config['run_root'])/'formal'
    private_output = formal/'private/ROLE_BOUNDARY_JOIN.json'
    public_output = formal/'public/ROLE_BOUNDARY_JOIN.json'
    if private_output.exists() or public_output.exists():
        raise FileExistsError('Preserve the existing score-independent role join')
    pool = read(formal/'private/CANDIDATE_POOL.json')
    # Project identities only; no need to reopen heldout files or inspect any verdict.
    identity_fields = ('query_id','dataset','image_path','image_sha256')
    catalog = outputs/'m3bench_data_runtime_finalization_v3/20260904T014138Z/data_static/STATIC_QUERY_INVENTORY.jsonl'
    identities = [{k:r[k] for k in identity_fields} for r in lines(catalog)]
    blocked = {tuple(g) for g in pool['preserved_evaluation_native_exclusions']}
    sources, exposed = [], set()
    for stage in ('medtrace_stage2_20260908_r02','medtrace_stage5_20260909_r01'):
        path = outputs/stage/'private/NEW_EPISODE_MANIFEST_PRIVATE.json'
        rows = [r for episode in read(path)['episodes'] for r in episode['rows']]
        evaluation = [r for r in rows if r['role'] not in ('native','fit')]
        groups = {group(r) for r in evaluation}
        blocked |= groups; exposed |= {group(r) for r in rows}
        sources.append(dict(stage=stage, path=str(path), role_counts=dict(Counter(r['role'] for r in rows)),
            evaluation_source_groups=len(groups)))
    path = outputs/'medtrace_stage14_20260911_r01/private/SUPPORT_OVERLAY.json'
    groups = {canonical(dataset,image) for dataset,image in read(path)['all_cohort_evaluation_images']}
    blocked |= groups; exposed |= groups
    sources.append(dict(stage='Stage14',path=str(path),evaluation_source_groups=len(groups)))
    blocked = identity_closure(blocked,identities)
    forbidden = identity_closure({tuple(g) for g in pool['forbidden_source_groups']},identities)
    kept, removed = filter_events(pool['events'],identities,blocked | forbidden)
    required = {e['edit_query_id'] for e in kept} | {q for e in kept for q in e['all_probe_query_ids']}
    lookup = {r['query_id']:r for r in identities}
    if not required <= set(pool['required_query_ids']):
        raise ValueError('Role correction must not expand the frozen Base input pool')
    if any(group(lookup[q]) in forbidden for q in required):
        raise ValueError('A preserved reservation overlaps the remaining input pool')
    whole_queue = identity_closure({group(lookup[q]) for q in required},identities)
    overlay_path = outputs/'medtrace_stage13r_20260911_r01/private/SOURCE_OVERLAY.json'
    overlay = read(overlay_path)
    train_rows = [r for r in overlay['rows'] if r['source_role']=='train'
        and overlay['image_roles'][r['source_group']]=='adaptation']
    support_rows = [r for r in train_rows if group(r) not in blocked | forbidden | whole_queue]
    tiers = Counter('previous_project_exposure' if group(lookup[e['edit_query_id']]) in exposed
        else 'unresolved_exposure_not_claimed_unseen' for e in kept)
    summary = dict(status='ROLE_JOIN_COMPLETE_BASE_AND_SUPPORT_RELATIONS_PENDING',
        base_verdicts_read=0, student_outputs_read=0, new_model_calls=0,
        initial_events=dict(Counter(e['task'] for e in pool['events'])),
        retained_events_before_Base_filter=dict(Counter(e['task'] for e in kept)),
        excluded_events=len(removed), protected_project_evaluation_groups=len(blocked),
        protected_reserved_groups=len(forbidden), required_Base_queries=len(required),
        unused_Base_candidate_queries=len(set(pool['required_query_ids'])-required),
        exposure_event_counts=dict(tiers), confirmed_unseen_claim=False,
        reviewed_source_overlay_rows=len(overlay['rows']), overlay_train_rows=len(train_rows),
        support_candidates_isolated_from_current_structural_superset=len(support_rows),
        support_candidate_image_groups=len({group(r) for r in support_rows}),
        support_masks_frozen=False, actual_N=None, GPU_training_started=False,
        H_U_G_relation_review='PENDING; source isolation is not proof of a valid H/U/G proposition',
        future_queue_must_apply_this_filter=True, old_candidate_and_Judge_files_modified=False,
        existing_identity_hashes_reused=True, new_image_hashes=0,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_new(private_output,dict(summary,sources=sources,retained_events=kept,excluded_events=removed,
        evaluation_source_groups=sorted(blocked),reserved_source_groups=sorted(forbidden),
        required_query_ids=sorted(required),support_overlay_source=str(overlay_path),
        support_candidate_source_ids=[dict(dataset=r['dataset'],source_qid=r['source_qid'],
            source_group=r['source_group']) for r in support_rows]))
    write_new(public_output,summary)
    import json
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('config',type=Path)
    run(read(parser.parse_args().config))
