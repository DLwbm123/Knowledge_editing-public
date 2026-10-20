"""Read-only source/role/relation audit. No model imports, candidate generation or sampling by scores."""
import inspect
import json
from pathlib import Path
import subprocess


def remote_audit(context):
    import json
    import re
    from collections import Counter, defaultdict
    from pathlib import Path

    reads = []

    def read(path):
        reads.append(str(path))
        return json.loads(path.read_text())

    config = read(Path(context['config']))
    tasks = read(Path(context['tasks']))['tasks']
    ledger = read(Path(context['ledger'])); queries = ledger['queries']
    cal = read(Path(context['calibration_manifest']))
    old_panel = read(Path(context['old_panel']))['rows']
    roles = read(Path(context['roles']))
    v5 = read(Path(context['v5']))
    pool = read(Path(context['pool']))
    old_single = read(Path(context['old_single']))
    role_index = defaultdict(set)
    for row in roles:
        role_index[row['dataset'], str(row['image_id']).lower()].update(row['roles'])
    def image_id(path):
        ids = set(re.findall(r'xmlab\d+|synpic\d+', str(path).lower()))
        return next(iter(ids)) if len(ids) == 1 else None
    used_v5 = {(r['dataset'], image_id(r['image_path'])) for r in v5}
    known_pool = {(r['dataset'], str(r['image_id']).lower()) for r in pool}
    data = Path(config['source_image_root'])/'SLAKE'
    train = read(data/'train.json')
    excluded = set(config['calibration_manifest']['edit_and_eval_source_groups'])
    excluded |= set(config['calibration_manifest']['calibration_source_groups'])
    excluded |= set(cal['calibration_source_groups'])
    for name in ['validation.json', 'test.json']:
        excluded |= {'SLAKE:'+r['img_name'].split('/')[0] for r in read(data/name)}
    first = []; seen = set()
    for row in sorted(train, key=lambda x: int(x['qid'])):
        group = 'SLAKE:'+row['img_name'].split('/')[0]
        if row['q_lang'] != 'en' or group in excluded or group in seen:
            continue
        seen.add(group)
        if len(first) < 8:
            first.append(row)
    rows = []
    for i, row in enumerate(first):
        ident = image_id(row['img_name']); key = ('SLAKE', ident)
        group = 'SLAKE:'+row['img_name'].split('/')[0]
        same = [x for x in train if x['q_lang'] == 'en' and x['img_name'].split('/')[0] == row['img_name'].split('/')[0]]
        bindings = [t for t in tasks if t['native']['source_group'] == group]
        # Do not manufacture task relations from answer equality, templates, or an image similarity score.
        rows.append(dict(alias='N'+str(i+1).zfill(2), image_available=(data/'imgs'/row['img_name']).is_file(),
            source_annotation_nonempty=bool(row['question'].strip() and str(row['answer']).strip()),
            same_group_English_annotations=len(same), role_known=bool(role_index.get(key)),
            historical_roles=sorted(role_index.get(key, set())), used_in_v5=key in used_v5,
            conflicting_roles=sorted(role_index.get(key, set()) & {
                'PROTECTED_SPLIT', 'FORMAL_OR_RESERVED_SOURCE', 'CAL_CHECK_SUPPORT_RESERVED', 'FROZEN_FIT_SOURCE'}),
            in_previous_screened_pool=key in known_pool,
            registered_queue_source=len(bindings), registered_source_T1G_T2G_relations=sum(
                len(e['all_probe_query_ids']) for t in bindings for e in t['events'] if e['task'] in ['T1G','T2G']),
            source_Base_correct='UNKNOWN_not_measured', multiple_correct_compliant_endpoints='UNKNOWN_not_generated',
            qualified_cross_image_generalization='NOT_CERTIFIED', qualified_near_boundary='NOT_CERTIFIED'))
    task_roles = {}
    for role in ['T1G', 'T2G', 'T1L', 'T2L']:
        values = []
        for task in tasks:
            ids = {q for e in task['events'] if e['task'] == role for q in e['all_probe_query_ids']}
            values.append(dict(entries=len(ids), cross_group=sum(queries[q]['source_group'] != task['native']['source_group'] for q in ids),
                different_image_binding=sum(queries[q]['image_sha256'] != task['native']['image_sha256'] for q in ids)))
        task_roles[role] = dict(tasks_with_two_entries=sum(r['entries'] >= 2 for r in values),
            tasks_with_two_different_images=sum(r['different_image_binding'] >= 2 for r in values),
            tasks_with_two_cross_groups=sum(r['cross_group'] >= 2 for r in values))
    forbidden_eval_groups = set(cal['calibration_source_groups']) | {t['native']['source_group'] for t in tasks[:3]}
    forbidden_eval_groups |= {r['query']['source_group'] for r in old_panel[:6]}
    forbidden_eval_groups |= {'SLAKE:'+r['img_name'].split('/')[0] for r in first}
    ordered_eval = []; eval_groups = set()
    for qid, q in sorted(queries.items()):
        if q['source_group'] in forbidden_eval_groups or q['source_group'] in eval_groups:
            continue
        eval_groups.add(q['source_group']); ordered_eval.append(qid)
    # Existing ledger Base flags are provenance, not a fresh native Base qualification.
    result = dict(status='SUPPORT_GAP_NO_BRIDGE_ADMISSION', model_calls=0, Judge=0, new_labels=0,
        source_selection='qid_ascending_first8_after_initial_group_exclusions_then_role_audit_no_score_selection',
        raw_remaining_train_groups=len(seen), preliminary_sources=rows, preliminary_count=len(first),
        fixed_source_role_conflicts=sum(bool(r['conflicting_roles']) for r in rows),
        complete_relation_certified_source_packages=0,
        historical_queue=dict(tasks=len(tasks), unique_source_groups=len({t['native']['source_group'] for t in tasks}),
            old_single_status=old_single['status'], old_single_committed=old_single['committed'],
            task_label_not_cross_image_certificate=True, role_binding_counts=task_roles),
        disjoint_ledger_unrelated_pool_groups=len(eval_groups), proposed_first16_unrelated_candidates=min(16, len(ordered_eval)),
        proposed_first16_historical_Base_true=sum(ledger['Base_correctness'].get(q) is True for q in ordered_eval[:16]),
        current_Base_correct_support='UNKNOWN_no_new_model_or_Judge',
        new_source_near_boundary_support='NOT_CERTIFIED', new_source_T2G_support='NOT_CERTIFIED',
        new_source_T1G_support='NOT_CERTIFIED_no_new_rephrase_labels', patient_independence='UNKNOWN',
        prior_screened_pool=dict(rows=len(pool), image_keys=len(known_pool)),
        checks=dict(calibration32=len(cal['calibration_source_groups']) == 32,
            fixed_first8=len(first) == 8, all_first8_readable=all(r['image_available'] for r in rows),
            no_score_based_source_filter=True, no_easy_source_replacement=True,
            no_relation_inference_from_label_names=True),
        missing=['source-role/exposure clearance for all new sources',
                 'certified source-specific untrained paraphrase/cross-image relationships',
                 'certified non-propagation near-boundary relationships',
                 'current native Base-correct evaluation support',
                 'at least four sources with two distinct correct and original-gate-compliant endpoints'],
        manifest_reads=len(reads), GPU_protocol_frozen=False, bridge_started=False)
    assert len(reads) <= 16
    assert all(result['checks'].values())
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    context = json.loads((root/'.local_run/bridge_support_context.json').read_text())
    code = inspect.getsource(remote_audit)+'\nprint(json.dumps(remote_audit(json.loads(sys.stdin.readline()))))'
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'pro5000',
        "python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(context)+'\n', capture_output=True, text=True, timeout=60)
    if response.returncode:
        raise RuntimeError(response.stderr)
    result = json.loads(response.stdout)
    (root/'reports/vigor_distributionBridge_review_v1_20261020/SUPPORT_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
