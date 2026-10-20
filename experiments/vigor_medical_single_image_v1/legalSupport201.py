"""Frozen metadata census only: no model imports, scores, role mutation or training."""
import inspect
import json
import subprocess
import time
from pathlib import Path


def classify(roles, historical_native=False, identity_known=True):
    protected = sorted(set(roles) & {'PROTECTED_SPLIT', 'CAL_CHECK_SUPPORT_RESERVED', 'FORMAL_OR_RESERVED_SOURCE'})
    if protected:
        return 'RESERVED_NO_TRAINING', protected
    if not identity_known or not roles:
        return 'PERMISSION_OR_PROVENANCE_UNKNOWN', ['MISSING_IDENTITY_OR_ROLE']
    if historical_native:
        return 'TRAINING_ALLOWED_EXPOSED_DEVELOPMENT', ['EXISTING_NATIVE_TRAINING_BINDING']
    return 'PERMISSION_OR_PROVENANCE_UNKNOWN', ['NO_EXPLICIT_TRAINING_CLEARANCE']


def remote_census(context):
    import json
    import re
    import signal
    from collections import Counter, defaultdict
    from pathlib import Path

    def limit(*_):
        raise TimeoutError('Metadata hard time limit')
    signal.signal(signal.SIGALRM, limit); signal.alarm(50)
    reads = []
    def read(key):
        path = Path(context[key]); reads.append(key)
        return json.loads(path.read_text())
    config = read('config'); tasks = read('tasks')['tasks']; queries = read('ledger')['queries']
    calibration = read('calibration_manifest'); roles = read('roles'); v5 = read('v5')
    pool = read('pool'); old_panel = read('old_panel')['rows']; old_single = read('old_single')
    data = Path(config['source_image_root'])/'SLAKE'
    annotations = {}
    for key, name in [('train', 'train.json'), ('validation', 'validation.json'), ('test', 'test.json')]:
        reads.append(key); annotations[key] = json.loads((data/name).read_text())
    excluded = set(config['calibration_manifest']['edit_and_eval_source_groups'])
    excluded |= set(config['calibration_manifest']['calibration_source_groups'])
    excluded |= set(calibration['calibration_source_groups'])
    excluded |= {'SLAKE:'+r['img_name'].split('/')[0] for k in ['validation', 'test'] for r in annotations[k]}
    first = {}; grouped = defaultdict(list)
    for row in sorted(annotations['train'], key=lambda r: int(r['qid'])):
        group = 'SLAKE:'+row['img_name'].split('/')[0]
        if row['q_lang'] != 'en' or group in excluded:
            continue
        first.setdefault(group, row); grouped[group].append(row)
    if len(first) != 201:
        raise ValueError('Frozen population changed; no substitution')
    def identity(value):
        ids = set(re.findall(r'xmlab\d+|synpic\d+', str(value).lower()))
        return next(iter(ids)) if len(ids) == 1 else None
    role_index = defaultdict(set)
    for row in roles:
        role_index[row['dataset'], str(row['image_id']).lower()].update(row['roles'])
    training = {t['native']['source_group'] for t in tasks}
    old_queries = {q['source_group'] for q in queries.values()}
    v5_images = {(r['dataset'], identity(r['image_path'])) for r in v5}
    pool_images = {(r['dataset'], str(r['image_id']).lower()) for r in pool}
    rows = []; relation_rows = []; private = []
    for index, (group, row) in enumerate(first.items(), 1):
        alias = f'P{index:03d}'; ident = identity(row['img_name']); key = ('SLAKE', ident)
        rr = sorted(role_index.get(key, set())); category, reasons = classify(rr, group in training, ident is not None)
        readable = (data/'imgs'/row['img_name']).is_file()
        bindings = [t for t in tasks if t['native']['source_group'] == group]
        relation_counts = {role: sum(len(e['all_probe_query_ids']) for t in bindings for e in t['events'] if e['task'] == role)
                           for role in ['T1G', 'T2G', 'T1L', 'T2L']}
        evidence = dict(original_image_file=readable, original_annotation=bool(row['question'].strip() and str(row['answer']).strip()),
                        image_content_review='NOT_PERFORMED', independent_medical_review='UNKNOWN')
        rows.append(dict(alias=alias, category=category, eligibility_reasons=reasons, historical_roles=rr,
            training_permission='NO' if category == 'RESERVED_NO_TRAINING' else 'EXISTING_DEVELOPMENT_ONLY' if category == 'TRAINING_ALLOWED_EXPOSED_DEVELOPMENT' else 'UNKNOWN',
            role_evidence='IDENTITY_ROLE_LEDGER_exact_dataset_image_id' if rr else 'MISSING_ROLE_RECORD',
            identity_key_unambiguous=ident is not None, evidence_available=evidence,
            English_annotation_count=len(grouped[group]), parent_group='SAME_OFFICIAL_IMAGE_FOLDER',
            derived_input_lineage='UNKNOWN_no_complete_lineage_manifest', patient_independence='UNKNOWN',
            exposure_history=dict(existing_native_training=group in training, existing_query_inventory=group in old_queries,
                V5_roles=key in v5_images, previously_screened_pool=key in pool_images,
                inventory_role_present=bool(set(rr) & {'BASE_QUERY_INVENTORY', 'BENCHMARK_METADATA', 'POOL_INVENTORY'}),
                all_historical_method_exposure_clearance='UNKNOWN'),
            selected_for_new_training=False))
        relation_rows.append(dict(alias=alias, source_role_admissible=category.startswith('TRAINING_ALLOWED'),
            original_annotation_present=evidence['original_annotation'], source_edit_object='NOT_CERTIFIED_FOR_NEW_BRIDGE',
            registered_relation_counts=relation_counts, same_group_annotation_count=len(grouped[group]),
            paraphrase_propagation_evidence='NOT_CERTIFIED', cross_image_propagation_evidence='NOT_CERTIFIED',
            near_boundary_nonpropagation_evidence='NOT_CERTIFIED', unrelated_panel_per_source_evidence='NOT_CERTIFIED',
            source_eval_image_links='NO_REGISTERED_SOURCE_BINDING' if not bindings else 'SEE_LEGACY_LINKS_NOT_MEDICAL_CERTIFICATE',
            evidence_review_disposition='NO_NEW_LABELS_FOR_RESERVED_OR_UNCLEARED_SOURCE'))
        private.append(dict(alias=alias, source_group=group, original_qid=row['qid'], source_image=str(data/'imgs'/row['img_name'])))
    links = []
    for i, task in enumerate(tasks, 1):
        source = task['native']
        for role in ['T1G', 'T2G', 'T1L', 'T2L']:
            ids = sorted({q for e in task['events'] if e['task'] == role for q in e['all_probe_query_ids']})
            vals = [queries[q] for q in ids]
            links.append(dict(legacy_source=f'L{i:03d}', role=role, entries=len(vals),
                eval_different_from_source_image=sum(q['image_sha256'] != source['image_sha256'] for q in vals),
                eval_different_from_source_group=sum(q['source_group'] != source['source_group'] for q in vals),
                eval_images_unique_among_themselves=len({q['image_sha256'] for q in vals}),
                eval_groups_unique_among_themselves=len({q['source_group'] for q in vals}),
                proposition_propagation_or_nonpropagation='UNKNOWN_NOT_CERTIFIED_BY_BINDING'))
    link_summary = {}
    for role in ['T1G', 'T2G', 'T1L', 'T2L']:
        rr = [r for r in links if r['role'] == role]
        link_summary[role] = {field: dict(Counter('0' if r[field] == 0 else '1' if r[field] == 1 else '2plus' for r in rr))
                             for field in ['eval_different_from_source_image', 'eval_different_from_source_group', 'eval_images_unique_among_themselves']}
    category_counts = {k: sum(r['category'] == k for r in rows) for k in [
        'TRAINING_ALLOWED_UNEXPOSED', 'TRAINING_ALLOWED_EXPOSED_DEVELOPMENT', 'RESERVED_NO_TRAINING', 'PERMISSION_OR_PROVENANCE_UNKNOWN']}
    assert len(rows) == len(relation_rows) == 201 and sum(category_counts.values()) == 201 and len(reads) == 12
    signal.alarm(0)
    return dict(role_census=dict(population=201, category_counts=category_counts, role_counts=dict(Counter(role for row in rows for role in row['historical_roles'])),
        rows=rows, scores_used_for_selection=False, role_mutations=0, new_sources_selected=0, reads=reads,
        model=0, GPU=0, Judge=0, original_pool_count=201, permissions_not_inferred_from_train_split=True),
        relation_pack=dict(population_rows=relation_rows, legacy_source_eval_links=links, legacy_summary=link_summary,
            new_medical_labels=0, image_bindings_not_patient_or_proposition_certificates=True,
            old_task_names_and_scores_unchanged=True), private_bindings=private)


def main():
    root = Path(__file__).resolve().parents[2]; out = root/'reports/vigor_legalSupport201_v1_20261020'
    freeze = json.loads((out/'FREEZE.json').read_text())
    if time.time() > freeze['frozen_epoch']+freeze['whole_seconds']:
        raise TimeoutError('Original preparation clock expired')
    if (out/'ROLE_CENSUS.json').exists():
        raise FileExistsError('Census exists; no repeated remote scan')
    context = json.loads((root/'.local_run/bridge_support_context.json').read_text())
    code = inspect.getsource(classify)+'\n'+inspect.getsource(remote_census)+'\nprint(json.dumps(remote_census(json.loads(sys.stdin.readline()))))'
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'pro5000',
        "python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(context)+'\n', capture_output=True, text=True, timeout=60)
    if response.returncode:
        (root/'.local_run/legalSupport201_failure.txt').write_text(response.stderr)
        raise RuntimeError('Read-only census failed; retained local private error')
    value = json.loads(response.stdout)
    (root/'.local_run/legalSupport201_private_binding.json').write_text(json.dumps(value.pop('private_bindings'), indent=2)+'\n')
    for key, filename in [('role_census', 'ROLE_CENSUS.json'), ('relation_pack', 'RELATION_PACK.json')]:
        (out/filename).write_text(json.dumps(value[key], indent=2)+'\n')
    print(json.dumps(dict(population=value['role_census']['population'], categories=value['role_census']['category_counts'],
        roles=value['role_census']['role_counts'], legacy_links=value['relation_pack']['legacy_summary'], model=0, GPU=0, Judge=0)))


if __name__ == '__main__':
    main()
