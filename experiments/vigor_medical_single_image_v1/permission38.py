"""Bounded provenance tracing only; existing role labels are never permissions to execute."""
import inspect
import json
import subprocess
import time
from pathlib import Path


def trace(context, targets):
    import json
    import re
    import signal
    from collections import Counter, defaultdict
    from pathlib import Path
    def expired(*_):
        raise TimeoutError('Permission trace time bound')
    signal.signal(signal.SIGALRM, expired); signal.alarm(50)
    reads = []; consumed = 0
    def read(path, alias):
        nonlocal consumed
        path = Path(path); size = path.stat().st_size
        if len(reads) >= 31 or consumed+size >= 128*1024**2:
            raise RuntimeError('Frozen input bound')
        consumed += size; reads.append(dict(alias=alias, bytes=size))
        return json.loads(path.read_text())
    receipts = read(context['receipts'], 'EXCLUSION_RECEIPTS')
    ledger = read(context['roles'], 'ORIGINAL_ROLE_LEDGER')
    ledger_roles = defaultdict(set)
    for row in ledger:
        ledger_roles[row['dataset'], str(row['image_id']).lower()].update(row['roles'])
    taskfiles = [(f'F{i+1:03d}', r) for i, r in enumerate(receipts['files']) if Path(r['path']).name.startswith('TASKS')]
    if len(taskfiles) > 16:
        raise RuntimeError('TASKS bound; no truncated source attribution')
    id_to_alias = {}
    for target in targets:
        dataset, identity = target['source_group'].split(':', 1)
        if dataset != 'SLAKE' or set(target['roles']) != ledger_roles[dataset, identity.lower()]:
            raise ValueError('Role or identity binding drift')
        id_to_alias[identity.lower()] = target['alias']
    def ids(value):
        found = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {'img_name', 'img_id'} or any(s in key.lower() for s in ['image', 'source_group', 'source_id', 'case_id', 'patient_id', 'study_id']):
                    found.update(x.lower() for x in re.findall(r'xmlab\d+|synpic\d+', str(item), re.I))
                if isinstance(item, (dict, list)):
                    found.update(ids(item))
        elif isinstance(value, list):
            for item in value:
                found.update(ids(item))
        return found
    matches = defaultdict(list); schemas = []; docpaths = []
    fit = {'native', 'U_fit', 'U_new', 'U_expanded', 'H_fit'}
    formal = {'evaluation', 'official_evaluation_full', 'probes'}
    def walk(value, file_alias, parent_native=None):
        if isinstance(value, dict):
            native = ids(value['native']) if 'native' in value else parent_native
            for key, item in value.items():
                if key in fit | formal:
                    for identity in ids(item) & set(id_to_alias):
                        matches[id_to_alias[identity]].append(dict(file=file_alias, field=key,
                            assigned_role='FROZEN_FIT_SOURCE' if key in fit else 'FORMAL_OR_RESERVED_SOURCE',
                            same_image_as_enclosing_native=identity in native if native is not None else None,
                            field_membership_not_execution=True))
                elif isinstance(item, (dict, list)):
                    walk(item, file_alias, native)
        elif isinstance(value, list):
            for item in value:
                walk(item, file_alias, parent_native)
    for file_alias, record in taskfiles:
        path = Path(record['path'])
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError('Frozen task file unavailable or changed size')
        value = read(path, file_alias); walk(value, file_alias)
        schemas.append(dict(file=file_alias, basename=path.name, top_level_keys=list(value) if isinstance(value, dict) else 'list',
                            membership_matches=sum(sum(x['file'] == file_alias for x in rows) for rows in matches.values())))
        for candidate in [path.parent/'FROZEN_PROTOCOL.json', path.parent.parent/'RUN_MANIFEST.json']:
            if candidate.is_file() and str(candidate) not in docpaths:
                docpaths.append(str(candidate))
    docs = []
    for i, path in enumerate(docpaths[:12], 1):
        value = read(path, f'D{i:02d}')
        def flags(node, prefix=''):
            result = []
            if isinstance(node, dict):
                for key, val in node.items():
                    if any(word in key.lower() for word in ['authoriz', 'permission', 'training', 'formal_edit', 'allow']):
                        if isinstance(val, bool):result.append(dict(key=prefix+key, value=val))
                        elif isinstance(val, dict):result.extend(flags(val, prefix+key+'.'))
            return result
        docs.append(dict(file=f'D{i:02d}', basename=Path(path).name,
                         metadata_keys=list(value) if isinstance(value, dict) else 'list',
                         explicit_boolean_flags=flags(value), current_38_source_authorization=False,
                         interpretation='historical metadata presence is not current source-specific permission'))
    result = []
    for target in targets:
        rows = matches[target['alias']]; fields = sorted({x['field'] for x in rows})
        has_fit = 'FROZEN_FIT_SOURCE' in target['roles']
        if has_fit and not any(x['assigned_role'] == 'FROZEN_FIT_SOURCE' for x in rows):
            raise ValueError('Fit role not recovered from original fields')
        if not has_fit and rows:
            raise ValueError('Unexpected task membership for inventory-only identity')
        result.append(dict(alias=target['alias'], original_roles=target['roles'], original_field_matches=rows,
            original_field_classes=fields, fit_role_attribution_resolved=has_fit,
            inventory_only_no_task_membership=not has_fit,
            direct_native_field_found='native' in fields,
            derived_lineage='UNKNOWN_fields_do_not_certify_full_derivation',
            prior_training_role_evidence='TASK_FIELD_MEMBERSHIP' if has_fit else 'NONE_IN_REGISTERED_TASKS',
            current_edit_source_permission='UNKNOWN_NO_AFFIRMATIVE_SOURCE_SPECIFIC_RECORD',
            independent_confirmation_permission='NOT_ESTABLISHED',
            execution_history='UNKNOWN_ROLE_MEMBERSHIP_NOT_EXECUTION',
            source_medical_relationships='NOT_CERTIFIED', role_changed=False))
    assert len(result) == 38
    signal.alarm(0)
    return dict(status='PROVENANCE_TRACED_CURRENT_PERMISSION_NOT_ESTABLISHED', rows=result,
        fit_attribution_resolved=sum(x['fit_role_attribution_resolved'] for x in result),
        inventory_only=sum(x['inventory_only_no_task_membership'] for x in result),
        native_field_sources=sum(x['direct_native_field_found'] for x in result),
        field_source_counts=dict(Counter(field for x in result for field in x['original_field_classes'])),
        task_files=schemas, historical_document_evidence=docs, document_candidates_not_read=max(0, len(docpaths)-12),
        read_ledger=reads, total_bytes=consumed, current_permission_proven=0,
        registry_presence_not_execution=True, historical_permission_not_current_permission=True,
        model=0, GPU=0, Judge=0, role_mutations=0)


def main():
    root = Path(__file__).resolve().parents[2]; out = root/'reports/vigor_permission38_v1_20261020'
    frozen = json.loads((out/'FREEZE.json').read_text())
    if time.time()-frozen['epoch'] >= frozen['whole_seconds']:
        raise TimeoutError('Original audit clock expired')
    if (out/'PERMISSION_TRACE.json').exists():
        raise FileExistsError('No repeated provenance scan')
    census = json.loads((root/'reports/vigor_legalSupport201_v1_20261020/ROLE_CENSUS.json').read_text())['rows']
    bindings = {x['alias']: x for x in json.loads((root/'.local_run/legalSupport201_private_binding.json').read_text())}
    targets = [dict(alias=x['alias'], source_group=bindings[x['alias']]['source_group'], roles=x['historical_roles'])
               for x in census if x['category'] == 'PERMISSION_OR_PROVENANCE_UNKNOWN']
    assert len(targets) == 38 and sum('FROZEN_FIT_SOURCE' in x['roles'] for x in targets) == 25
    context = json.loads((root/'.local_run/permission38_context.json').read_text())
    code = inspect.getsource(trace)+'\np=json.loads(sys.stdin.readline());print(json.dumps(trace(p["context"],p["targets"])))'
    run = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'pro5000', "python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
        input=json.dumps(code)+'\n'+json.dumps(dict(context=context, targets=targets))+'\n', capture_output=True, text=True, timeout=60)
    if run.returncode:
        (root/'.local_run/permission38_failure.txt').write_text(run.stderr)
        raise RuntimeError('Provenance tracing failed; preserve private receipt, no retry')
    value = json.loads(run.stdout); (out/'PERMISSION_TRACE.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: value[k] for k in ['status', 'fit_attribution_resolved', 'inventory_only', 'native_field_sources',
        'field_source_counts', 'document_candidates_not_read', 'total_bytes', 'current_permission_proven']}))


if __name__ == '__main__':
    main()
