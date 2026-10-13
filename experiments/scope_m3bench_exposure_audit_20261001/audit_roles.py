"""Identity-only role attribution; registry presence is not execution evidence."""
import collections
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get('RUN_ROOT', '/tmp'))
SOURCE = Path(os.environ.get('SOURCE_ROOT', '/tmp'))
PRIORITY = ['FROZEN_FIT_SOURCE', 'FORMAL_OR_RESERVED_SOURCE', 'CAL_CHECK_SUPPORT_RESERVED', 'PROTECTED_SPLIT', 'BASE_QUERY_INVENTORY', 'POOL_INVENTORY', 'BENCHMARK_METADATA', 'UNATTRIBUTED']


def ids(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {'img_name', 'img_id'} or any(k in key.lower() for k in ['image', 'source_group', 'source_id', 'case_id', 'patient_id', 'study_id']):
                found.update(x.lower() for x in re.findall(r'xmlab\d+|synpic\d+', str(item), re.I))
            if isinstance(item, (list, dict)):
                found.update(ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(ids(item))
    return found


def task_roles(value, groups):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ['native', 'U_fit', 'U_new', 'U_expanded', 'H_fit']:
                groups['FROZEN_FIT_SOURCE'].update(ids(item))
            elif key in ['evaluation', 'official_evaluation_full', 'probes']:
                groups['FORMAL_OR_RESERVED_SOURCE'].update(ids(item))
            elif isinstance(item, (dict, list)):
                task_roles(item, groups)
    elif isinstance(value, list):
        for item in value:
            task_roles(item, groups)


def selfcheck():
    groups = collections.defaultdict(set)
    task_roles({'tasks': [{'native': {'image_path': 'xmlab1/source.jpg'}, 'evaluation': [{'image_id': 'xmlab2'}], 'U_fit': [{'image_id': 'synpic3.jpg'}]}]}, groups)
    assert groups['FROZEN_FIT_SOURCE'] == {'xmlab1', 'synpic3'}
    assert groups['FORMAL_OR_RESERVED_SOURCE'] == {'xmlab2'}
    assert ids({'question': 'xmlab99', 'image_id': 'xmlab2'}) == {'xmlab2'}
    assert ids({'img_name': 'xmlab7/source.jpg', 'img_id': 7}) == {'xmlab7'}


def write(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    started = time.time()
    selfcheck()
    manifest = json.loads((ROOT / 'RUN_MANIFEST.json').read_text())
    argv = Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not any(k in argv.lower() for k in ['wangbomin', 'knowledge_editing', 'scope'])
    write('PROCESS_RECEIPT.json', dict(pid=os.getpid(), argv=argv, started_epoch=started, CUDA_VISIBLE_DEVICES=''))
    receipts = json.loads((SOURCE / 'private/EXCLUSION_RECEIPTS.json').read_text())
    rows = json.loads((SOURCE / 'private/SOURCE_QA_ROLE_AUDIT.json').read_text())
    recovery = json.loads((ROOT / 'SCHEMA_RECOVERY.json').read_text()) if (ROOT / 'SCHEMA_RECOVERY.json').exists() else {}
    prior_cpu = recovery.get('prior_CPU_seconds', 0)
    groups = collections.defaultdict(set)
    consumed = recovery.get('prior_input_bytes', 0) + (SOURCE / 'private/EXCLUSION_RECEIPTS.json').stat().st_size + (SOURCE / 'private/SOURCE_QA_ROLE_AUDIT.json').stat().st_size
    files = []
    for record in receipts['files']:
        p = Path(record['path'])
        assert p.stat().st_size == record['bytes'], 'Source manifest changed since prior audit'
        name = p.name
        if name.startswith('TASKS'):
            role = 'TASK_FIELDS'
        elif name in ['validation.json', 'test.json']:
            role = 'PROTECTED_SPLIT'
        elif name.endswith('.csv'):
            role = 'BENCHMARK_METADATA'
        elif 'FORMAL_RECORDS' in name or name in ['PRESSURE_VALIDATION_FROZEN.json', 'PRESSURE_TEST_FROZEN.json', 'LOCALITY_STRESS_HOLDOUT.json']:
            role = 'FORMAL_OR_RESERVED_SOURCE'
        elif any(k in name for k in ['CAL_', 'CHECK_', 'U_bg', 'G_SUPPORTS', 'SCOPE_ROLE']):
            role = 'CAL_CHECK_SUPPORT_RESERVED'
        elif any(k in name for k in ['BASE_QUERY_INVENTORY', 'BASE_QUERY_LINEAGE']):
            role = 'BASE_QUERY_INVENTORY'
        else:
            role = 'POOL_INVENTORY'
        if role == 'BENCHMARK_METADATA':
            import csv
            with p.open(encoding='utf-8-sig') as f:
                value = list(csv.DictReader(f))
        elif p.suffix == '.jsonl':
            value = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        else:
            value = json.loads(p.read_text())
        consumed += p.stat().st_size
        assert consumed < 256 * 1024**2 and time.time() < manifest['deadline_epoch'] and time.time() - started + prior_cpu < 60 and not (ROOT / 'STOP').exists()
        if role == 'TASK_FIELDS':
            task_roles(value, groups)
        else:
            groups[role].update(ids(value))
        files.append(dict(name=name, role=role))
    ledger = []
    assert groups['PROTECTED_SPLIT'], 'SLAKE protected source identities were not parsed'
    for row in rows:
        flags = [k for k in PRIORITY if row['image_id'] in groups[k]]
        ledger.append(dict(dataset=row['dataset'], image_id=row['image_id'], qid=row['qid'], roles=flags, primary=flags[0] if flags else 'UNATTRIBUTED'))
    assert len(ledger) == 7167
    public = dict(status='COMPLETE_NOT_DATA_ADMITTED', source_QA_rows=len(ledger), primary_QA_counts=dict(collections.Counter(x['primary'] for x in ledger)), role_QA_counts={k: sum(k in x['roles'] for x in ledger) for k in PRIORITY}, role_source_image_counts={k: len(groups[k]) for k in PRIORITY}, combinations=dict(collections.Counter('|'.join(x['roles']) or 'UNATTRIBUTED' for x in ledger)), per_dataset={d: dict(collections.Counter(x['primary'] for x in ledger if x['dataset'] == d)) for d in ['SLAKE', 'VQA-RAD']}, registry_presence_not_execution=True, source_image_overlap_not_patient_independence=True, no_labels_or_QA_mining=True, candidate_only_not_automatically_admitted=True, new_verified_CAL=0, GPU_hours=0, new_Judge=0, independent_CONFIRM=False)
    write('private/IDENTITY_ROLE_LEDGER.json', ledger)
    write('public/ROLE_AGGREGATES.json', public)
    write('public/FINAL_EXECUTION_AUDIT.json', dict(status='COMPLETE', denominator_binding='PASS', role_selfcheck='PASS', protected_img_name_schema='PASS', CPU_seconds=time.time()-started, prior_CPU_seconds=prior_cpu, cumulative_CPU_seconds=time.time()-started+prior_cpu, input_bytes=consumed, GPU_hours=0, new_Judge=0, new_verified_CAL=0, first_clock_preserved=True))
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 16*1024**2
    write('RUN_STATUS.json', dict(status='COMPLETE', phase='CLOSED_WAITING_VERIFIED_SCOPE', epoch=time.time()))
    print(json.dumps(public))


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        selfcheck()
        print('PASS: fit/formal source separation and question-ID exclusion')
    else:
        try:
            main()
        except Exception as e:
            write('FAILURE.json', dict(error=str(e), traceback=traceback.format_exc()))
            write('RUN_STATUS.json', dict(status='FAILED_PRESERVED'))
            raise
