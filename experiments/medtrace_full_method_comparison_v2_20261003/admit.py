"""CPU-only admission of the unchanged146 queue and source-qualified available H."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(os.environ['RUN_ROOT'])
OLD = Path('/data/bmw/Knowledge_editing/outputs/medtrace-full-method-comparison-20261003/run')
SOURCE = Path('/data/bmw/Knowledge_editing/outputs/scope-text-guard-20261001/run/source')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n'); tmp.replace(p)


def relocate(p):
    p = str(p)
    if '/derived_inputs/' in p:
        return str(ROOT/'private/derived_inputs'/p.split('/derived_inputs/',1)[1])
    marker = '/DataP/'
    if marker in p:
        p = '/data/bmw/DataP/' + p.split(marker, 1)[1]
    return p


def main():
    started = time.time()
    private = ROOT / 'private'
    if (private / 'CPU_ADMISSION.json').exists():
        raise FileExistsError('Existing admission is not automatically rerun')
    ledger = read(OLD / 'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    frozen = ledger['freeze_id']
    digest = hashlib.sha256(json.dumps({k:v for k,v in ledger.items() if k != 'freeze_id'}, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    # Original digest implementation is checked after the source assembly below.
    lookup = {t['edit_id']:t for t in ledger['tasks']}
    tasks = [lookup[e] for e in ledger['main_T0']]
    assert len(tasks) == len(set(ledger['main_T0'])) == 146
    h = read(OLD / 'private/H_SUPPORT_FINAL/SOURCE_VERIFIED_H_FIT.json')
    banned = set(map(tuple, read(Path('/data/bmw/Knowledge_editing/outputs/h-completion-v2-20261003/run/private/GLOBAL_SOURCE_EXCLUSION.json'))['banned_images']))
    assert len(h) == 12 and len({r['edit_id'] for r in h}) == 8
    H = {}
    for row in h:
        assert row['admitted_H_fit'] is True and row['source_binding'] == 'PASS'
        assert row['edit_id'] in ledger['main_T0']
        s = row['source']; assert (s['dataset'], str(s['image_id'])) not in banned
        assert Path(relocate(s['image_path'])).is_file()
        # The approved relation carries its original source answer, never a new model answer.
        H.setdefault(row['edit_id'], []).append(dict(question=lookup[row['edit_id']]['native']['question'], reference=s['answer'], image_path=relocate(s['image_path']), source_group=str(s['dataset']) + ':' + str(s['image_id']), evidence=row))
    paths = set()
    for t in tasks:
        assert t['fit_status'] == 'SUPPORTED' and len(t['fit_questions']) == 4
        assert t['roles']['U_fit'] and t['U_fit']
        paths.add(t['native']['image_path'])
        for u in t['U_fit']: paths.add(u['image_path'])
        for e in t['events']:
            for q in e['all_probe_query_ids']: paths.add(ledger['queries'][q]['image_path'])
    resolved = {p:relocate(p) for p in paths}
    missing = [p for p,q in resolved.items() if not Path(q).is_file()]
    assert not missing, 'Bound image assets missing: ' + str(len(missing))
    assert shutil.disk_usage(ROOT).free > 72 * 1024**3
    for name in ['COHORT_AND_SUPPORT_LEDGER.json','BINDINGS.json','ROLE_BOUNDARY_JOIN.json','SOURCE_OVERLAY.json','JUDGE_LOCK.json','VERDICTS_ASTRA.jsonl']:
        dest = private / 'legacy_stage17' / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(OLD/'private/legacy_stage17'/name, dest)
    write(private / 'H_AVAILABLE.json', H)
    write(private / 'IMAGE_RELOCATION.json', resolved)
    execution = private / 'source'
    execution.mkdir(exist_ok=True)
    for folder in ['scripts', 'methods', 'm3bench_repro']:
        shutil.copytree(SOURCE / folder, execution / folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc'), dirs_exist_ok=True)
    shutil.copytree(SOURCE/'third_party/LLaVA-Med', execution/'third_party/LLaVA-Med', ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'), dirs_exist_ok=True)
    # Operational argv repair: keep the original official Git checks, pass the path via cwd.
    gate=execution/'scripts/editor_paperspec_formal.py'
    text=gate.read_text(); old='["git", "-C", str(source), *arguments], text=True, stderr=subprocess.STDOUT'
    assert old in text
    gate.write_text(text.replace(old,'["git", *arguments], cwd=source, text=True, stderr=subprocess.STDOUT'))
    import sys
    sys.path.insert(0, str(execution))
    from scripts.medtrace.stage17_prepare import digest as frozen_digest
    assert frozen_digest({k:v for k,v in ledger.items() if k != 'freeze_id'}) == frozen
    write(private/'SOURCE_ASSEMBLY.json', dict(source=str(SOURCE), copied=['scripts','methods','m3bench_repro','third_party/LLaVA-Med'], old_sources_read_only=True, new_execution_source=str(execution)))
    write(private/'CPU_ADMISSION.json', dict(status='PASS', N=146, original_freeze=frozen, original_order_preserved=True, U_covered=146, H_covered=8, H_relations=12, pending_H_used=0, new_H=0, original_U_source_not_new_independent_CAL=True, U_original_source_future594_overlap=True, H_whole_future594_exclusion_pass=True, resolved_image_assets=len(resolved), image_file_checks='existence only; actual tensor/prompt binding required at GPU', H_eval=None, patient_independence='UNKNOWN', seconds=time.time()-started, GPU_jobs=0, Judge=0))
    write(ROOT/'public/PROGRESS.json', dict(status='CPU_ADMISSION_PASS_RUNTIME_PENDING', N=146, H_covered=8, H_relations=12, next='actual GPU mechanical parity before any formal training'))
    print('CPU_ADMISSION_PASS', 146, 8, 12, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        write(ROOT/'private/CPU_ADMISSION_FAILURE.json', dict(error=repr(exc),traceback=traceback.format_exc(),GPU_jobs=0))
        raise
