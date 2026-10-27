"""Freeze patient-disjoint source panels without model outputs or GPU use."""
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import time
import zipfile


def main():
    root = Path(os.environ['TASK_ROOT'])
    output = root / '.private/capability_source_v102_20261024'
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'WINDOW.json').exists(), 'Source phase already started'
    started = time.time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Source phase hard stop')))
    signal.alarm(3600)
    (output / 'WINDOW.json').write_text(json.dumps(dict(original_start=started, hard_end=started+3600, local_bytes_limit=256000000, gpu_calls=0), indent=2))
    source = Path(json.loads((root / '.private/input_signal_v101_20261024/CONFIG.json').read_text())['source_root'])
    spec = importlib.util.spec_from_file_location('support', root / 'experiments/protection_admission_v98/source.py')
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    previous = json.loads((root / '.private/relation_source_v97_20261024/CANDIDATES_FROZEN.json').read_text())
    excluded = {r['patient'] for r in previous}
    for row in json.loads((root / '.private/protection_admission_v98_20261024/CANDIDATES_FROZEN.json').read_text()):
        excluded.add(row['patient'])
        if row['positive_control']:
            excluded.add(row['positive_control']['patient'])
    assert len(excluded) == 60
    with zipfile.ZipFile(source / 'metadata/master_table.csv.zip') as z:
        table = list(csv.DictReader(io.TextIOWrapper(z.open('master_table.csv'))))
    splits = {}
    for row in table:
        splits.setdefault(row['PatientID'], set()).add(row['split'])
    assert all(len(s) == 1 for s in splits.values()), 'Patient split overlap'
    meta = {r['ImageID']: r for r in table if r['split'] == 'train'}
    records = sorted((r for r in json.loads((source / 'metadata/grounded_reports_20240819.json').read_text()) if r['ImageID'] in meta and meta[r['ImageID']]['PatientID'] not in excluded), key=lambda r: (meta[r['ImageID']]['PatientID'], str(r['StudyID']), r['ImageID']))
    temporal = re.compile(r'known|stable|unchanged|previous|prior|increas|decreas|persist|compar|resolv|improv|follow.?up', re.I)
    pools = {}
    for label in ['nodule', 'infiltrates']:
        for polarity in ['positive', 'negative']:
            pool = []
            for record in records:
                if polarity == 'positive':
                    matches = [f for f in record['findings'] if support.positive(f) and f['labels'] == [label] and not f.get('progression') and not temporal.search(f['sentence_en'])]
                    finding = matches[0] if matches else None
                else:
                    finding = support.negatives(record).get(label)
                if finding is not None:
                    pool.append(dict(patient=meta[record['ImageID']]['PatientID'], study=record['StudyID'], image=record['ImageID'], label=label, polarity=polarity, finding=finding, source_record=record))
            pools[label, polarity] = pool
    selected, used, counts = [], set(excluded), {}
    for panel in ['D', 'C']:
        for label in ['nodule', 'infiltrates']:
            for polarity in ['positive', 'negative']:
                key = f'{panel}_{label}_{polarity}'
                counts[key] = 0
                for row in pools[label, polarity]:
                    if row['patient'] in used:
                        continue
                    used.add(row['patient'])
                    counts[key] += 1
                    selected.append(dict(row, panel=panel, case=f'{panel}{len(selected)+1:02d}'))
                    if counts[key] == 4:
                        break
    (output / 'CANDIDATES_FROZEN.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    audit = dict(status='FROZEN_PENDING_WHOLE_SOURCE_REVIEW', original_start=started, selection_end=time.time(), excluded_prior_patients=len(excluded), selected_patients=len(selected), stratum_counts=counts, patient_overlap=0, model_calls=0, image_extractions=0, replacement=False)
    (output / 'SOURCE_AUDIT.json').write_text(json.dumps(audit, indent=2))
    assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) < 256000000
    print(json.dumps(audit))
    for row in selected:
        print(json.dumps(dict(case=row['case'], panel=row['panel'], label=row['label'], polarity=row['polarity'], selected_sentence=row['finding']['sentence_en'], whole_report=[dict(sentence=f['sentence_en'], abnormal=f['abnormal'], labels=f.get('labels', []), progression=f.get('progression')) for f in row['source_record']['findings']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
