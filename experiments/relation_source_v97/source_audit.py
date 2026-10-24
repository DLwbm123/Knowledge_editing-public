"""Bounded TRAIN-only source admission; no model answers enter selection."""
import bisect
import csv
import io
import json
from pathlib import Path
import re
import sys
import zipfile


class Volumes(io.RawIOBase):
    """Seek over existing split archive without making a 38 GB joined copy."""
    def __init__(self, paths):
        self.paths = paths
        self.ends = [0]
        for p in paths:
            self.ends.append(self.ends[-1] + p.stat().st_size)
        self.pos = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.ends[-1] if whence == 2 else 0)
        return self.pos

    def read(self, size=-1):
        size = self.ends[-1] - self.pos if size < 0 else size
        blocks = []
        while size and self.pos < self.ends[-1]:
            i = bisect.bisect_right(self.ends, self.pos) - 1
            n = min(size, self.ends[i + 1] - self.pos)
            with self.paths[i].open('rb') as f:
                f.seek(self.pos - self.ends[i])
                blocks.append(f.read(n))
            self.pos += n
            size -= n
        return b''.join(blocks)


def side(finding):
    loc = ' '.join(finding.get('locations', [])).lower()
    sentence = finding['sentence_en'].lower()
    # Require report and location agreement, no inferred absence or box-derived diagnosis.
    if ('bilateral' in loc or ('left' in loc and 'right' in loc)) and re.search(r'\bbilateral\b|\bboth\b', sentence):
        return 'bilateral'
    for value, other in [('left', 'right'), ('right', 'left')]:
        if value in loc and other not in loc and 'bilateral' not in loc and re.search(r'\b'+value+r'\b', sentence) and not re.search(r'\b'+other+r'\b|\bbilateral\b', sentence):
            return value


def main(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source/'metadata/master_table.csv.zip') as z:
        table = list(csv.DictReader(io.TextIOWrapper(z.open('master_table.csv'))))
    patients, images = {}, {}
    for row in table:
        patients.setdefault(row['PatientID'], set()).add(row['split'])
        images.setdefault(row['ImageID'], row)
    assert all(len(v) == 1 for v in patients.values()), 'Patient split overlap'
    train = {k: v for k, v in images.items() if v['split'] == 'train'}
    # JSON must be parsed, but only TRAIN source facts are used or displayed.
    records = [r for r in json.loads((source/'metadata/grounded_reports_20240819.json').read_text()) if r['ImageID'] in train]
    archive = zipfile.ZipFile(Volumes(sorted((source/'Padchest_GR_files').glob('PadChest_GR.zip.*'))))
    members = {Path(x.filename).name: x for x in archive.infolist() if not x.is_dir()}
    candidates = {k: [] for k in ['left', 'right', 'bilateral']}
    for r in sorted(records, key=lambda x: (train[x['ImageID']]['PatientID'], x['StudyID'], x['ImageID'])):
        positives = [f for f in r['findings'] if f['abnormal'] and f.get('boxes') and len(f.get('labels', [])) == 1 and not re.search(r'suggest|probab|possib|suspic|cannot|may |could ', f['sentence_en'], re.I)]
        for f in positives:
            laterality = side(f)
            if not laterality:
                continue
            other = next((g for g in positives if g['labels'] != f['labels']), None)
            member = members.get(r['ImageID'])
            if other is None or member is None:
                continue
            candidates[laterality].append(dict(image=r['ImageID'], patient=train[r['ImageID']]['PatientID'], study=r['StudyID'], side=laterality, target=f, other=other, archive_member=member.filename))
            break
    selected, used = [], set()
    for side_name in candidates:
        for row in candidates[side_name]:
            if row['patient'] not in used:
                selected.append(row)
                used.add(row['patient'])
            if sum(r['side'] == side_name for r in selected) == 8:
                break
    audit = dict(train_images=len(train), total_images=len(images), patients=len(patients), cross_split_patients=0,
                 archive_members=len(members), candidate_counts={k:len(v) for k,v in candidates.items()}, selected_counts={k:sum(r['side']==k for r in selected) for k in candidates}, model_answers_seen=0)
    (output/'SOURCE_AUDIT.json').write_text(json.dumps(audit, indent=2))
    assert len(selected) == 24 and all(v == 8 for v in audit['selected_counts'].values()), audit
    (output/'CANDIDATES_FROZEN.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    # Extract only the admitted fixed cohort; archive CRC checks remain built in.
    from PIL import Image
    rows = []
    (output/'images').mkdir(exist_ok=True)
    for n, row in enumerate(selected):
        group = f'p{n+1:02d}'
        data = archive.read(row['archive_member'])
        im = Image.open(io.BytesIO(data))
        im.load()
        # Match published archive pixels; PNG retains original integer intensities.
        path = output/'images'/f'{group}.png'
        path.write_bytes(data)
        label, other_label = row['target']['labels'][0], row['other']['labels'][0]
        questions = [
            ('target', f'On which side is the {label} in this chest X-ray: left, right, or bilateral?', row['side']),
            ('protect_fit', f'Is {label} present in this chest X-ray?', 'Yes'),
            ('protect_holdout', f'Is {other_label} present in this chest X-ray?', 'Yes'),
            ('brief_control', f'On which side is the {label} in this chest X-ray: left, right, or bilateral? Answer briefly with only the side.', row['side'])]
        for role, question, ref in questions:
            rows.append(dict(qid=f'{group}_{role}', source_group=group, role=role, question=question, reference=ref, image_path=str(path.resolve()), source='PadChest-GR official TRAIN', source_evidence=row['other'] if role=='protect_holdout' else row['target']))
    (output/'SOURCE_ROWS.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    prepare_admitted(output)
    print(json.dumps(audit))


def prepare_admitted(output):
    """Frozen pre-Base source vetoes, then one fixed full-range conversion."""
    from PIL import Image
    import numpy as np
    excluded = {'p03': 'OTHER_PROJECTION_DEPENDENCY',
                'p09': 'DEVICE_LEAD_LOCATION_NOT_DEVICE_LATERALITY',
                'p10': 'BORDERLINE_PROTECTION', 'p15': 'TEMPORAL_PROTECTION'}
    rows = [r for r in json.loads((output/'SOURCE_ROWS.json').read_text())
            if r['source_group'] not in excluded]
    (output/'prepared').mkdir(exist_ok=True)
    for group in dict.fromkeys(r['source_group'] for r in rows):
        with Image.open(output/'images'/f'{group}.png') as original:
            pixels = np.asarray(original, dtype=np.float64)
        Image.fromarray(np.rint(pixels / 257).astype('uint8')).convert('RGB').save(output/'prepared'/f'{group}.png')
    for i, row in enumerate(rows):
        row.update(qid=970000+i, query_id=row['source_group']+'_'+row['role'],
                   image_path=str((output/'prepared'/f"{row['source_group']}.png").resolve()),
                   audit_role='SOURCE_QUALIFICATION',
                   content_type='Laterality' if row['role'] in ['target','brief_control'] else 'Presence')
    (output/'SOURCE_ROWS_ADMITTED.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (output/'SOURCE_REVIEW.json').write_text(json.dumps(dict(original=24, admitted=20,
        excluded=excluded, replacement=False, image_conversion='round(uint16/257) RGB',
        model_answers_seen=0), indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
