"""Build private weak spans from an already admitted training slice only."""
import collections
import json
import os
from pathlib import Path
import re

root = Path(os.environ['TASK_INPUT_ROOT'])
out = Path(os.environ['TASK_PRIVATE_OUT'])
admission = json.loads((root / 'ROLE_ADMISSION.json').read_text())
tasks = json.loads((root / 'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
if admission['status'] != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or len(tasks) != 16:
    raise ValueError('Unexpected training admission')
if [str(e['task']['order']) for e in tasks] != [str(x) for x in admission['owners']]:
    raise ValueError('Owner order mismatch')
terms = {
    'anatomy': 'brain lung lungs heart liver kidney kidneys spleen pancreas stomach intestine bowel bladder breast chest abdomen abdominal pelvis pelvic spine spinal head neck'.split(),
    'modality': ['x-ray', 'xray', 'radiograph', 'ct', 'mri', 'ultrasound', 'pet'],
    'observation': 'lesion mass nodule tumor fracture opacity effusion hemorrhage bleeding infection'.split(),
    'qualifier': 'left right bilateral upper lower anterior posterior no not without absent possible suspected uncertain'.split(),
}
patterns = {k: re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, sorted(v, key=len, reverse=True))) + r')(?!\w)', re.I) for k, v in terms.items()}
word = re.compile(r'[A-Za-z]+')
counts = collections.Counter()
records = []
for index, entry in enumerate(tasks):
    rows = entry['source_rows']
    if len(rows) != 5 or [r['audit_role'] for r in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
        raise ValueError('Training role mismatch')
    refs = [r['reference'].strip() for r in rows]
    if len(set(refs)) != 1:
        raise ValueError('Training reference mismatch')
    answer = refs[0]
    matches = {name: list(pattern.finditer(answer)) for name, pattern in patterns.items()}
    occupied = [m.span() for group in matches.values() for m in group]
    record = {'index': index, 'status': 'FALLBACK'}
    if matches['anatomy'] and matches['observation']:
        counts['two_category_candidates'] += 1
        med = [matches['anatomy'][0], matches['observation'][0]]
        if med[0].span() == med[1].span():
            raise ValueError('Overlapping category spans')
        choices = [m for m in word.finditer(answer) if not any(m.start() < b and a < m.end() for a, b in occupied)]
        picked = []
        for m in med:
            item = next((w for w in choices if w not in picked and len(w.group()) == len(m.group())), None)
            if item is None:
                break
            picked.append(item)
        if len(picked) == 2:
            record.update(status='WEAK_CHARACTER_MATCH_ONLY', medical_spans=[m.span() for m in med], control_spans=[m.span() for m in picked])
            counts['character_matched'] += 1
        else:
            counts['control_unmatched'] += 1
    else:
        counts['category_missing'] += 1
    records.append(record)
if sum(counts[k] for k in ('character_matched', 'control_unmatched', 'category_missing')) != 16:
    raise ValueError('Coverage incomplete')
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    json.dump({'status': 'WEAK_SPANS_NOT_ADMITTED', 'records': records}, f, separators=(',', ':'))
print(json.dumps({'edits': 16, 'counts': dict(counts), 'qa_returned': 0, 'models': 0, 'pixels': 0}))
