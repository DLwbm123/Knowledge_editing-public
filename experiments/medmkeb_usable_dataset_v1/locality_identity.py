"""Augment the private identity lock with frozen locality-image matches only."""
import collections
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['TASK_REPO'])
sys.path.insert(0, str(root / 'experiments/medmkeb_usable_dataset_v1'))
from metadata_inventory import Groups, article, image_key
import score_index as si
import stream_extract as s

base = root / '.private/medmkeb_usable_dataset_v1_20261031'
oldpath = base / 'IDENTITY_QUARANTINE_V18.json'
v18 = root / '.private/medmkeb_expanded_v18_20261019/CANDIDATES_FROZEN.json'
state = json.loads((root / '.private/medmkeb_independent_admission_v15_20261019/STATE.json').read_text())
data = Path(state['local_data_root']) / 'raw'
paths = [data / 'train_data.json', data / 'eval_data_threehop_final.json']
inputs = paths + [oldpath, v18]
before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]
old = json.loads(oldpath.read_text())
assert old['status'] == 'IDENTITY_ONLY_NOT_ADMITTED' and len(old['records']) == 6987

roles = None
with v18.open('rb') as f:
    assert s.nonspace(f) == b'{'
    while True:
        t = s.nonspace(f)
        if t == b'}': break
        assert t == b'"'
        key = s.string(f, True)
        assert s.nonspace(f) == b':'
        t = s.nonspace(f); start = f.tell() - 1; si.skip(f, t); end = f.tell()
        if key == 'actual_locality_image_roles':
            assert roles is None and end - start < 10_000_000
            f.seek(start); roles = json.loads(f.read(end - start)); f.seek(end)
        t = s.nonspace(f)
        if t == b'}': break
        assert t == b','
    assert not f.read().strip()
assert isinstance(roles, dict) and len(roles) == 512
assert all(isinstance(k, str) and v in ('RETENTION_TRAIN', 'RETENTION_EVAL') for k, v in roles.items())

rows = []
for split, path, expected in zip(('train', 'eval'), paths, (4490, 2497)):
    n = 0
    with path.open('rb') as f:
        assert s.nonspace(f) == b'['
        while True:
            t = s.nonspace(f)
            if t == b']': break
            assert t == b'{' 
            obj = si.fields(f, {'image', 'image_rephrase', 'm_loc'})
            assert set(obj) == {'image', 'image_rephrase', 'm_loc'}
            assert all(isinstance(v, str) and v for v in obj.values())
            rows.append((split, n, image_key(obj['image']), image_key(obj['image_rephrase']), article(obj['m_loc']), image_key(obj['m_loc'])))
            n += 1
            t = s.nonspace(f)
            if t == b']': break
            assert t == b','
        assert not f.read().strip()
    assert n == expected
g = Groups()
for _, _, a, b, _, _ in rows: g.union(a, b)
counts = collections.Counter(); locked = []
for prior, (split, i, a, b, pmc, locality) in zip(old['records'], rows):
    assert (prior['official_split'], prior['index'], prior['edit_image_group'], prior['locality_article']) == (split, i, g.root(a), pmc)
    entry = dict(prior); entry['v18_frozen_locality_image_role'] = roles.get(locality)
    if locality in roles: counts[roles[locality]] += 1
    locked.append(entry)
assert len(locked) == len(old['records']) and before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]
out = Path(os.environ['TASK_PRIVATE_OUT'])
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    json.dump({'status': 'IDENTITY_ONLY_NOT_ADMITTED', 'input_stats': before, 'records': locked}, f, separators=(',', ':'))
print(json.dumps({'records': len(locked), 'locality_image_matches': dict(counts), 'qa_decoded': 0, 'pixels': 0, 'models': 0}))
