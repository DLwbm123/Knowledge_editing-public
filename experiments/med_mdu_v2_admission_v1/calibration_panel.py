"""Reserve an image-disjoint development panel for choosing the common KL coefficient."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

from freeze_data import FAMILIES, SEED, scope


def eligible_by_image(root, panel_names):
    def load(name):
        return json.loads((root / name).read_text())

    inventory = load('inventory.private.json')
    excluded = {v['source']['source_rows'][0]['image_sha256']
                for name in panel_names for v in load(name)['entries']}
    negatives = defaultdict(lambda: defaultdict(list))
    base = load('base_continuations.private.json')
    for n in load('relation_labels.private.json'):
        if n['label'] == 'independent' and n['image_sha256'] not in excluded:
            negatives[n['owner']][n['role']].append(dict(n, base_continuation=base[n['query_id']]))
    views = defaultdict(list)
    for v in load('view_candidates.private.json'):
        if v['role'] in ('train', 'validation'):
            views[v['owner']].append(v)
    options = defaultdict(list)
    for owner, ns in sorted(negatives.items()):
        src = inventory['sources'][str(owner)]
        image = src['source_rows'][0]
        train = {v['family'] for v in views[owner] if v['role'] == 'train'}
        if not ns['train'] or not ns['holdout'] or not set(FAMILIES[1:]) <= train:
            continue
        data = views[owner].copy()
        if 'ordinary' not in train:
            fit = src['source_rows'][1]
            data.append(dict(role='train', family='ordinary', template='historical_fit1',
                             question=fit['question'], reference=fit['reference']))
        e = dict(owner=owner, source=src, views=data, negative=dict(ns),
                 scope=scope(image['question']))
        options[image['image_sha256']].append(e)
    return options, excluded


def build(root):
    options, excluded = eligible_by_image(root, ['pilot_inputs.private.json'])
    rng = random.Random(SEED + 1)
    picked = []
    for category in ('whole_image_findings', 'specific_finding_or_property'):
        candidates = sorted(image for image, rows in options.items()
                            if any(r['scope'] == category for r in rows))
        for image in rng.sample(candidates, 4):
            picked.append(rng.choice([r for r in options.pop(image) if r['scope'] == category]))
    rng.shuffle(picked)
    assert len({e['source']['source_rows'][0]['image_sha256'] for e in picked}) == 8
    assert all(e['source']['source_rows'][0]['image_sha256'] not in excluded for e in picked)
    with (root / 'calibration_panel.private.json').open('x') as out:
        json.dump(dict(status='DEVELOPMENT_KL_ONLY', seed=SEED + 1, entries=picked), out, ensure_ascii=False)
    print('calibration_images', len(picked), 'negatives',
          {role: sum(len(x['negative'][role]) for x in picked) for role in ('train', 'holdout')})


if __name__ == '__main__':
    build(Path(sys.argv[1]))
