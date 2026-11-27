"""Freeze eligible comparison order before the Stage-1 outcomes are available."""
from collections import Counter
import json
from pathlib import Path
import random
import sys

from calibration_panel import eligible_by_image
from freeze_data import SEED


def main(root):
    options, excluded = eligible_by_image(root, ['pilot_inputs.private.json', 'calibration_panel.private.json'])
    rng = random.Random(SEED + 3)
    by_scope = {}
    for image in sorted(options):
        rows = options[image]
        chosen = rng.choice(rows)
        by_scope.setdefault(chosen['scope'], []).append(chosen)
    for value in by_scope.values():
        rng.shuffle(value)
    ordered = []
    while any(by_scope.values()):
        for scope in sorted(by_scope):
            if by_scope[scope]:
                ordered.append(by_scope[scope].pop())
    images = {r['source']['source_rows'][0]['image_sha256'] for r in ordered}
    assert len(images) == len(ordered) == len(options) and not images & excluded
    summary = dict(status='ELIGIBLE_ORDER_FROZEN_AWAITING_THROUGHPUT_AND_STAGE1',
                   seed=SEED + 3, images=len(images),
                   scope=dict(Counter(r['scope'] for r in ordered)),
                   candidate_cap=20, updates_at_cap=5 * 20 * 160, new_model_calls=0)
    with (root / 'comparison_order.private.json').open('x') as out:
        json.dump(dict(summary, entries=ordered), out, ensure_ascii=False)
    print(json.dumps(summary))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
