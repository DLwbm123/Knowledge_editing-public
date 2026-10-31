"""Aggregate connectivity of authorized identity fields; never decode QA."""
import collections
import json
import os
import sys
from pathlib import Path

root = Path(os.environ['TASK_REPO'])
sys.path.insert(0, str(root / 'experiments/medmkeb_usable_dataset_v1'))
from metadata_inventory import records, Groups

state = json.loads((root / '.private/medmkeb_independent_admission_v15_20261019/STATE.json').read_text())
data = Path(state['local_data_root'])
paths = [data / 'raw/train_data.json', data / 'raw/eval_data_threehop_final.json']
if any(not p.is_file() or p.stat().st_size > 64 * 1024**2 for p in paths):
    raise ValueError('source missing or oversized')
rows = [records(p) for p in paths]
if [len(x) for x in rows] != [4490, 2497]:
    raise ValueError('source count changed')
g = Groups()
for split in rows:
    for image, rephrase, article in split:
        if not image or not rephrase or not article:
            raise ValueError('unresolved identity')
        g.union('I:' + image, 'I:' + rephrase)
        g.union('I:' + image, 'A:' + article)
counts = collections.defaultdict(lambda: [0, 0])
for split_id, split in enumerate(rows):
    for image, _, _ in split:
        counts[g.root('I:' + image)][split_id] += 1
sizes = sorted((sum(x) for x in counts.values()), reverse=True)
result = {
    'source_records': [len(x) for x in rows],
    'joint_components': len(counts),
    'components_with_both_official_splits': sum(a > 0 and b > 0 for a, b in counts.values()),
    'records_in_both_split_components': sum(a+b for a, b in counts.values() if a and b),
    'largest_component_records': sizes[0] if sizes else 0,
    'largest_ten_component_records': sizes[:10],
    'questions_answers_decoded': 0, 'pixels_read': 0, 'model_calls': 0,
    'role_admission': 'UNKNOWN',
    'limitations': ['source identity aliases may be incomplete', 'joint edges represent partition constraints, not same-image claims', 'protected historical roles not yet excluded']
}
out = Path(os.environ['TASK_OUT'])
with out.open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=False))
