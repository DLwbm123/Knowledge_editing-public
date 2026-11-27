"""Inventory new same-image negative candidates before semantic admission."""
import json
from collections import defaultdict
from pathlib import Path
import re
import unicodedata


def read(path):
    return json.loads(path.read_text())


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', str(text)).casefold()))


root = Path.cwd()
outputs = root.parent
bank = outputs / 'med-mdu150-v1-20261105/run/MDU/BANK_INPUTS_PRIVATE.json'
frozen = {(r['query']['image_sha256'], normalize(r['query']['question']))
          for r in read(bank)['records']}
sources = {}
for name in ('med-medinit-loss-v1-20261103','med-medinit-101-v1-20261104','med-mdu150-v1-20261105'):
    for entry in read(outputs / name / 'run/MDU/TRAIN_SOURCE_BOUND_PRIVATE.json'):
        owner = entry['task']['order']
        row = entry['source_rows'][0]
        pair = row['image_sha256'], normalize(row['reference'])
        if owner in sources:
            assert sources[owner] == pair
        sources[owner] = pair
assert len(sources) == 150
images = {sha for sha, _ in sources.values()}
inventory = outputs / 'm3bench_task_specific_core9_data/20260903T013245Z/base_inventory/BASE_QUERY_INVENTORY_CORE9.jsonl'
questions = defaultdict(lambda: defaultdict(dict))
with inventory.open() as stream:
    for line in stream:
        row = json.loads(line)
        sha = row['image_sha256']
        if sha not in images or row['source_task'] != 'PUBLIC_SOURCE_QA':
            continue
        question = normalize(row['question'])
        answer = normalize(row['gold_answer'])
        if question and answer and (sha, question) not in frozen:
            questions[sha][question][answer] = {k: row[k] for k in ('query_id','question','gold_answer')}
candidates = []
for owner, (sha, target) in sorted(sources.items()):
    for _, answers in sorted(questions[sha].items()):
        if len(answers) == 1 and target not in answers:
            candidates.append({'owner':owner, 'image_sha256':sha, **next(iter(answers.values()))})
with (root/'negative_candidates.private.json').open('x') as stream:
    json.dump({'status':'PENDING_SEMANTIC_AND_ROLE_ADMISSION', 'candidates':candidates}, stream)
print(json.dumps({
    'status':'PENDING_SEMANTIC_AND_ROLE_ADMISSION',
    'edit_question_pairs':len(candidates),
    'edit_support':len({v['owner'] for v in candidates}),
    'unique_image_questions':len({(v['image_sha256'], normalize(v['question'])) for v in candidates}),
    'image_support':len({v['image_sha256'] for v in candidates}),
}))
