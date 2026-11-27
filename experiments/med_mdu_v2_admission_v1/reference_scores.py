"""Freeze method-blind input and aggregate CORE for the new reference panel."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'medical_proposition_v133'))
from med133_score_contract import PROTOCOL, PROMPT, digest


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False)


def prepare(panel, root):
    rows = json.loads(panel.read_text())
    assert len(rows) == 90
    payloads = {}
    consumers = []
    for row in rows:
        for arm in ('MED', 'MDU'):
            question = row['question']
            gold = row['reference']
            answer = row['answers'][arm]['answer']
            key = digest([PROTOCOL, question, gold, answer])
            payloads[key] = dict(opaque_query_id=key, question=question, gold_answer=gold,
                                 raw_base_answer=answer)
            consumers.append(dict(key=key, owner=row['owner'], role=row['role'], arm=arm))
    assert len(consumers) == len({(r['owner'], r['role'], r['arm']) for r in consumers}) == 180
    order = sorted(payloads)
    random.Random(20261127).shuffle(order)
    write(root / 'SUPPLEMENT_INPUT.private.json',
          dict(status='SUPPLEMENT_IDENTITIES_FROZEN', payloads=payloads,
               categories=dict(history_unverified=order, conflict_unknown=[], known_unknown=[])))
    write(root / 'consumers.private.json', consumers)
    write(root / 'contract.json', dict(protocol=PROTOCOL, prompt=PROMPT, model='gpt-6-astra',
                                       reasoning_effort='medium', snapshot=None))
    print(len(order))


def summarize(root, output):
    panel = json.loads((root / 'consumers.private.json').read_text())
    ledger = json.loads((root / 'DECISIONS.json').read_text())
    receipt = json.loads((root / 'COMPLETE.json').read_text())
    assert set(ledger['decisions']) | set(ledger['unknown']) == {r['key'] for r in panel}
    groups = defaultdict(lambda: defaultdict(dict))
    for row in panel:
        decided = ledger['decisions'].get(row['key'])
        groups[row['role']][row['owner']][row['arm']] = None if decided is None else decided['core_correct']
    result = {}
    for role, owners in groups.items():
        pairs = list(owners.values())
        assert len(pairs) == 30 and all(set(r) == {'MED', 'MDU'} for r in pairs)
        result[role] = {'events': len(pairs),
                        'gain': sum(r['MED'] is False and r['MDU'] is True for r in pairs),
                        'loss': sum(r['MED'] is True and r['MDU'] is False for r in pairs),
                        'paired_unknown': sum(None in r.values() for r in pairs),
                        'arms': {a: {'correct': sum(r[a] is True for r in pairs),
                                     'unknown': sum(r[a] is None for r in pairs)} for a in ('MED', 'MDU')}}
    write(output, dict(receipt=receipt, results=result, immutable_judge_snapshot=None))


if __name__ == '__main__':
    mode, first, second = sys.argv[1:]
    {'prepare': prepare, 'summarize': summarize}[mode](Path(first), Path(second))
