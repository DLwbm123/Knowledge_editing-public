"""User-requested medium Astra rejudging of all frozen current binary payloads."""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import time

RUN = Path(os.environ['RUN_ROOT'])
spec = importlib.util.spec_from_file_location('astra_base_queue', RUN/'private/tools/astra_base_queue.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.ROOT = RUN/'private/judge_astra_medium'
q.EFFORT = 'medium'
q.BATCH_LEDGER = 'Astra_medium_batches'
EPOCH = 'MEDTRACE_POST32_ALL_OUTPUTS_ASTRA_MEDIUM_20261007_V1'


def convert(row):
    full = json.loads(row['binding'])
    old_record = json.loads(row['record'])
    assert set(old_record) == {'opaque_query_id', 'question', 'gold_answer', 'raw_base_answer'}
    assert (old_record['question'], old_record['gold_answer'], old_record['raw_base_answer']) == (
        full['question'], full['reference'], full['output']['raw_answer'])
    full['judge'] = dict(model='gpt-6-astra', reasoning_effort='medium', protocol=q.PROTOCOL, prompt=q.PROMPT)
    key = q.digest(full)
    record = dict(old_record, opaque_query_id=q.digest([EPOCH, key]))
    return key, record, full


def initialize(db):
    receipt = q.ROOT/'EPOCH_MANIFEST.json'
    if receipt.exists():
        lock = q.read(receipt)
        assert (lock['epoch'], lock['model'], lock['reasoning_effort'], lock['prompt']) == (EPOCH, 'gpt-6-astra', 'medium', q.PROMPT)
        assert db.execute('SELECT count(*) FROM payload').fetchone()[0] == lock['payloads']
        return
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0] == 0, 'Incomplete initialization: inspect before recovery'
    source = sqlite3.connect('file:'+str(RUN/'private/judge_common/queue.sqlite')+'?mode=ro', uri=True)
    source.row_factory = sqlite3.Row
    source.execute('BEGIN')
    rows = source.execute('SELECT * FROM payload ORDER BY rowid').fetchall()
    assert len(rows) == 2137 and all(r['status'] == 'FORMAT_VALID' for r in rows)
    db.execute('CREATE TABLE paired (old_key TEXT PRIMARY KEY, new_key TEXT UNIQUE, qwen_correct INTEGER)')
    for row in rows:
        key, record, full = convert(row)
        db.execute('INSERT INTO payload VALUES (?,?,?,\'PENDING\',NULL,NULL)', (key, json.dumps(record, ensure_ascii=False), json.dumps(full, ensure_ascii=False)))
        db.execute('INSERT INTO paired VALUES (?,?,?)', (row['key'], key, row['correct']))
    mapping = dict(db.execute('SELECT old_key,new_key FROM paired'))
    consumers = source.execute('SELECT * FROM consumer ORDER BY rowid').fetchall()
    assert len(consumers) == 9501
    for row in consumers:
        values = list(row)
        values[-1] = mapping[row['payload_key']]
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)', values)
    db.execute('INSERT INTO done VALUES (\'generation\')')
    db.commit()
    source.close()
    m = q.read(RUN/'RUN_MANIFEST.json')
    attempts = q.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']
    assert attempts + len(rows) <= m['Judge_limit']
    q.write(receipt, dict(epoch=EPOCH, model='gpt-6-astra', reasoning_effort='medium', protocol=q.PROTOCOL,
        prompt=q.PROMPT, payloads=len(rows), consumers=len(consumers), original_Qwen_preserved=True,
        all_current_binary_payloads_included=True, original_masks_preserved=True, batch_size=50,
        semantic_retries=0, transport_retries=0, old_Astra_verdicts_reused=0, attempts_per_payload=1,
        Judge_before=attempts, Judge_limit=m['Judge_limit'], deadline_epoch=m['deadline_epoch'],
        authorized_by='User requested Astra with medium reasoning for current outputs', created_epoch=time.time()))


def selfcheck():
    row = dict(binding=json.dumps(dict(question='Q', reference='A', output=dict(raw_answer='A', raw_token_ids=[1]), judge={})),
               record=json.dumps(dict(opaque_query_id='old', question='Q', gold_answer='A', raw_base_answer='A')))
    key, record, full = convert(row)
    assert full['judge']['reasoning_effort'] == 'medium' and record['opaque_query_id'] != 'old'
    assert convert(row)[0] == key
    full['output']['raw_token_ids'] = [2]
    assert convert(dict(row, binding=json.dumps(full)))[0] != key
    try:
        convert(dict(row, record=json.dumps(dict(record, raw_base_answer='changed'))))
    except AssertionError:
        pass
    else:
        raise AssertionError('Changed output was accepted')
    return dict(status='PASS', tests=['medium identity', 'exact sharing', 'token separation', 'mismatch rejection'])


if __name__ == '__main__':
    operation = json.loads(os.environ['QUEUE_REQUEST'])
    if operation['action'] == 'selfcheck':
        answer = selfcheck()
    else:
        db = q.connect()
        initialize(db)
        q.ingest = lambda db: None  # Frozen snapshot; no further generation is authorized here.
        answer = q.request(db, operation)
    print(json.dumps(answer, ensure_ascii=False))
