"""Private CPU queue: one new common Astra epoch, exact payload sharing, no retries."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

RUN = Path(os.environ['RUN_ROOT'])
sys.path.insert(0, str(RUN / 'private/source'))
from scripts.medtrace.stage17_prepare import PROMPT, PROTOCOL, digest
from scripts.medtrace.astra_judge_bundle import validate

ROOT = RUN / 'private/judge_common'
EFFORT = 'high'
BATCH_LEDGER = 'Judge_batches'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n'); tmp.replace(p)


@contextlib.contextmanager
def resources():
    with (RUN / 'RESOURCE_LEDGER.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = read(RUN / 'RESOURCE_LEDGER.json')
        yield d
        write(RUN / 'RESOURCE_LEDGER.json', d)


def connect():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / 'queue.sqlite', timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS payload (key TEXT PRIMARY KEY, record TEXT, binding TEXT, status TEXT, batch TEXT, correct INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS consumer (id TEXT PRIMARY KEY, method TEXT, mode TEXT, prefix INTEGER, edit_order INTEGER, folder TEXT, query_id TEXT, path TEXT, output_binding TEXT, payload_key TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS done (name TEXT PRIMARY KEY)')
    return db


def initialize(db):
    lock = dict(epoch='MEDTRACE_ORIGINAL146_COMMON_ASTRA_20261003_V1', model='gpt-6-astra',
        reasoning_effort='high', immutable_snapshot=None, protocol=PROTOCOL, prompt=PROMPT,
        batch_size=50, semantic_retries=0, transport_retries=0, attempts_per_payload=1,
        consecutive_transport_failure_limit=3, original_masks_preserved=True,
        old_success_scores_reused=0, payload_identity='query/image/prompt/attention/runtime/generation/raw_text/raw_tokens/judge',
        consumer_identity='method/mode/prefix/edit/ancestry/training and complete output binding; retained separately',
        source_commit=read(RUN / 'private/SOURCE_COMMIT.json')['commit'],
        starting_epoch=read(RUN / 'RUN_MANIFEST.json')['starting_epoch'],
        deadline_epoch=read(RUN / 'RUN_MANIFEST.json')['deadline_epoch'], Judge_limit=32000)
    lock['config_binding'] = digest(lock)
    path = ROOT / 'EPOCH_MANIFEST.json'
    if path.exists():
        assert read(path) == lock, 'Common Judge epoch changed'
    else:
        assert read(RUN / 'RESOURCE_LEDGER.json')['Judge_attempts'] == 0
        write(path, lock)
    old = read(RUN / 'private/legacy_stage17/JUDGE_LOCK.json')
    assert (old['model'], old['prompt'], old['protocol']) == (lock['model'], PROMPT, PROTOCOL)
    old_base = [json.loads(x) for x in (RUN / 'private/legacy_stage17/VERDICTS_ASTRA.jsonl').read_text().splitlines() if x.strip()]
    assert all(type(x['is_correct']) is bool for x in old_base)
    missing_root = Path('/data/bmw/Knowledge_editing/outputs/scope-text-guard-20261001/run/private/judge_sol')
    missing = read(missing_root / 'JUDGE_MISSING_LOCK.json')['keys']
    old_missing = [read(missing_root / 'pending' / (k + '.json')) for k in missing]
    assert len(missing) == 41 and all(x['judge_binding']['model'] == 'gpt-6.1-sol' and x['judge_binding']['protocol'] != PROTOCOL for x in old_missing)
    write(ROOT / 'PERMANENT_MISSING_INHERITANCE_AUDIT.json', dict(
        original_Astra_Base_records=len(old_base), original_Astra_Base_permanent_missing=0,
        historical_SOL_permanent_missing_preserved=len(missing),
        same_full_Astra_model_prompt_payload_missing=0,
        rule='Only identical full model/prompt/payload identity inherits; SOL missing keys remain untouched, never rejudged as SOL',
        historical_missing_protocols=sorted({x['judge_binding']['protocol'] for x in old_missing})))


def payload(db, q, b, output):
    assert (q['question'], q['reference'], q['image_sha256']) == (b['question'], b['reference'], b['image_sha256'])
    assert isinstance(output['raw_answer'], str) and isinstance(output['raw_token_ids'], list)
    assert all(type(x) is int for x in output['raw_token_ids'])
    lock = read(ROOT / 'EPOCH_MANIFEST.json')
    full = dict(query_id=q['query_id'], question=q['question'], reference=q['reference'],
        image_sha256=q['image_sha256'], image_path=b['image_path'], prompt_ids=b['prompt_ids'],
        attention_mask=b['attention_mask'], runtime=b['runtime'], generation=b['generation'], output=output,
        judge={k: lock[k] for k in ['model', 'reasoning_effort', 'protocol', 'prompt']})
    key = digest(full)
    record = dict(opaque_query_id=digest([lock['epoch'], key]), question=q['question'],
        gold_answer=q['reference'], raw_base_answer=output['raw_answer'])
    encoded = json.dumps(full, ensure_ascii=False)
    exists = db.execute('SELECT binding FROM payload WHERE key=?', (key,)).fetchone()
    if exists:
        assert json.loads(exists['binding']) == full
    else:
        db.execute('INSERT INTO payload VALUES (?,?,?,\'PENDING\',NULL,NULL)',
            (key, json.dumps(record, ensure_ascii=False), encoded))
    return key


def ingest(db):
    l = read(RUN / 'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    b = read(RUN / 'private/legacy_stage17/BINDINGS.json')
    tasks = {t['edit_id']: t for t in l['tasks']}
    ordered = [tasks[e] for e in l['main_T0']]
    assert len(ordered) == 146
    def add(row, output=None):
        cid = digest(row)
        if db.execute('SELECT 1 FROM consumer WHERE id=?', (cid,)).fetchone():
            return
        q = l['queries'][row['query_id']]; base = b[q['opaque_Base_id']]
        if output is None:
            d = read(row['path'])
            assert d['binding']['input'] == q and d['Base_cache_id'] == q['opaque_Base_id']
            output = d.get('modes', d)[row['arm_key']]
            out_binding = digest(d)
        else:
            out_binding = digest(base)
        key = payload(db, q, base, output)
        parents = [p.name for p in Path(row['path']).parents]
        order = row.get('edit_order', next((int(n[1:]) for n in parents if n.startswith('e') and n[1:].isdigit()), 0))
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
            (cid, row['method'], row['mode'], row['prefix'], order, row.get('folder', 'panel'), row['query_id'], row['path'], out_binding, key))
    if not db.execute('SELECT 1 FROM done WHERE name=\'Base\'').fetchone():
        for q in l['queries'].values():
            base = b[q['opaque_Base_id']]
            add(dict(method='Base', mode='base', prefix=0, query_id=q['query_id'], path=str(RUN / 'private/legacy_stage17/BINDINGS.json')),
                dict(raw_answer=base['output']['model_answer_raw'], raw_token_ids=base['output']['raw_generated_token_ids']))
        db.execute('INSERT INTO done VALUES (\'Base\')')
    if not db.execute('SELECT 1 FROM done WHERE name=\'baselines\'').fetchone():
        assert read(RUN / 'private/BASELINE_ADMISSION.json')['status'] == 'PASS'
        assert read(RUN / 'private/BASELINE_METHOD_LOCK_ADMISSION.json')['status'] == 'PASS'
        for row in read(RUN / 'private/BASELINE_RAW_CONSUMERS.json'):
            add(row)
        db.execute('INSERT INTO done VALUES (\'baselines\')')
    for t in ordered:
        folder = RUN / 'private/edits' / f"e{t['order']:03d}"
        done = folder / 'COMPLETE.json'
        if not done.exists():
            continue
        receipt = read(done); phase = receipt['binding']
        assert receipt['status'] == 'GENERATED_NOT_SCORED' and phase['freeze_id'] == l['freeze_id']
        assert phase['input'] == t['native'] and phase['fit'] == t['fit_questions']
        assert phase['code'] == read(RUN / 'private/SOURCE_COMMIT.json')['commit']
        ids = list(dict.fromkeys([t['edit_id']] + [q for e in t['events'] for q in e['all_probe_query_ids']]))
        files = list((folder / 'single').glob('*.json'))
        assert len(files) == len(ids) == receipt['queries']
        assert {read(p)['binding']['input']['query_id'] for p in files} == set(ids)
        for p in files:
            d = read(p); q = d['binding']['input']; qb = d['binding']
            assert q['query_id'] in ids and qb['phase'] == phase and qb['prefix'] == 1 and qb['inserted'] == [t['edit_id']]
            assert qb['generation'] == b[q['opaque_Base_id']]['generation']
            add(dict(method='medtrace', mode='single', prefix=1, query_id=q['query_id'], path=str(p), arm_key='R0', edit_order=t['order']))
    if (RUN / 'private/GENERATION_COMPLETE.json').exists():
        phase = dict(arm='MedTRACE_AVAILABLE_H_R0', freeze=l['freeze_id'], source=read(RUN / 'private/SOURCE_COMMIT.json')['commit'], order=l['main_T0'], prefixes=[1, 50, 100, 146])
        for i, t in enumerate(ordered, 1):
            groups = [('native', [t['edit_id']])]
            if i in [1, 50, 100, 146]:
                groups.append(('panel', list(dict.fromkeys(q for row in ordered[:i] for q in [row['edit_id']] + [z for e in row['events'] for z in e['all_probe_query_ids']]))))
            for folder, expected in groups:
                files = list((RUN / 'private/sequential' / f'e{i:03d}' / folder).glob('*.json'))
                assert len(files) == len(expected)
                assert {read(p)['binding']['input']['query_id'] for p in files} == set(expected)
                for p in files:
                    d = read(p); qb = d['binding']; q = qb['input']
                    assert qb['phase'] == phase and qb['prefix'] == i and qb['inserted'] == l['main_T0'][:i] and q['query_id'] in expected
                    assert qb['generation'] == b[q['opaque_Base_id']]['generation']
                    add(dict(method='medtrace', mode='sequential', prefix=i, query_id=q['query_id'], path=str(p), arm_key='R0', folder=folder, edit_order=i))
        db.execute('INSERT OR IGNORE INTO done VALUES (\'generation\')')
    db.commit()


def reconcile(db):
    ledger = read(RUN / 'RESOURCE_LEDGER.json')
    for a in ledger.get(BATCH_LEDGER, []):
        for key in a['keys']:
            if a['status'] == 'RESERVED':
                db.execute("UPDATE payload SET status='RESERVED',batch=? WHERE key=? AND status='PENDING'", (a['id'], key))
            elif a['status'] == 'FAILED_NO_RETRY':
                db.execute("UPDATE payload SET status='MISSING',batch=? WHERE key=? AND status!='FORMAT_VALID'", (a['id'], key))
    db.commit()


def status(db):
    ledger = read(RUN / 'RESOURCE_LEDGER.json'); manifest = read(RUN / 'RUN_MANIFEST.json')
    stopped = ((RUN / 'STOP').exists() or (RUN / 'private/BUDGET_STOP.json').exists()
        or (RUN / 'private/CONTROLLER_FAILURE.json').exists() or time.time() >= manifest['deadline_epoch'] - 600
        or ledger['Judge_attempts'] >= manifest['Judge_limit'])
    return dict(status='STOPPED' if stopped else ('GENERATION_COMPLETE' if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() else 'GENERATING'),
        payloads=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status').fetchall()),
        consumers=[dict(x) for x in db.execute('SELECT method,mode,count(*) AS count FROM consumer GROUP BY method,mode')],
        resource_Judge_attempts=read(RUN / 'RESOURCE_LEDGER.json')['Judge_attempts'])


def request(db, operation):
    reconcile(db)
    if operation['action'] == 'next':
        ingest(db)
        remaining = read(RUN / 'RUN_MANIFEST.json')['Judge_limit'] - read(RUN / 'RESOURCE_LEDGER.json')['Judge_attempts']
        return dict(state=status(db), rows=[dict(key=r['key'], record=json.loads(r['record'])) for r in db.execute("SELECT key,record FROM payload WHERE status='PENDING' ORDER BY rowid LIMIT ?", (min(50, max(0, remaining)),))])
    if operation['action'] == 'reserve':
        keys = operation['keys']; bid = operation['batch_id']
        assert keys and len(keys) == len(set(keys)) <= 50
        for k in keys:
            assert db.execute('SELECT status FROM payload WHERE key=?', (k,)).fetchone()['status'] == 'PENDING'
        with resources() as ledger:
            m = read(RUN / 'RUN_MANIFEST.json')
            assert time.time() < m['deadline_epoch'] - 600 and not (RUN / 'STOP').exists()
            assert ledger['Judge_attempts'] + len(keys) <= m['Judge_limit']
            attempts = ledger.setdefault(BATCH_LEDGER, [])
            assert not any(a['id'] == bid or set(a['keys']) & set(keys) for a in attempts)
            previous = list(reversed(attempts))
            consecutive = 0
            for a in previous:
                if a['status'] == 'FAILED_NO_RETRY' and a.get('transport_failure'):
                    consecutive += 1
                else:
                    break
            assert consecutive < 3, 'Three transport failures: no new requests'
            ledger['Judge_attempts'] += len(keys)
            attempts.append(dict(id=bid, keys=keys, status='RESERVED', started_epoch=time.time(), model='gpt-6-astra', reasoning_effort=EFFORT))
        reconcile(db)
        return dict(status='RESERVED', items=len(keys))
    if operation['action'] == 'publish':
        bid = operation['batch_id']; evidence = operation['evidence']; response = operation.get('response')
        ledger = read(RUN / 'RESOURCE_LEDGER.json'); a = next(x for x in ledger[BATCH_LEDGER] if x['id'] == bid)
        rows = [db.execute('SELECT * FROM payload WHERE key=?', (k,)).fetchone() for k in a['keys']]
        batch = dict(batch_id=bid, records=[json.loads(r['record']) for r in rows])
        valid = evidence.get('status') == 'FORMAT_VALID' and response is not None
        if valid:
            assert evidence['actual_model'] == 'gpt-6-astra' and evidence['reasoning_effort'] == EFFORT and evidence['input_binding'] == digest(batch)
            assert evidence['exit_code'] == 0 and not evidence.get('errors') and not evidence['tool_event_types'] and all(evidence['isolation_checks'].values())
            decisions = validate(batch, response)
        else:
            decisions = [None] * len(rows)
        destination = ROOT / 'evidence' / (bid + '.json')
        saved = dict(batch=batch, evidence=evidence, response=response)
        if destination.exists():
            assert read(destination) == saved
        else:
            write(destination, saved)
        for row, decision in zip(rows, decisions):
            new = 'FORMAT_VALID' if valid else 'MISSING'
            if row['status'] == 'FORMAT_VALID':
                assert valid and row['correct'] == int(decision['is_correct'])
            else:
                assert row['status'] in ('RESERVED', 'MISSING')
                assert row['status'] != 'MISSING' or not valid, 'Permanent missing cannot be rejudged'
                db.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',
                    (new, int(decision['is_correct']) if valid else None, bid, row['key']))
        db.commit()
        with resources() as ledger:
            a = next(x for x in ledger[BATCH_LEDGER] if x['id'] == bid)
            a.update(status='FORMAT_VALID' if valid else 'FAILED_NO_RETRY', ended_epoch=time.time(), transport_failure=operation.get('transport_failure', False))
            if not valid:
                ledger['Judge_failed_payloads'] = ledger.get('Judge_failed_payloads', 0) + len(rows)
        write(ROOT / 'STATUS.json', status(db))
        return dict(status='FORMAT_VALID' if valid else 'MISSING', items=len(rows))
    if operation['action'] == 'status':
        return status(db)
    if operation['action'] == 'reserved':
        return [a for a in read(RUN / 'RESOURCE_LEDGER.json').get(BATCH_LEDGER, []) if a['status'] == 'RESERVED']
    raise ValueError('Unknown action')


def main():
    db = connect(); initialize(db)
    operation = json.loads(os.environ['QUEUE_REQUEST'])
    if operation['action'] == 'selfcheck':
        test = sqlite3.connect(':memory:'); test.row_factory = sqlite3.Row
        test.execute('CREATE TABLE payload (key TEXT PRIMARY KEY,record TEXT,binding TEXT,status TEXT,batch TEXT,correct INTEGER)')
        q = dict(query_id='synthetic-boundary', question='synthetic-boundary', reference='synthetic-boundary', image_sha256='image-A')
        b = dict(question=q['question'], reference=q['reference'], image_sha256=q['image_sha256'], image_path='synthetic-A', prompt_ids=[1], attention_mask=[1], runtime={'binding':'runtime-A'}, generation={'max_new_tokens':1024})
        out = dict(raw_answer='same decoded text', raw_token_ids=[1])
        first = payload(test, q, b, out)
        assert first == payload(test, q, b, out), 'Exact payload must share one key'
        assert first != payload(test, q, b, dict(out, raw_token_ids=[2])), 'Token difference must not share by text'
        assert first != payload(test, dict(q, image_sha256='image-B'), dict(b, image_sha256='image-B'), out), 'Image difference must not share by text'
        assert first != payload(test, q, dict(b, prompt_ids=[2]), out), 'Prompt difference must not share'
        assert first != payload(test, q, dict(b, runtime={'binding':'runtime-B'}), out), 'Runtime difference must not share'
        assert test.execute('SELECT count(*) FROM payload').fetchone()[0] == 5
        answer = dict(status='PASS', checks=['exact full identity shares', 'different tokens/image/prompt/runtime do not share by text'], new_Judge_requests=0)
    else:
        answer = request(db, operation)
    print(json.dumps(answer, ensure_ascii=False))


if __name__ == '__main__':
    main()
