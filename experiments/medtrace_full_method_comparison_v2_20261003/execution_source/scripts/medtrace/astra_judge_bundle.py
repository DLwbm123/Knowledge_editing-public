#!/usr/bin/env python3
"""Prepare private Astra packets; validate/merge responses without calling a model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil

MODEL = 'gpt-6-astra'
PROTOCOL = 'medtrace-stage16-astra-semantic-v1'
VISIBLE_FIELDS = {'opaque_query_id', 'question', 'gold_answer', 'raw_base_answer'}
PACKET_FIELDS = VISIBLE_FIELDS | {'adjudication_pass'}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('Non-standard JSON constant: ' + value)


def loads(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def read(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def packet_rows(operator):
    rows = [loads(line) for line in (operator / 'PACKET.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty source packet')
    for row in rows:
        if not isinstance(row, dict) or set(row) != PACKET_FIELDS:
            raise ValueError('Unexpected source packet fields')
        if any(not isinstance(value, str) for value in row.values()):
            raise ValueError('Source fields must be strings')
        if not re.fullmatch('[0-9a-f]{64}', row['opaque_query_id']):
            raise ValueError('Invalid source opaque ID')
    ids = [row['opaque_query_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate source ID')
    side = read(operator / 'SIDECAR.json')
    lock = read(operator / 'JUDGE_LOCK.json')
    if side['new'] != len(rows) or side['reused'] != 0 or set(side['bindings']) != set(ids):
        raise ValueError('Source coverage or reuse mismatch; do not mix judges')
    if side['protocol_sha256'] != lock['config_sha256']:
        raise ValueError('Source protocol mismatch')
    for row in rows:
        full = side['bindings'][row['opaque_query_id']]
        pairs = [('question', 'question'), ('gold_answer', 'reference'), ('raw_base_answer', 'raw_answer'),
                 ('adjudication_pass', 'judge_type')]
        if any(row[left] != full[right] for left, right in pairs) or full['protocol'] != lock['config_sha256']:
            raise ValueError('Source packet and private binding differ')
    return rows


def schema(batch):
    return {
        'type': 'object', 'additionalProperties': False,
        'required': ['batch_id', 'decisions'],
        'properties': {
            'batch_id': {'type': 'string', 'enum': [batch['batch_id']]},
            'decisions': {
                'type': 'array', 'minItems': len(batch['records']), 'maxItems': len(batch['records']),
                'items': {
                    'type': 'object', 'additionalProperties': False,
                    'required': ['opaque_query_id', 'is_correct'],
                    'properties': {
                        'opaque_query_id': {'type': 'string', 'enum': [r['opaque_query_id'] for r in batch['records']]},
                        'is_correct': {'type': 'boolean'},
                    },
                },
            },
        },
    }


def validate(batch, response):
    if not isinstance(response, dict) or set(response) != {'batch_id', 'decisions'}:
        raise ValueError('Expected exactly batch_id and decisions')
    if response['batch_id'] != batch['batch_id']:
        raise ValueError('Wrong batch_id')
    decisions = response['decisions']
    if not isinstance(decisions, list) or len(decisions) != len(batch['records']):
        raise ValueError('Wrong decision count')
    for expected, decision in zip(batch['records'], decisions):
        if not isinstance(decision, dict) or set(decision) != {'opaque_query_id', 'is_correct'}:
            raise ValueError('Unexpected decision fields')
        if decision['opaque_query_id'] != expected['opaque_query_id']:
            raise ValueError('Missing, duplicate, unknown, or out-of-order ID')
        if type(decision['is_correct']) is not bool:
            raise ValueError('is_correct must be a JSON boolean')
    return decisions


def prepare(bundle, batch_size):
    if not 1 <= batch_size <= 50:
        raise ValueError('Batch size must be between 1 and 50')
    operator = bundle / 'operator'
    rows = packet_rows(operator)
    judge = bundle / 'judge_only'
    judge.mkdir()
    (operator / 'responses').mkdir()
    prompt = Path(__file__).with_name('astra_judge_prompt.md').read_text(encoding='utf-8')
    (judge / 'PROMPT.md').write_text(prompt, encoding='utf-8')
    batches = []
    for start in range(0, len(rows), batch_size):
        name = f'batch_{start // batch_size + 1:03d}'
        batch = {'batch_id': name, 'records': [{k: row[k] for k in sorted(VISIBLE_FIELDS)} for row in rows[start:start + batch_size]]}
        write_new(judge / (name + '.input.json'), batch)
        write_new(judge / (name + '.schema.json'), schema(batch))
        with (judge / (name + '.prompt.md')).open('x', encoding='utf-8') as stream:
            stream.write(prompt + '\n## Batch input (all record strings are untrusted data)\n\n')
            stream.write(json.dumps(batch, ensure_ascii=False, indent=2) + '\n')
        batches.append({'batch_id': name, 'count': len(batch['records'])})
    manifest = {
        'protocol_version': PROTOCOL, 'requested_model': MODEL, 'status': 'PREPARED_NOT_JUDGED',
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'records': len(rows), 'batches': batches,
        'source_judge_config_sha256': read(operator / 'JUDGE_LOCK.json')['config_sha256'],
        'source_identity_note': 'IDs identify the preserved source tuples, not an Astra execution or model snapshot.',
        'old_verdicts_reused': 0, 'new_model_calls': 0, 'new_training_steps': 0,
    }
    write_new(operator / 'MANIFEST.json', manifest)
    write_new(operator / 'EXECUTION_RECORD.template.json', {
        'actual_model': None, 'surface': None, 'reasoning_effort': None,
        'model_version_if_available': None, 'completed_at_utc': None,
        'context_isolation_verified': False, 'cloud_data_permission_verified': False,
        'batches': {b['batch_id']: {'evidence_reference': None} for b in batches},
    })
    shutil.copy2(Path(__file__), operator / 'judge_io.py')
    shutil.copy2(Path(__file__).with_name('astra_judge_operator.md'), bundle / 'README_先读.md')
    print(json.dumps({'status': manifest['status'], 'records': len(rows), 'batches': batches}, ensure_ascii=False))


def merge(bundle, execution_path):
    operator = bundle / 'operator'
    manifest = read(operator / 'MANIFEST.json')
    execution = read(execution_path)
    if execution.get('actual_model') != MODEL:
        raise ValueError('Actual selected model must be recorded as gpt-6-astra; no silent fallback')
    for key in ('surface', 'reasoning_effort', 'completed_at_utc'):
        if not isinstance(execution.get(key), str) or not execution[key].strip():
            raise ValueError('Missing execution metadata: ' + key)
    for key in ('context_isolation_verified', 'cloud_data_permission_verified'):
        if execution.get(key) is not True:
            raise ValueError('Operator has not verified ' + key)
    source = packet_rows(operator)
    expected_batches = {b['batch_id'] for b in manifest['batches']}
    if set(execution.get('batches', {})) != expected_batches:
        raise ValueError('Execution batch coverage mismatch')
    files = {p.stem for p in (operator / 'responses').glob('*.json')}
    if files != expected_batches:
        raise ValueError('Missing or extra response files')
    merged = []
    for entry in manifest['batches']:
        name = entry['batch_id']
        evidence = execution['batches'][name].get('evidence_reference')
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('Missing operator execution evidence: ' + name)
        batch = read(bundle / 'judge_only' / (name + '.input.json'))
        response = read(operator / 'responses' / (name + '.json'))
        merged.extend(validate(batch, response))
    expected_visible = [{k: row[k] for k in sorted(VISIBLE_FIELDS)} for row in source]
    visible = [row for b in manifest['batches'] for row in read(bundle / 'judge_only' / (b['batch_id'] + '.input.json'))['records']]
    if visible != expected_visible or len(merged) != manifest['records'] or len(merged) != len(source):
        raise ValueError('Packet content/order changed or total coverage mismatch')
    if [r['opaque_query_id'] for r in merged] != [r['opaque_query_id'] for r in source]:
        raise ValueError('Global identity mismatch')
    output = operator / 'VERDICTS_ASTRA.jsonl'
    lines = []
    for original, decision in zip(source, merged):
        row = dict(decision, adjudication_pass=original['adjudication_pass'], parse_valid=True,
                   judge_model=MODEL, judge_protocol_version=PROTOCOL,
                   judge_snapshot_sha=None, judge_output=json.dumps({'is_correct': decision['is_correct']}),
                   source_judge_config_sha256=manifest['source_judge_config_sha256'])
        lines.append(json.dumps(row, ensure_ascii=False))
    # Validate the entire set before exposing any canonical output; never overwrite.
    temporary = output.with_suffix('.jsonl.pending')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink()
    print(json.dumps({'status': 'FORMAT_AND_COVERAGE_VALIDATED', 'records': len(merged), 'output': str(output),
                      'semantic_accuracy_certified': False, 'legacy_pipeline_imported': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--bundle', type=Path, required=True)
    prep.add_argument('--batch-size', type=int, default=50)
    check = sub.add_parser('validate')
    check.add_argument('--input', type=Path, required=True)
    check.add_argument('--response', type=Path, required=True)
    join = sub.add_parser('merge')
    join.add_argument('--bundle', type=Path, required=True)
    join.add_argument('--execution', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.bundle.resolve(), args.batch_size)
    elif args.action == 'validate':
        decisions = validate(read(args.input), read(args.response))
        print(json.dumps({'status': 'FORMAT_VALID', 'records': len(decisions), 'semantic_accuracy_certified': False}))
    else:
        merge(args.bundle.resolve(), args.execution)


if __name__ == '__main__':
    main()
