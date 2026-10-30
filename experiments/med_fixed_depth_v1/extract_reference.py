"""Read only the fixed, admitted DEV reference after all predictions are frozen."""
from pathlib import Path
import stream_extract as s
import index
from extract_training import decode_field


def query_ref(f):
    result = {}
    while True:
        token = s.nonspace(f)
        if token == b'}':
            break
        if token != b'"':
            raise ValueError('Expected query field')
        key = s.string(f, True)
        if s.nonspace(f) != b':':
            raise ValueError('Expected colon')
        token = s.nonspace(f)
        if key in ('query_id', 'reference'):
            if key in result:
                raise ValueError('Duplicate reference field')
            result[key] = decode_field(f, token)
        else:
            index.skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            break
        if token != b',':
            raise ValueError('Expected query delimiter')
    if set(result) != {'query_id', 'reference'}:
        raise ValueError('Reference missing')
    return result


def plan_ref(f, locked):
    start, end = locked['_span']
    f.seek(start)
    if s.nonspace(f) != b'{':
        raise ValueError('Plan identity changed')
    row = {}
    while True:
        token = s.nonspace(f)
        if token == b'}':
            break
        if token != b'"':
            raise ValueError('Expected plan field')
        key = s.string(f, True)
        if s.nonspace(f) != b':':
            raise ValueError('Expected plan colon')
        token = s.nonspace(f)
        if key in ('method', 'mode', 'prefix', 'owner'):
            if key in row:
                raise ValueError('Duplicate plan field')
            row[key] = index.value(f, token)
        elif key == 'query':
            if token != b'{' or key in row:
                raise ValueError('Invalid query')
            row[key] = query_ref(f)
        else:
            index.skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            break
        if token != b',':
            raise ValueError('Expected plan delimiter')
    if f.tell() != end or any(row.get(k) != locked.get(k) for k in ('method', 'mode', 'prefix', 'owner')):
        raise ValueError('Plan identity changed')
    if row['query']['query_id'] != locked['query']['query_id']:
        raise ValueError('Reference query identity changed')
    return {'owner': row['owner'], **row['query']}


def extract(path, lock, admission, completion):
    if lock.get('status') != 'BASE_ELIGIBILITY_ONLY_NOT_FULL_ROLE_ADMISSION':
        raise ValueError('Expected fixed identity lock')
    if admission.get('status') != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or admission.get('owners') != lock['index']['owners']:
        raise ValueError('Role admission mismatch')
    if completion.get('status') != 'ALL_PREDICTIONS_FROZEN' or completion.get('events') != len(lock['index']['plan']):
        raise ValueError('All predictions must be frozen before reading references')
    with Path(path).open('rb') as f:
        rows = [plan_ref(f, row) for row in lock['index']['plan']]
    if len({(row['owner'], row['query_id']) for row in rows}) != len(rows):
        raise ValueError('Duplicate reference consumer')
    return rows
