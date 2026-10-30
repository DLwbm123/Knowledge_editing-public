"""Read only permitted question/image fields after the fixed role admission."""
import json
from pathlib import Path
import stream_extract as s
import index
from extract_training import decode_field


def query_object(f):
    wanted = {'query_id', 'question', 'image_path', 'image_sha256'}
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
        if key in wanted:
            if key in result:
                raise ValueError('Duplicate query field')
            result[key] = decode_field(f, token)
        else:
            index.skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            break
        if token != b',':
            raise ValueError('Expected query delimiter')
    if set(result) != wanted or any(not result[k] for k in wanted):
        raise ValueError('Incomplete query input')
    return result


def plan_at(f, locked):
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
            raise ValueError('Expected colon')
        token = s.nonspace(f)
        if key in ('method', 'mode', 'prefix', 'owner'):
            if key in row:
                raise ValueError('Duplicate plan field')
            row[key] = index.value(f, token)
        elif key == 'query':
            if token != b'{' or key in row:
                raise ValueError('Invalid plan query')
            row[key] = query_object(f)
        else:
            index.skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            break
        if token != b',':
            raise ValueError('Expected plan delimiter')
    if f.tell() != end or any(row.get(k) != locked.get(k) for k in ('method', 'mode', 'prefix', 'owner')):
        raise ValueError('Plan identity or span changed')
    query, prior = row['query'], locked['query']
    if query['query_id'] != prior.get('query_id') or query['image_path'] != prior.get('image_path'):
        raise ValueError('Query identity changed')
    return row


def extract(path, lock, admission):
    if lock.get('status') != 'BASE_ELIGIBILITY_ONLY_NOT_FULL_ROLE_ADMISSION':
        raise ValueError('Expected fixed eligibility identity lock')
    owners = lock['index']['owners']
    if admission.get('status') != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or admission.get('owners') != owners:
        raise ValueError('Full source and evaluation role admission required')
    with Path(path).open('rb') as f:
        return [plan_at(f, row) for row in lock['index']['plan']]
