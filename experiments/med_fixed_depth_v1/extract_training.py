"""Extract only native/FIT fields for the locked, role-checked owners."""
import json
from pathlib import Path
import stream_extract as s
import index


def decode_field(f, token):
    start = f.tell() - 1
    index.skip(f, token)
    end = f.tell()
    if end - start > 1_048_576:
        raise ValueError('Training field exceeds size limit')
    f.seek(start)
    value = json.loads(f.read(end - start))
    f.seek(end)
    return value


def task_at(f, locked):
    start, end = locked['_span']
    f.seek(start)
    if s.nonspace(f) != b'{':
        raise ValueError('Task object changed')
    wanted = {'order', 'edit_id', 'seed', 'native', 'fit_questions'}
    result = {}
    while True:
        token = s.nonspace(f)
        if token == b'}':
            break
        if token != b'"':
            raise ValueError('Task field expected')
        key = s.string(f, True)
        if s.nonspace(f) != b':':
            raise ValueError('Task colon expected')
        token = s.nonspace(f)
        if key in wanted:
            if key in result:
                raise ValueError('Duplicate training field')
            result[key] = decode_field(f, token)
        else:
            index.skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            break
        if token != b',':
            raise ValueError('Task separator expected')
    if f.tell() != end or set(result) != wanted or result['order'] != locked['order']:
        raise ValueError('Task identity or span changed')
    native = result['native']
    actual_group = index.group(native.get('source_group') or native.get('image_path'))
    locked_group = index.group(locked.get('native', {}).get('source_group') or locked.get('native', {}).get('image_path'))
    if not actual_group or actual_group != locked_group:
        raise ValueError('Source group changed')
    if len(result['fit_questions']) != 4 or not all(native.get(k) for k in ('question', 'reference', 'image_path')):
        raise ValueError('Incomplete native/FIT fields')
    rows = [dict(native, query_id=f"edit_{result['order']}_{i}", question=q,
                 audit_role='NATIVE' if i == 0 else 'FIT')
            for i, q in enumerate([native['question']] + result['fit_questions'])]
    return {'task': result, 'source_rows': rows}


def source_rows(path, allowed):
    selected = {}
    with Path(path).open('rb') as f:
        if s.nonspace(f) != b'{':
            raise ValueError('Expected root')
        while True:
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b'"':
                raise ValueError('Expected root field')
            key = s.string(f, True)
            if s.nonspace(f) != b':':
                raise ValueError('Expected root colon')
            token = s.nonspace(f)
            if key == 'source_rows':
                if token != b'{':
                    raise ValueError('Expected source row map')
                while True:
                    token = s.nonspace(f)
                    if token == b'}':
                        break
                    if token != b'"':
                        raise ValueError('Expected owner key')
                    owner = s.string(f, True)
                    if s.nonspace(f) != b':':
                        raise ValueError('Expected source row colon')
                    token = s.nonspace(f)
                    if owner in allowed:
                        if owner in selected or token != b'[':
                            raise ValueError('Invalid selected source row')
                        selected[owner] = decode_field(f, token)
                    else:
                        index.skip(f, token)
                    token = s.nonspace(f)
                    if token == b'}':
                        break
                    if token != b',':
                        raise ValueError('Expected source row delimiter')
            else:
                index.skip(f, token)
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b',':
                raise ValueError('Expected root delimiter')
    if set(selected) != set(allowed):
        raise ValueError('Selected original source rows missing')
    return selected


def extract(path, lock, admission):
    if lock.get('status') != 'BASE_ELIGIBILITY_ONLY_NOT_FULL_ROLE_ADMISSION':
        raise ValueError('Expected the exact fixed identity lock')
    tasks = lock['index']['tasks']
    if len(tasks) != 16 or len({t['order'] for t in tasks}) != 16:
        raise ValueError('Frozen identity list incomplete')
    owners = [task['order'] for task in tasks]
    if admission.get('status') != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or admission.get('owners') != owners:
        raise ValueError('Full, ordered source and evaluation role admission required')
    with Path(path).open('rb') as f:
        result = [task_at(f, task) for task in tasks]
    original = source_rows(path, {str(owner) for owner in owners})
    for entry in result:
        task = entry['task']
        rows = original[str(task['order'])]
        if len(rows) != 5 or [row['question'] for row in rows] != [task['native']['question']] + task['fit_questions']:
            raise ValueError('Original source/FIT questions differ')
        if [row['audit_role'] for row in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
            raise ValueError('Original source/FIT roles differ')
        expected_group = index.group(task['native'].get('source_group') or task['native'].get('image_path'))
        if any(index.group(row.get('source_group') or row.get('image_path')) != expected_group for row in rows):
            raise ValueError('Original source group differs')
        entry['source_rows'] = rows
    return result
