"""Select native/FIT records only after a separately frozen identity admission."""
import json
from pathlib import Path
import index
import stream_extract as stream
import extract_training


def _select(f, allowed):
    out = {}
    while True:
        token = stream.nonspace(f)
        if token == b'}':
            return out
        if token != b'"':
            raise ValueError('object key expected')
        key = stream.string(f, True)
        if stream.nonspace(f) != b':':
            raise ValueError('colon expected')
        token = stream.nonspace(f)
        if key in allowed:
            if key in out:
                raise ValueError('duplicate selected key')
            out[key] = extract_training.decode_field(f, token)
        else:
            index.skip(f, token)
        token = stream.nonspace(f)
        if token == b'}':
            return out
        if token != b',':
            raise ValueError('object separator expected')


def extra_rows(path):
    result = []
    with Path(path).open('rb') as f:
        if stream.nonspace(f) != b'{':
            raise ValueError('root object expected')
        while True:
            token = stream.nonspace(f)
            if token == b'}':
                break
            if token != b'"':
                raise ValueError('root key expected')
            key = stream.string(f, True)
            if stream.nonspace(f) != b':':
                raise ValueError('root colon expected')
            token = stream.nonspace(f)
            if key == 'tasks':
                if token != b'[':
                    raise ValueError('tasks array expected')
                while True:
                    token = stream.nonspace(f)
                    if token == b']':
                        break
                    if token != b'{':
                        raise ValueError('task object expected')
                    record = {}
                    while True:
                        token = stream.nonspace(f)
                        if token == b'}':
                            break
                        if token != b'"':
                            raise ValueError('task key expected')
                        k = stream.string(f, True)
                        if stream.nonspace(f) != b':':
                            raise ValueError('task colon expected')
                        token = stream.nonspace(f)
                        if k == 'task':
                            if token != b'{':
                                raise ValueError('nested task expected')
                            record[k] = _select(f, {'order', 'edit_id', 'seed', 'native', 'fit_questions'})
                        elif k == 'source_rows':
                            record[k] = extract_training.decode_field(f, token)
                        else:
                            index.skip(f, token)
                        token = stream.nonspace(f)
                        if token == b'}':
                            break
                        if token != b',':
                            raise ValueError('task separator expected')
                    if set(record) != {'task', 'source_rows'}:
                        raise ValueError('missing source fields')
                    result.append(record)
                    token = stream.nonspace(f)
                    if token == b']':
                        break
                    if token != b',':
                        raise ValueError('tasks separator expected')
            else:
                index.skip(f, token)
            token = stream.nonspace(f)
            if token == b'}':
                break
            if token != b',':
                raise ValueError('root separator expected')
    if [x['task']['order'] for x in result] != [147, 148, 149, 150]:
        raise ValueError('additional owner sequence mismatch')
    for entry in result:
        task, rows = entry['task'], entry['source_rows']
        if len(rows) != 5 or [r['audit_role'] for r in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
            raise ValueError('training role mismatch')
        if [r['question'] for r in rows] != [task['native']['question']] + task['fit_questions']:
            raise ValueError('training input mismatch')
    return result


def original_rows(path, frozen):
    if frozen.get('status') != 'SOURCE_IDENTITIES_ADMITTED' or frozen.get('qa_decoded') != 0:
        raise ValueError('source identity admission missing')
    tasks = frozen['tasks']
    owners = [x['order'] for x in tasks]
    if not owners or len(owners) != len(set(owners)) or any(not 1 <= x <= 146 for x in owners):
        raise ValueError('invalid admitted owners')
    with Path(path).open('rb') as f:
        result = [extract_training.task_at(f, task) for task in tasks]
    old = extract_training.source_rows(path, {str(owner) for owner in owners})
    for entry in result:
        task = entry['task']
        rows = old[str(task['order'])]
        if len(rows) != 5 or [r['audit_role'] for r in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
            raise ValueError('original training role mismatch')
        if [r['question'] for r in rows] != [task['native']['question']] + task['fit_questions']:
            raise ValueError('original training input mismatch')
        group = index.group(task['native'].get('source_group') or task['native'].get('image_path'))
        if any(index.group(row.get('source_group') or row.get('image_path')) != group for row in rows):
            raise ValueError('original source group changed')
        entry['source_rows'] = rows
    return result
