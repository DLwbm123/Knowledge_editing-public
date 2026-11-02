"""Freeze training identities without decoding any question, answer or score."""
import json
import re
from pathlib import Path
import index
import stream_extract as stream


def group(v):
    match = re.search(r'(?i)(xmlab[0-9]+)', str(v))
    return match.group(1).lower() if match else None


def tasks(path):
    result = None
    with Path(path).open('rb') as f:
        if stream.nonspace(f) != b'{':
            raise ValueError('root object expected')
        while True:
            t = stream.nonspace(f)
            if t == b'}':
                break
            if t != b'"':
                raise ValueError('root key expected')
            key = stream.string(f, True)
            if stream.nonspace(f) != b':':
                raise ValueError('root colon expected')
            t = stream.nonspace(f)
            if key == 'tasks':
                if result is not None or t != b'[':
                    raise ValueError('invalid tasks')
                result = index.array(f, 'task')
            else:
                index.skip(f, t)
            t = stream.nonspace(f)
            if t == b'}':
                break
            if t != b',':
                raise ValueError('root separator expected')
    if result is None or [t['order'] for t in result] != list(range(1, 147)):
        raise ValueError('original cohort identity mismatch')
    return result


def replay_groups(path, expected):
    values = []
    with Path(path).open('rb') as f:
        if stream.nonspace(f) != b'[':
            raise ValueError('replay array expected')
        while True:
            t = stream.nonspace(f)
            if t == b']':
                break
            if t != b'{':
                raise ValueError('replay object expected')
            source = None
            while True:
                t = stream.nonspace(f)
                if t == b'}':
                    break
                if t != b'"':
                    raise ValueError('replay key expected')
                key = stream.string(f, True)
                if stream.nonspace(f) != b':':
                    raise ValueError('replay colon expected')
                t = stream.nonspace(f)
                if key == 'source_group':
                    if source is not None or t != b'"':
                        raise ValueError('duplicate replay group')
                    source = group(stream.string(f, True))
                else:
                    index.skip(f, t)
                t = stream.nonspace(f)
                if t == b'}':
                    break
                if t != b',':
                    raise ValueError('replay separator expected')
            if source is None:
                raise ValueError('unknown replay group')
            values.append(source)
            t = stream.nonspace(f)
            if t == b']':
                break
            if t != b',':
                raise ValueError('replay array separator expected')
    result = set(values)
    if len(result) != expected:
        raise ValueError('replay identity count mismatch')
    return result


def build(original, data_receipt, fixed_receipt, additions_receipt, deny, v106, replay):
    if data_receipt.get('status') != 'PASS' or data_receipt.get('owners') != 146:
        raise ValueError('original training admission missing')
    if fixed_receipt.get('status') != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED':
        raise ValueError('fixed source admission missing')
    fixed = set(fixed_receipt['owners'])
    if len(fixed) != 16:
        raise ValueError('fixed owner coverage mismatch')
    if additions_receipt.get('status') != 'MECHANICAL_SOURCE_AND_ROLE_PASS_WITH_SEMANTIC_LIMIT' or additions_receipt.get('native_sources') != 4:
        raise ValueError('four additions admission missing')
    if len(deny['V95']) != 24 or len(deny['V111']) != 15 or len(v106) != 8:
        raise ValueError('protected identity snapshot incomplete')
    sets = [set(map(group, deny['V95'])), set(map(group, deny['V111'])), set(map(group, v106)),
            replay_groups(Path(replay) / 'REPLAY_CHECK.json', 32), replay_groups(Path(replay) / 'REPLAY_FIT.json', 64)]
    if any(None in s for s in sets):
        raise ValueError('unknown protected group')
    blocked = set().union(*sets)
    chosen, unknown, overlaps = [], [], 0
    for task in tasks(original):
        native = task.get('native', {})
        value = native.get('source_group') or native.get('image_path')
        g = group(value)
        if g:
            if g in blocked:
                overlaps += 1
                unknown.append(task['order'])
            else:
                chosen.append(task)
        elif task['order'] in fixed and re.search(r'vqa[-_]?rad', str(value), re.I):
            chosen.append(task)
        else:
            unknown.append(task['order'])
    if len(chosen) + len(unknown) != 146 or not fixed.issubset({t['order'] for t in chosen}):
        raise ValueError('fixed identity or cohort coverage mismatch')
    return {'status': 'SOURCE_IDENTITIES_ADMITTED', 'tasks': chosen, 'unknown_owners': unknown,
            'protected_group_overlaps': overlaps, 'qa_decoded': 0, 'original_owners': 146,
            'additional_owners': [147, 148, 149, 150], 'unknown_not_extracted': True}
