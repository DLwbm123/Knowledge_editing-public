"""Read batch and opaque payload identities without decoding historical QA."""
import re
from pathlib import Path
import stream_extract as s

HEX = re.compile(r'^[0-9a-f]{64}$')


def skip(f, token):
    if token == b'"':
        s.string(f)
    elif token in (b'{', b'['):
        s.skip_compound(f, token)
    else:
        s.scalar(f, token)


def object_fields(f, handlers):
    seen = set()
    while True:
        token = s.nonspace(f)
        if token == b'}':
            return
        if token != b'"':
            raise ValueError('Object field expected')
        key = s.string(f, True)
        if s.nonspace(f) != b':':
            raise ValueError('Colon expected')
        token = s.nonspace(f)
        if key in handlers:
            if key in seen:
                raise ValueError('Duplicate selected field')
            seen.add(key)
            handlers[key](f, token)
        else:
            skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            return
        if token != b',':
            raise ValueError('Object delimiter expected')


def array(f, token, consume):
    if token != b'[':
        raise ValueError('Array expected')
    while True:
        token = s.nonspace(f)
        if token == b']':
            return
        if token != b'{':
            raise ValueError('Array record must be object')
        consume(f)
        token = s.nonspace(f)
        if token == b']':
            return
        if token != b',':
            raise ValueError('Array delimiter expected')


def manifest(path):
    batches = []
    def batch(f):
        state = {'ids': []}
        def string_field(name):
            def read(f, token):
                if token != b'"':
                    raise ValueError('Identity must be a string')
                value = s.string(f, True)
                if not HEX.fullmatch(value):
                    raise ValueError('Invalid opaque identity')
                state[name] = value
            return read
        def record(f):
            item = {}
            def oid(f, token):
                if token != b'"':
                    raise ValueError('Payload identity must be string')
                value = s.string(f, True)
                if not HEX.fullmatch(value):
                    raise ValueError('Invalid payload identity')
                item['id'] = value
            object_fields(f, {'opaque_query_id': oid})
            if set(item) != {'id'}:
                raise ValueError('Missing payload identity')
            state['ids'].append(item['id'])
        object_fields(f, {'batch_id': string_field('id'), 'records': lambda f, t: array(f, t, record)})
        if 'id' not in state or not state['ids'] or len(set(state['ids'])) != len(state['ids']):
            raise ValueError('Invalid batch coverage')
        batches.append(state)
    with Path(path).open('rb') as f:
        if s.nonspace(f) != b'{':
            raise ValueError('Manifest object expected')
        object_fields(f, {'batches': lambda f, t: array(f, t, batch)})
        if f.read().strip():
            raise ValueError('Trailing manifest content')
    if len({x['id'] for x in batches}) != len(batches):
        raise ValueError('Duplicate batch identity')
    return batches


def attempted(root, batches):
    """Conservatively reserve every manifested payload, including uncertain requests.

    Historical runners did not all create ATTEMPT markers. A response or a
    preflight receipt alone cannot establish that no request occurred. Treating
    all manifest entries as attempted can leave some untried entries UNKNOWN,
    but never turns missing evidence into permission for a retry.
    """
    root = Path(root)
    result = set()
    for batch in batches:
        evidence = root / 'batches' / batch['id'] / 'operator'
        if not evidence.is_dir():
            raise ValueError('Missing batch evidence directory')
        result.update(batch['ids'])
    return result
