"""Select original score keys and decisions without decoding unrelated QA or labels."""
import json
import stream_extract as s


def skip(f, token):
    if token == b'"':
        s.string(f)
    elif token in (b'{', b'['):
        s.skip_compound(f, token)
    else:
        s.scalar(f, token)


def scalar(f, token):
    start = f.tell() - 1
    skip(f, token)
    end = f.tell()
    if end - start > 32:
        raise ValueError('Scalar too long')
    f.seek(start)
    value = json.loads(f.read(end - start))
    f.seek(end)
    return value


def fields(f, wanted):
    out = {}
    while True:
        token = s.nonspace(f)
        if token == b'}':
            return out
        if token != b'"':
            raise ValueError('Expected field')
        key = s.string(f, True)
        if s.nonspace(f) != b':':
            raise ValueError('Expected colon')
        token = s.nonspace(f)
        if key in wanted:
            if key in out:
                raise ValueError('Duplicate field')
            out[key] = s.string(f, True) if token == b'"' else scalar(f, token)
        else:
            skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            return out
        if token != b',':
            raise ValueError('Expected delimiter')


def consumers(path, allowed):
    wanted = {'arm', 'mode', 'prefix', 'owner', 'query_id', 'payload_key'}
    result = {}
    with open(path, 'rb') as f:
        if s.nonspace(f) != b'{':
            raise ValueError('Expected root')
        while True:
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b'"':
                raise ValueError('Expected root field')
            name = s.string(f, True)
            if s.nonspace(f) != b':':
                raise ValueError('Expected root colon')
            token = s.nonspace(f)
            if name == 'consumers':
                if token != b'[':
                    raise ValueError('Expected consumers array')
                while True:
                    token = s.nonspace(f)
                    if token == b']':
                        break
                    if token != b'{':
                        raise ValueError('Expected consumer object')
                    row = fields(f, wanted)
                    key = (row.get('owner'), row.get('query_id'))
                    if row.get('mode') == 'single' and row.get('arm') in ('MED', 'BASE') and key in allowed:
                        pair = (row['arm'], *key)
                        if pair in result or not row.get('payload_key'):
                            raise ValueError('Duplicate or missing payload identity')
                        result[pair] = row['payload_key']
                    token = s.nonspace(f)
                    if token == b']':
                        break
                    if token != b',':
                        raise ValueError('Expected consumer delimiter')
            else:
                skip(f, token)
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b',':
                raise ValueError('Expected root delimiter')
    if len(result) != 2 * len(allowed):
        raise ValueError('MED/Base score-key coverage incomplete')
    return result


def decisions(path, allowed):
    wanted = {'opaque_query_id', 'core_correct', 'full_supported', 'unsupported_expansion', 'extra_contradiction'}
    result = {}
    with open(path, 'rb') as f:
        if s.nonspace(f) != b'{':
            raise ValueError('Expected root')
        while True:
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b'"':
                raise ValueError('Expected root field')
            name = s.string(f, True)
            if s.nonspace(f) != b':':
                raise ValueError('Expected colon')
            token = s.nonspace(f)
            if name == 'decisions':
                if token != b'{':
                    raise ValueError('Expected decisions object')
                while True:
                    token = s.nonspace(f)
                    if token == b'}':
                        break
                    if token != b'"':
                        raise ValueError('Expected decision identity')
                    key = s.string(f, True)
                    if s.nonspace(f) != b':':
                        raise ValueError('Expected decision colon')
                    token = s.nonspace(f)
                    if key in allowed:
                        if key in result or token != b'{':
                            raise ValueError('Invalid selected decision')
                        result[key] = fields(f, wanted)
                    else:
                        skip(f, token)
                    token = s.nonspace(f)
                    if token == b'}':
                        break
                    if token != b',':
                        raise ValueError('Expected decision delimiter')
            else:
                skip(f, token)
            token = s.nonspace(f)
            if token == b'}':
                break
            if token != b',':
                raise ValueError('Expected root delimiter')
    if set(result) != set(allowed) or any(set(v) != wanted for v in result.values()):
        raise ValueError('Selected original decisions incomplete')
    for key, row in result.items():
        if row['opaque_query_id'] != key or any(type(row[k]) is not bool for k in wanted - {'opaque_query_id'}):
            raise ValueError('Selected original decision is invalid')
        if row['full_supported'] and (not row['core_correct'] or row['unsupported_expansion'] or row['extra_contradiction']):
            raise ValueError('Selected original FULL decision is inconsistent')
    return result
