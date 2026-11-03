"""Read consumer identities from historical packets without decoding payloads."""
import prior_binding as pb
import stream_extract as stream


def consumers(path):
    result = {}
    with open(path, 'rb') as f:
        if stream.nonspace(f) != b'{':
            raise ValueError('packet object expected')
        found = False
        while True:
            token = stream.nonspace(f)
            if token == b'}':
                break
            if token != b'"':
                raise ValueError('packet field expected')
            name = stream.string(f, True)
            if stream.nonspace(f) != b':':
                raise ValueError('colon expected')
            token = stream.nonspace(f)
            if name != 'consumers':
                pb.skip(f, token)
            else:
                if found or token != b'[':
                    raise ValueError('invalid consumer array')
                found = True
                while True:
                    token = stream.nonspace(f)
                    if token == b']':
                        break
                    fields = {k: True for k in ('consumer_id', 'query_id', 'arm', 'key', 'owner', 'role', 'route', 'candidate_expert')}
                    row = pb.select(f, token, fields)
                    key = (row['consumer_id'], row['arm'])
                    if key in result:
                        raise ValueError('duplicate consumer')
                    result[key] = row
                    token = stream.nonspace(f)
                    if token == b']':
                        break
                    if token != b',':
                        raise ValueError('array delimiter expected')
            token = stream.nonspace(f)
            if token == b'}':
                break
            if token != b',':
                raise ValueError('object delimiter expected')
    if not found or len(result) != 4296:
        raise ValueError('historical consumer coverage differs')
    return result


def common(d, du):
    if set(d) != set(du):
        raise ValueError('historical consumer sets differ')
    chosen = []
    for key, row in d.items():
        other = du[key]
        if any(row[k] != other[k] for k in ('consumer_id', 'query_id', 'arm', 'owner', 'role', 'route')):
            raise ValueError('historical event identity differs')
        if row['arm'] == 'MED' and row['key'] != other['key']:
            raise ValueError('historical MED score identity differs')
        if row['arm'] == 'MED_D' and row['candidate_expert']:
            if not other['candidate_expert']:
                raise ValueError('D expert missing from DU')
            chosen.append(row['consumer_id'])
    if len(chosen) != 392 or len(set(chosen)) != 392:
        raise ValueError('routed cohort differs')
    return chosen
