"""Read only locked records from a JSON array nested in an object.

The identity pass skips all other fields, including question and answer text.
The content pass decodes only the byte ranges that matched the locked IDs.
"""
import json
import prior_binding as pb
import stream_extract as stream


def identities(path, array_name, fields):
    result = []
    with open(path, 'rb') as f:
        if stream.nonspace(f) != b'{':
            raise ValueError('object expected')
        found = False
        while True:
            token = stream.nonspace(f)
            if token == b'}':
                break
            if token != b'"':
                raise ValueError('object field expected')
            name = stream.string(f, True)
            if stream.nonspace(f) != b':':
                raise ValueError('colon expected')
            token = stream.nonspace(f)
            if name != array_name:
                pb.skip(f, token)
            else:
                if found or token != b'[':
                    raise ValueError('invalid records array')
                found = True
                while True:
                    token = stream.nonspace(f)
                    if token == b']':
                        break
                    result.append(pb.select(f, token, fields))
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
    if not found:
        raise ValueError('records missing')
    return result


def top_array(path, identity_fields, id_field, wanted):
    """Decode only selected records from a top-level array, preserving order."""
    wanted = set(wanted)
    spans = {}
    with open(path, 'rb') as f:
        if stream.nonspace(f) != b'[':
            raise ValueError('array expected')
        while True:
            token = stream.nonspace(f)
            if token == b']':
                break
            start = f.tell() - 1
            identity = pb.select(f, token, identity_fields)
            end = f.tell()
            key = id_field(identity)
            if key in wanted:
                if key in spans or end - start > 1_048_576:
                    raise ValueError('duplicate or oversized selected record')
                spans[key] = (start, end)
            token = stream.nonspace(f)
            if token == b']':
                break
            if token != b',':
                raise ValueError('array delimiter expected')
        if not wanted or set(spans) != wanted:
            raise ValueError('selected array coverage differs')
        result = []
        for key, (start, end) in spans.items():
            f.seek(start)
            row = json.loads(f.read(end - start))
            if id_field(row) != key:
                raise ValueError('array identity changed')
            result.append(row)
    return result


def read_selected(path, array_name, identity_fields, id_field, wanted, content_fields=None):
    wanted = set(wanted)
    if not wanted:
        raise ValueError('empty identity selection')
    spans = {}
    with open(path, 'rb') as f:
        if stream.nonspace(f) != b'{':
            raise ValueError('object expected')
        seen_array = False
        while True:
            b = stream.nonspace(f)
            if b == b'}':
                break
            if b != b'"':
                raise ValueError('field expected')
            field = stream.string(f, True)
            if stream.nonspace(f) != b':':
                raise ValueError('colon expected')
            b = stream.nonspace(f)
            if field != array_name:
                pb.skip(f, b)
            else:
                if seen_array or b != b'[':
                    raise ValueError('invalid records array')
                seen_array = True
                while True:
                    b = stream.nonspace(f)
                    if b == b']':
                        break
                    if b != b'{':
                        raise ValueError('record object expected')
                    start = f.tell() - 1
                    identity = pb.select(f, b, identity_fields)
                    end = f.tell()
                    key = id_field(identity)
                    if key in wanted:
                        if key in spans or end - start > 1_048_576:
                            raise ValueError('duplicate or oversized selected record')
                        spans[key] = (start, end, identity)
                    b = stream.nonspace(f)
                    if b == b']':
                        break
                    if b != b',':
                        raise ValueError('record delimiter expected')
            b = stream.nonspace(f)
            if b == b'}':
                break
            if b != b',':
                raise ValueError('object delimiter expected')
        if not seen_array or set(spans) != wanted:
            raise ValueError('locked identities not fully found')
        rows = {}
        for key, (start, end, identity) in spans.items():
            f.seek(start)
            if content_fields is None:
                row = json.loads(f.read(end - start))
            else:
                row = pb.select(f, stream.nonspace(f), content_fields)
                if f.tell() != end:
                    raise ValueError('selected content span changed')
            if id_field(row) != key:
                raise ValueError('selected identity changed')
            rows[key] = row
    return rows
