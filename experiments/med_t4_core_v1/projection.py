"""Project explicit JSON fields without deserializing excluded answer values."""
import json
import stream_extract as s


def skip(f, b):
    if b == b'"':
        s.string(f)
    elif b in (b'{', b'['):
        s.skip_compound(f, b)
    else:
        s.scalar(f, b)


def project(f, b, rule):
    if rule is True:
        start = f.tell() - 1
        skip(f, b)
        end = f.tell()
        f.seek(start)
        return json.loads(f.read(end - start))
    if isinstance(rule, list):
        assert b == b'['
        rows = []
        while True:
            b = s.nonspace(f)
            if b == b']':
                return rows
            item = project(f, b, rule[0])
            if item is not None:
                rows.append(item)
            b = s.nonspace(f)
            if b == b']':
                return rows
            assert b == b','
    if '__select__' in rule:
        field, wanted = rule['__select__']
        start = f.tell() - 1
        identity = project(f, b, {field: True})
        end = f.tell()
        if identity.get(field) not in wanted:
            return None
        f.seek(start)
        row = project(f, s.nonspace(f), rule['fields'])
        assert f.tell() == end and row[field] == identity[field]
        return row
    assert b == b'{'
    result = {}
    while True:
        b = s.nonspace(f)
        if b == b'}':
            return result
        assert b == b'"'
        key = s.string(f, True)
        assert s.nonspace(f) == b':'
        b = s.nonspace(f)
        if key in rule or '*' in rule:
            assert key not in result
            result[key] = project(f, b, rule.get(key, rule.get('*')))
        else:
            skip(f, b)
        b = s.nonspace(f)
        if b == b'}':
            return result
        assert b == b','


def read(path, rule):
    with open(path, 'rb') as f:
        return project(f, s.nonspace(f), rule)
