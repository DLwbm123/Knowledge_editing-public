"""Count fixed plan source identities; never decode question or reference."""
import collections
import re
import stream_extract as s


def skip(f, b):
    if b == b'"': s.string(f)
    elif b in (b'{', b'['): s.skip_compound(f, b)
    else: s.scalar(f, b)


def query(f):
    image = None
    while True:
        assert s.nonspace(f) == b'"'
        key = s.string(f, True); assert s.nonspace(f) == b':'
        b = s.nonspace(f)
        if key == 'image_path':
            assert b == b'"' and image is None
            image = s.string(f, True)
        else: skip(f, b)
        b = s.nonspace(f)
        if b == b'}': break
        assert b == b','
    assert image is not None
    return image


def row(f):
    out = {}
    while True:
        assert s.nonspace(f) == b'"'
        k = s.string(f, True); assert s.nonspace(f) == b':'
        b = s.nonspace(f)
        if k in ('mode', 'method'):
            assert b == b'"' and k not in out; out[k] = s.string(f, True)
        elif k == 'owner':
            assert b in b'0123456789' and k not in out; out[k] = s.scalar(f, b, True)
        elif k == 'query':
            assert b == b'{' and k not in out; out[k] = query(f)
        else: skip(f, b)
        b = s.nonspace(f)
        if b == b'}': break
        assert b == b','
    assert set(out) == {'mode', 'method', 'owner', 'query'}
    return out


def audit(stream, deny):
    reserved = set(deny['V95']) | set(deny['V111'])
    assert len(deny['V95']) == 24 and len(deny['V111']) == 15
    assert s.nonspace(stream) == b'{'
    counts = collections.Counter(); found = False
    while True:
        assert s.nonspace(stream) == b'"'
        k = s.string(stream, True); assert s.nonspace(stream) == b':'
        b = s.nonspace(stream)
        if k == 'plan':
            assert not found and b == b'['; found = True
            while True:
                b = s.nonspace(stream)
                if b == b']': break
                assert b == b'{'; v = row(stream)
                if v['mode'] == 'single' and v['owner'] in (47,117,93,87):
                    assert v['method'] == 'MED'; counts['total'] += 1
                    matches = set(re.findall(r'xmlab\d+', v['query'], re.I))
                    if len(matches) == 1:
                        group = next(iter(matches)).lower()
                        counts['slake'] += 1
                        counts['reserved'] += group in reserved
                    else:
                        counts['unresolved'] += 1
                b = s.nonspace(stream)
                if b == b']': break
                assert b == b','
        else: skip(stream, b)
        b = s.nonspace(stream)
        if b == b'}': break
        assert b == b','
    assert found and counts['total'] == 69
    return dict(total=69, slake=counts['slake'], reserved=counts['reserved'],
                unresolved=counts['unresolved'], QA_decoded=0)
