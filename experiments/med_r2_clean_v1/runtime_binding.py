"""Extract only runtime and generation configuration from a frozen binding."""
import json
import stream_extract as s


def skip(f, first):
    if first == b'"': s.string(f)
    elif first in (b'{', b'['): s.skip_compound(f, first)
    else: s.scalar(f, first)


def extract(stream):
    if s.nonspace(stream) != b'{': raise ValueError('binding map expected')
    if s.nonspace(stream) != b'"': raise ValueError('nonempty binding map expected')
    s.string(stream)  # opaque key; never decode the full binding
    if s.nonspace(stream) != b':' or s.nonspace(stream) != b'{':
        raise ValueError('binding object expected')
    result = {}
    while True:
        if s.nonspace(stream) != b'"': raise ValueError('binding field expected')
        key = s.string(stream, True)
        if s.nonspace(stream) != b':': raise ValueError('colon expected')
        first = s.nonspace(stream)
        start = stream.tell() - 1
        skip(stream, first)
        end = stream.tell()
        if key in ('runtime', 'generation'):
            if key in result or first != b'{' or end - start > 65536:
                raise ValueError('invalid configuration field')
            stream.seek(start)
            result[key] = json.loads(stream.read(end - start))
            stream.seek(end)
        marker = s.nonspace(stream)
        if marker == b'}': break
        if marker != b',': raise ValueError('binding delimiter expected')
    if set(result) != {'runtime', 'generation'}:
        raise ValueError('missing configuration')
    return {'configuration_only': result}
