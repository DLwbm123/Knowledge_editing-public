"""Count fixed single-owner plan records without decoding question or answer fields."""
import collections
import json
import stream_extract as s


def skip_value(f, first):
    if first == b'"':
        s.string(f)
    elif first in (b'{', b'['):
        s.skip_compound(f, first)
    else:
        s.scalar(f, first)


def row(f):
    data = {}
    while True:
        assert s.nonspace(f) == b'"'
        key = s.string(f, True)
        assert s.nonspace(f) == b':'
        first = s.nonspace(f)
        if key in ('mode', 'method'):
            assert first == b'"' and key not in data
            data[key] = s.string(f, True)
        elif key == 'owner':
            assert first in b'0123456789' and key not in data
            data[key] = s.scalar(f, first, True)
        else:
            skip_value(f, first)
        delimiter = s.nonspace(f)
        if delimiter == b'}':
            break
        assert delimiter == b','
    assert set(data) == {'owner', 'mode', 'method'}
    return data


def count(f):
    assert s.nonspace(f) == b'{' 
    result = collections.Counter()
    found = False
    while True:
        assert s.nonspace(f) == b'"'
        key = s.string(f, True)
        assert s.nonspace(f) == b':'
        first = s.nonspace(f)
        if key == 'plan':
            assert not found and first == b'['
            found = True
            while True:
                first = s.nonspace(f)
                if first == b']':
                    break
                assert first == b'{'
                item = row(f)
                if item['mode'] == 'single' and item['owner'] in (47, 117, 93, 87):
                    assert item['method'] == 'MED'
                    result[item['owner']] += 1
                delimiter = s.nonspace(f)
                if delimiter == b']':
                    break
                assert delimiter == b','
        else:
            skip_value(f, first)
        delimiter = s.nonspace(f)
        if delimiter == b'}':
            break
        assert delimiter == b','
    assert found and set(result) == {47, 117, 93, 87}
    return dict(total=sum(result.values()), per_owner=sorted(result.values()), unselected_QA_decoded=0)


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'rb') as stream:
        print(json.dumps(count(stream)))
