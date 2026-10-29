"""Inspect source group and role fields only, never deserialize QA or labels."""
import collections
import json
import re
import stream_extract as s


def skip(f, first):
    if first == b'"':
        s.string(f)
    elif first in (b'{', b'['):
        s.skip_compound(f, first)
    else:
        s.scalar(f, first)


def fields(f, select):
    result = {}
    while True:
        assert s.nonspace(f) == b'"'
        key = s.string(f, True)
        assert s.nonspace(f) == b':'
        first = s.nonspace(f)
        if key in select:
            assert first == b'"' and key not in result
            result[key] = s.string(f, True)
        elif key == 'row':
            assert first == b'{' and 'row' not in result
            result['row'] = fields(f, {'source_group', 'audit_role'})
        else:
            skip(f, first)
        delimiter = s.nonspace(f)
        if delimiter == b'}':
            break
        assert delimiter == b','
    return result


def audit(stream, deny):
    assert set(deny) == {'V95', 'V111'} and len(deny['V95']) == 24 and len(deny['V111']) == 15
    reserved = set(deny['V95']) | set(deny['V111'])
    assert s.nonspace(stream) == b'{'
    counts, overlap = collections.Counter(), collections.Counter()
    found = False
    while True:
        assert s.nonspace(stream) == b'"'
        key = s.string(stream, True)
        assert s.nonspace(stream) == b':'
        first = s.nonspace(stream)
        if key == 'rows':
            assert not found and first == b'['
            found = True
            while True:
                first = s.nonspace(stream)
                if first == b']':
                    break
                assert first == b'{'
                meta = fields(stream, set())['row']
                assert set(meta) == {'source_group', 'audit_role'}
                match = re.fullmatch(r'SLAKE:(xmlab\d+)', meta['source_group'], re.I)
                assert match, 'unrecognized source group'
                role = meta['audit_role']
                assert role in ('BASIS', 'HELDOUT')
                counts[role] += 1
                overlap[role] += match.group(1).lower() in reserved
                delimiter = s.nonspace(stream)
                if delimiter == b']':
                    break
                assert delimiter == b','
        else:
            skip(stream, first)
        delimiter = s.nonspace(stream)
        if delimiter == b'}':
            break
        assert delimiter == b','
    assert found and sum(counts.values()) == 192
    return dict(rows=dict(counts), reserved_group_overlap=dict(overlap),
                metadata_fields=['source_group', 'audit_role'], QA_or_labels_decoded=0,
                scope='group identity only; does not establish full role admission')


def audit_mapping(stream, deny):
    reserved = set(deny['V95']) | set(deny['V111'])
    assert s.nonspace(stream) == b'{'
    found, total, overlap = False, 0, 0
    while True:
        assert s.nonspace(stream) == b'"'
        key = s.string(stream, True)
        assert s.nonspace(stream) == b':'
        first = s.nonspace(stream)
        if key == 'mapping':
            assert not found and first == b'['
            found = True
            while True:
                first = s.nonspace(stream)
                if first == b']':
                    break
                assert first == b'{'
                row = fields(stream, set())['row']
                match = re.fullmatch(r'SLAKE:(xmlab\d+)', row['source_group'], re.I)
                assert match, 'unrecognized mapping source group'
                total += 1
                overlap += match.group(1).lower() in reserved
                delimiter = s.nonspace(stream)
                if delimiter == b']':
                    break
                assert delimiter == b','
        else:
            skip(stream, first)
        delimiter = s.nonspace(stream)
        if delimiter == b'}':
            break
        assert delimiter == b','
    assert found and total == 31
    return dict(mapped_sources=total, reserved_group_overlap=overlap, QA_or_labels_decoded=0)
