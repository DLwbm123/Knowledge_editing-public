"""Compare frozen V95 candidate qid pools and V133 mapped source identities only."""
import re
import stream_extract as s


def skip(f, b):
    if b == b'"': s.string(f)
    elif b in (b'{', b'['): s.skip_compound(f, b)
    else: s.scalar(f, b)


def int_list(f):
    values = []
    while True:
        b = s.nonspace(f)
        if b == b']': return values
        assert b in b'0123456789'
        values.append(s.scalar(f, b, True))
        b = s.nonspace(f)
        if b == b']': return values
        assert b == b','


def fields(f, kind):
    result = {}
    while True:
        assert s.nonspace(f) == b'"'
        key = s.string(f, True)
        assert s.nonspace(f) == b':'
        b = s.nonspace(f)
        if ((key in ('source_group', 'role') and kind in ('candidate', 'row')) or
                (key == 'query_id' and kind in ('row', 'unmapped'))):
            assert b == b'"' and key not in result
            result[key] = s.string(f, True)
        elif key == 'source_qid' and kind == 'row':
            assert b in b'0123456789' and key not in result
            result[key] = s.scalar(f, b, True)
        elif key == 'qid_pool' and kind == 'candidate':
            assert b == b'[' and key not in result
            result[key] = int_list(f)
        elif key == 'row' and kind == 'mapping':
            assert b == b'{' and key not in result
            result[key] = fields(f, 'row')
        else:
            skip(f, b)
        b = s.nonspace(f)
        if b == b'}': return result
        assert b == b','


def array_from_root(f, name, kind):
    assert s.nonspace(f) == b'{'
    found = False
    rows = []
    while True:
        assert s.nonspace(f) == b'"'
        key = s.string(f, True)
        assert s.nonspace(f) == b':'
        b = s.nonspace(f)
        if key == name:
            assert b == b'[' and not found
            found = True
            while True:
                b = s.nonspace(f)
                if b == b']': break
                assert b == b'{' 
                rows.append(fields(f, kind))
                b = s.nonspace(f)
                if b == b']': break
                assert b == b','
        else: skip(f, b)
        b = s.nonspace(f)
        if b == b'}': break
        assert b == b','
    assert found
    return rows


def group(value):
    match = re.fullmatch(r'(?:SLAKE:)?(xmlab\d+)', value, re.I)
    assert match
    return match.group(1).lower()


def compare(candidates, mapping, reserved):
    assert len(mapping) == 31 and len(reserved) == 24
    bygroup = {}
    for value in candidates:
        g = group(value['source_group'])
        if g in reserved:
            assert g not in bygroup
            bygroup[g] = value
    overlapping = []
    for record in mapping:
        row = record['row']
        g = group(row['source_group'])
        if g in reserved:
            assert g in bygroup and isinstance(row['source_qid'], int)
            overlapping.append((row['source_qid'] in bygroup[g]['qid_pool'], bygroup[g]['role']))
    return dict(mapped_count=len(mapping), reserved_group_overlaps=len(overlapping),
                same_qid_pool=sum(v[0] for v in overlapping),
                roles=sorted(v[1] for v in overlapping),
                source_QA_or_labels_decoded=0, scope='identity comparison only')


def compare_qualification(candidates, records, reserved):
    assert len(records) == 192 and len(reserved) == 24
    bygroup = {group(v['source_group']): v for v in candidates if group(v['source_group']) in reserved}
    matches = []
    for record in records:
        row = record['row']
        g = group(row['source_group'])
        if g in reserved:
            assert g in bygroup and isinstance(row['source_qid'], int)
            matches.append((row['source_qid'] in bygroup[g]['qid_pool'], bygroup[g]['role']))
    return dict(qualification_records=len(records), reserved_group_overlaps=len(matches),
                same_qid_pool=sum(v[0] for v in matches),
                roles=sorted(v[1] for v in matches), source_QA_or_labels_decoded=0,
                scope='identity comparison only; all qualification rows are still protected from bulk decoding')


def compare_fallback(candidates, records, fallback, deny):
    assert len(records) == 192 and len(fallback) == 30
    rows = {v['row']['query_id']: v['row'] for v in records}
    assert len(rows) == len(records)
    chosen = [rows[v['query_id']] for v in fallback]
    bygroup = {group(v['source_group']): v for v in candidates}
    counts = {'V95': 0, 'V111': 0}
    exact = 0
    for row in chosen:
        g = group(row['source_group'])
        for role in counts:
            counts[role] += g in set(deny[role])
        if g in set(deny['V95']):
            assert g in bygroup
            exact += row['source_qid'] in bygroup[g]['qid_pool']
    return dict(fallback_count=len(chosen), overlap_by_role=counts, V95_same_qid_pool=exact,
                source_QA_or_labels_decoded=0, scope='identity comparison only')
