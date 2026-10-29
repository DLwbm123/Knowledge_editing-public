"""Lock fixed single-consumer identities before decoding selected plan rows."""
import json
import re
import stream_extract as s

OWNERS = {47, 117, 93, 87}


def skip(f, b):
    if b == b'"': s.string(f)
    elif b in (b'{', b'['): s.skip_compound(f, b)
    else: s.scalar(f, b)


def fields(f, kind):
    out = {}
    while True:
        if s.nonspace(f) != b'"': raise ValueError('field expected')
        key = s.string(f, True)
        if s.nonspace(f) != b':': raise ValueError('colon expected')
        b = s.nonspace(f)
        if kind == 'row' and key in ('mode', 'method') or kind == 'query' and key in ('query_id', 'image_path'):
            if b != b'"' or key in out: raise ValueError('invalid identity field')
            out[key] = s.string(f, True)
        elif kind == 'row' and key == 'owner':
            if b not in b'0123456789' or key in out: raise ValueError('invalid owner')
            out[key] = s.scalar(f, b, True)
        elif kind == 'row' and key == 'query':
            if b != b'{' or key in out: raise ValueError('invalid query')
            out[key] = fields(f, 'query')
        else: skip(f, b)
        b = s.nonspace(f)
        if b == b'}': return out
        if b != b',': raise ValueError('delimiter expected')


def lock(stream, denied):
    if s.nonspace(stream) != b'{': raise ValueError('root expected')
    result = []; found = False
    while True:
        if s.nonspace(stream) != b'"': raise ValueError('root field expected')
        key = s.string(stream, True)
        if s.nonspace(stream) != b':': raise ValueError('colon expected')
        b = s.nonspace(stream)
        if key == 'plan':
            if found or b != b'[': raise ValueError('invalid plan')
            found = True
            while True:
                b = s.nonspace(stream)
                if b == b']': break
                if b != b'{': raise ValueError('row expected')
                start = stream.tell() - 1; row = fields(stream, 'row'); end = stream.tell()
                if row.get('mode') == 'single' and row.get('owner') in OWNERS:
                    q = row.get('query', {}); path = q.get('image_path', '')
                    if row.get('method') != 'MED' or not q.get('query_id'):
                        raise ValueError('missing plan identity')
                    groups = {v.lower() for v in re.findall(r'xmlab\d+', path, re.I)}
                    rad = any(v in ('vqa-rad', 'vqa_rad', 'vqarad') for v in path.lower().split('/'))
                    if len(groups) == 1 and not rad:
                        group = next(iter(groups)); dataset = 'SLAKE'
                    elif not groups and rad:
                        group = None; dataset = 'VQA-RAD'
                    else: raise ValueError('ambiguous dataset identity')
                    result.append(dict(owner=row['owner'], query_id=q['query_id'], dataset=dataset,
                                       group=group, image_path=path, excluded=group in denied, span=[start, end]))
                b = s.nonspace(stream)
                if b == b']': break
                if b != b',': raise ValueError('plan delimiter expected')
        else: skip(stream, b)
        b = s.nonspace(stream)
        if b == b'}': break
        if b != b',': raise ValueError('root delimiter expected')
    if not found or len(result) != 69 or {v['owner'] for v in result} != OWNERS:
        raise ValueError('frozen consumer identity count mismatch')
    if len({(v['owner'], v['query_id']) for v in result}) != len(result):
        raise ValueError('duplicate consumer identity')
    return dict(label='PLAN_IDENTITY_LOCK_ONLY', items=result, QA_decoded=0)


def decode(stream, locked, admission):
    if locked.get('label') != 'PLAN_IDENTITY_LOCK_ONLY': raise ValueError('plan not locked')
    if admission.get('status') != 'PLAN_CONTENT_AUTHORIZED': raise ValueError('content not admitted')
    if admission.get('consumers') != sum(not item['excluded'] for item in locked['items']):
        raise ValueError('admitted count mismatch')
    output = []
    for item in locked['items']:
        if item['excluded']: continue
        start, end = item['span']
        if start < 0 or not 0 < end-start <= 1_048_576: raise ValueError('invalid span')
        stream.seek(start); row = json.loads(stream.read(end-start))
        if (row.get('mode') != 'single' or row.get('method') != 'MED' or
                row.get('owner') != item['owner'] or row.get('query', {}).get('query_id') != item['query_id'] or
                row.get('query', {}).get('image_path') != item['image_path']):
            raise ValueError('locked plan identity changed')
        output.append(row)
    return output
