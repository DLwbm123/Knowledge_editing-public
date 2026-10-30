"""Identity-first replay FIT slicing; caller must supply independently verified excluded groups."""
import collections
import json
import re
import stream_extract as s


def metadata(f):
    result={}
    while True:
        b=s.nonspace(f)
        if b==b'}': break
        if b!=b'"': raise ValueError('record key expected')
        key=s.string(f,True)
        if s.nonspace(f)!=b':': raise ValueError('colon expected')
        b=s.nonspace(f)
        if key in ('source_group','query_id','role'):
            if key in result or b!=b'"': raise ValueError('invalid identity')
            result[key]=s.string(f,True)
        elif key=='source_qid':
            if key in result or b not in b'-0123456789': raise ValueError('invalid qid')
            result[key]=s.scalar(f,b,True)
        elif b==b'"': s.string(f)
        elif b in (b'{',b'['): s.skip_compound(f,b)
        else: s.scalar(f,b)
        b=s.nonspace(f)
        if b==b'}': break
        if b!=b',': raise ValueError('record delimiter expected')
    if set(result)!={'source_group','query_id','role','source_qid'}: raise ValueError('missing identity')
    m=re.fullmatch(r'SLAKE:(xmlab\d+)',result['source_group'],re.I)
    if not m: raise ValueError('unidentified group')
    result['group_key']=m.group(1).lower()
    return result


def index(f):
    if s.nonspace(f)!=b'[': raise ValueError('array expected')
    records=[]
    while True:
        b=s.nonspace(f)
        if b==b']': break
        if b!=b'{': raise ValueError('object expected')
        start=f.tell()-1; r=metadata(f); end=f.tell()
        if r['role']!='REPLAY_FIT': raise ValueError('non-FIT role')
        records.append(dict(r,start=start,end=end))
        b=s.nonspace(f)
        if b==b']': break
        if b!=b',': raise ValueError('array delimiter expected')
    counts=collections.Counter(r['group_key'] for r in records)
    if len(records)!=192 or len(counts)!=64 or set(counts.values())!={3} or len({r['query_id'] for r in records})!=192:
        raise ValueError('historic FIT identity coverage mismatch')
    return records


def lock(records, exclusion_sets):
    required={'V95','V106','V111','CHECK32','PANEL','OTHER_RESERVED'}
    if set(exclusion_sets)!=required or any(x is None for x in exclusion_sets.values()):
        raise ValueError('complete independent role exclusions required')
    denied=set().union(*(set(v) for v in exclusion_sets.values()))
    selected=[r for r in records if r['group_key'] not in denied]
    if not selected or len(selected)%3: raise ValueError('invalid remaining groups')
    if any(r['group_key'] in denied for r in selected): raise ValueError('excluded group survived')
    return {'status':'IDENTITY_FROZEN_ONLY','records':selected,'groups':len({r['group_key'] for r in selected}),
            'excluded_groups':len({r['group_key'] for r in records}&denied),'qa_decoded':0}


def decode(f, frozen, content_admission):
    if frozen.get('status')!='IDENTITY_FROZEN_ONLY' or content_admission!='FIT_CONTENT_AUTHORIZED':
        raise ValueError('independent content authorization required')
    rows=[]
    for r in frozen['records']:
        if not 0<r['end']-r['start']<=1048576: raise ValueError('invalid record span')
        f.seek(r['start']); v=json.loads(f.read(r['end']-r['start']))
        for k in ('source_group','query_id','role','source_qid'):
            if v.get(k)!=r[k]: raise ValueError('identity changed after freeze')
        rows.append(v)
    if len(rows)!=len(frozen['records']): raise ValueError('incomplete content')
    return rows
