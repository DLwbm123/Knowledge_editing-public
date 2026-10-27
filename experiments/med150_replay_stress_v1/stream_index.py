"""Index only group/expert/qid; decode selected old CHECK records in a second pass."""
import collections
import io
import json
import re
import stream_extract as s


def record_meta(f):
    meta={};b=s.nonspace(f)
    if b==b'}':raise ValueError('Empty record')
    f.seek(-1,1)
    while True:
        if s.nonspace(f)!=b'"':raise ValueError('Expected key')
        key=s.string(f,True)
        if s.nonspace(f)!=b':':raise ValueError('Expected colon')
        b=s.nonspace(f)
        if key in ('source_group','query_id'):
            if b!=b'"' or key in meta:raise ValueError('Invalid string identity')
            meta[key]=s.string(f,True)
        elif key in ('forced_expert_index','source_qid'):
            if b not in b'-0123456789' or key in meta:raise ValueError('Invalid numeric identity')
            meta[key]=s.scalar(f,b,True)
        elif b==b'"':s.string(f)
        elif b in (b'{',b'['):s.skip_compound(f,b)
        else:s.scalar(f,b)
        b=s.nonspace(f)
        if b==b'}':break
        if b!=b',':raise ValueError('Invalid object separator')
    assert set(meta)=={'source_group','query_id','forced_expert_index','source_qid'}
    m=re.fullmatch(r'SLAKE:(xmlab\d+)',meta['source_group'],re.I);assert m
    meta['group_key']=m.group(1).lower();return meta


def index(f):
    if s.nonspace(f)!=b'[':raise ValueError('Expected array')
    rows=[]
    while True:
        b=s.nonspace(f)
        if b==b']':break
        if b!=b'{':raise ValueError('Expected object')
        start=f.tell()-1;meta=record_meta(f);end=f.tell();rows.append(dict(**meta,start=start,end=end))
        b=s.nonspace(f)
        if b==b']':break
        if b!=b',':raise ValueError('Invalid array separator')
    return rows


def select(rows,denied):
    assert len(rows)==96 and len({r['query_id'] for r in rows})==96
    counts=collections.Counter(r['group_key'] for r in rows);assert len(counts)==32 and set(counts.values())=={3}
    for group in counts:
        assert len({r['forced_expert_index'] for r in rows if r['group_key']==group})==1
    keep=[r for r in rows if r['group_key'] not in denied]
    assert len(keep)==93 and len({r['group_key'] for r in keep})==31
    return keep


def decode_selected(f,rows):
    out=[]
    for row in rows:
        assert 0<row['end']-row['start']<=1048576
        f.seek(row['start']);v=json.loads(f.read(row['end']-row['start']))
        for key in ('source_group','query_id','forced_expert_index','source_qid'):assert v[key]==row[key]
        out.append(v)
    return out


def selfcheck():
    rows=[dict(source_group='SLAKE:xmlab1',query_id='a',forced_expert_index=0,source_qid=1,question='SECRET'),dict(source_group='SLAKE:xmlab2',query_id='b',forced_expert_index=1,source_qid=2,question='selected')]
    f=io.BytesIO(json.dumps(rows).encode());meta=index(f);assert len(meta)==2 and all('question' not in m for m in meta)
    values=decode_selected(f,[meta[1]]);assert len(values)==1 and values[0]['question']=='selected'
    assert all(x['source_group']!='SLAKE:xmlab1' for x in values)
    return dict(status='PASS',unselected_QA_decoded=0)

if __name__=='__main__':print(selfcheck())
