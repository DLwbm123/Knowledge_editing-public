"""Two-pass array scan: decode only explicitly selected qids."""
import io
import json
import re


def one(f):
    b=f.read(1)
    if not b:raise ValueError('Unexpected EOF')
    return b

def nonspace(f):
    b=one(f)
    while b in b' \t\r\n':b=one(f)
    return b

def string(f,keep=False):
    data=bytearray(b'"') if keep else None
    escaped=False
    while True:
        b=one(f)
        if keep:
            data.extend(b)
            if len(data)>256:raise ValueError('Field name too long')
        if escaped:escaped=False;continue
        if b==b'\\':escaped=True;continue
        if b==b'"':return json.loads(data) if keep else None

def skip_compound(f,first):
    stack=[b'}' if first==b'{' else b']']
    while stack:
        b=one(f)
        if b==b'"':string(f)
        elif b==b'{':stack.append(b'}')
        elif b==b'[':stack.append(b']')
        elif b in (b'}',b']'):
            if b!=stack.pop():raise ValueError('Mismatched nesting')

def scalar(f,first,keep=False):
    data=bytearray(first) if keep else None
    while True:
        b=f.read(1)
        if not b:break
        if b in b',}] \t\r\n':f.seek(-1,1);break
        if keep:
            data.extend(b)
            if len(data)>32:raise ValueError('qid scalar too long')
    return int(data) if keep else None

def object_qid(f):
    qid=None;b=nonspace(f)
    if b==b'}':raise ValueError('Empty record')
    f.seek(-1,1)
    while True:
        if nonspace(f)!=b'"':raise ValueError('Expected field name')
        key=string(f,True)
        if nonspace(f)!=b':':raise ValueError('Expected colon')
        b=nonspace(f)
        if key=='qid':
            if qid is not None or b not in b'-0123456789':raise ValueError('Invalid qid')
            qid=scalar(f,b,True)
        elif b==b'"':string(f)
        elif b in (b'{',b'['):skip_compound(f,b)
        else:scalar(f,b)
        b=nonspace(f)
        if b==b'}':break
        if b!=b',':raise ValueError('Expected record comma')
    if qid is None:raise ValueError('Missing qid')
    return qid

def extract(f,selected):
    wanted=set(selected);assert len(wanted)==len(selected)
    if nonspace(f)!=b'[':raise ValueError('Expected top-level array')
    matches={};records=0
    while True:
        b=nonspace(f)
        if b==b']':break
        if b!=b'{':raise ValueError('Expected top-level record')
        start=f.tell()-1;q=object_qid(f);end=f.tell();records+=1
        if q in wanted:
            if q in matches:raise ValueError('Duplicate selected qid')
            if end-start>1048576:raise ValueError('Selected record too large')
            f.seek(start);payload=f.read(end-start);value=json.loads(payload);assert value['qid']==q
            matches[q]=value;f.seek(end)
        b=nonspace(f)
        if b==b']':break
        if b!=b',':raise ValueError('Expected array comma')
    if set(matches)!=wanted:raise ValueError('Selected qid missing')
    return matches,dict(scanned_records=records,selected_records=len(matches),unselected_records_decoded=0)


def selfcheck():
    sample=[{'qid':1,'question':'UNSELECTED_SECRET_1','nested':{'a':['x\\"y',4]}},{'qid':2,'question':'selected','answer':'x'},{'qid':3,'answer':'UNSELECTED_SECRET_2'}]
    matches,receipt=extract(io.BytesIO(json.dumps(sample).encode()),[2])
    assert list(matches)==[2] and matches[2]['question']=='selected' and receipt['unselected_records_decoded']==0
    try:extract(io.BytesIO(json.dumps(sample).encode()),[2,4])
    except ValueError:pass
    else:raise AssertionError('Missing selection must fail')
    try:extract(io.BytesIO(json.dumps([sample[1],sample[1]]).encode()),[2])
    except ValueError:pass
    else:raise AssertionError('Duplicate selection must fail')
    return {'status':'PASS','unselected_records_decoded':0,'missing_and_duplicate_rejected':True}

if __name__=='__main__':print(selfcheck())
