import json,os,sys
from pathlib import Path
w=Path(os.environ['FIXED_WORKTREE'])
sys.path[:0]=[str(w/'experiments/med150_source_review_v1')]
import stream_extract as s

def skip(f,b):
 if b==b'"': s.string(f)
 elif b in (b'{',b'['): s.skip_compound(f,b)
 else: s.scalar(f,b)
def string(f,b):
 if b!=b'"': raise ValueError('string required')
 return s.string(f,True)
def obj(f, mode):
 out={}
 while True:
  b=s.nonspace(f)
  if b==b'}': return out
  if b!=b'"': raise ValueError('key required')
  k=s.string(f,True)
  if s.nonspace(f)!=b':': raise ValueError('colon')
  b=s.nonspace(f)
  if mode=='row' and k in ('BASE','MED'):
   if b!=b'{': raise ValueError('nested score')
   out[k]=obj(f,'arm')
  elif (mode=='arm' and k=='payload_key') or (mode=='row' and k=='query_id'):
   out[k]=string(f,b)
  elif mode=='row' and k=='owner': out[k]=s.scalar(f,b,True)
  else: skip(f,b)
  b=s.nonspace(f)
  if b==b'}': return out
  if b!=b',': raise ValueError('comma')
source=w/'.private/med_fixed_depth_v1_20261030/BASE_M0_SCORE_BINDINGS.json'
dest=w/'.private/med_fixed_depth_v1_20261030/AGREEMENT_DECISIONS_FROZEN.json'
if dest.exists(): raise FileExistsError('Frozen decisions exist')
rows=[]
with open(source,'rb') as f:
 if s.nonspace(f)!=b'{': raise ValueError('root')
 while True:
  b=s.nonspace(f)
  if b==b'}': break
  if b!=b'"': raise ValueError('root key')
  k=s.string(f,True)
  if s.nonspace(f)!=b':': raise ValueError('root colon')
  b=s.nonspace(f)
  if k=='rows':
   if b!=b'[': raise ValueError('array')
   while True:
    b=s.nonspace(f)
    if b==b']': break
    if b!=b'{': raise ValueError('row')
    r=obj(f,'row')
    if set(r)!={'owner','query_id','BASE','MED'} or any(set(r[a])!={'payload_key'} for a in ('BASE','MED')): raise ValueError('fields')
    rows.append({'owner':r['owner'],'query_id':r['query_id'],'selected':'MED' if r['BASE']['payload_key']==r['MED']['payload_key'] else 'CDEEP'})
    b=s.nonspace(f)
    if b==b']': break
    if b!=b',': raise ValueError('row comma')
  else: skip(f,b)
  b=s.nonspace(f)
  if b==b'}': break
  if b!=b',': raise ValueError('root comma')
if len(rows)!=508 or len({(r['owner'],r['query_id']) for r in rows})!=508: raise ValueError('coverage')
with dest.open('x') as f: json.dump({'status':'FROZEN','rows':rows,'method':'exact_full_payload_key_match','scores_decoded':0},f)
print(json.dumps({'status':'FROZEN','events':len(rows),'scores_decoded':0}))
