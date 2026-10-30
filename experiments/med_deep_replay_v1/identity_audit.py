"""Bounded metadata-only overlap check for the pre-existing replay FIT role."""
import json, pathlib, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[2]
snap=json.loads((root/'.private/prospective_confirmation_v111_20261025/EXPOSURE_SNAPSHOT.json').read_text())
v106=snap['V106_audited']
history=sorted(set(snap['history_groups'])-set(snap['replay64']))
source=(root/'experiments/med150_source_review_v1/stream_extract.py').read_text().split("if __name__==",1)[0]
script='V106='+repr(v106)+'\nHISTORY='+repr(history)+'\n'+source+'''\nimport json, pathlib, re, collections
cfg=json.loads(pathlib.Path('/data/bmw/.q.20261027.150/ENV.json').read_text()); replay=pathlib.Path(cfg['REPLAY_PARENT'])
p=pathlib.Path('/data/bmw/Knowledge_editing/outputs/med-fixed-depth-v1-20261030/run')
deny=json.loads(pathlib.Path('/data/bmw/Knowledge_editing/outputs/med150-replay-stress-v1-20261028/run/DENY_GROUPS.private.json').read_text())
locked=json.loads((p/'PANEL_IDENTITY_LOCK.json').read_text())['index']
def norm(x):
 m=re.search(r'(?i)(xmlab[0-9]+)',str(x)); return m.group(1).lower() if m else None
reserve={norm(v) for k in ('V95','V111') for v in deny[k]} | {norm(v) for v in V106}
assert None not in reserve
historic={norm(v) for v in HISTORY};assert None not in historic
panel=set()
for row in locked['tasks']:
 n=norm(row['native']['source_group'] or row['native']['image_path'])
 if n: panel.add(n)
for row in locked['plan']:
 n=norm(row['query']['image_path'])
 if n: panel.add(n)
def scan(name,expected):
 rows=[]
 with (replay/'private'/name).open('rb') as f:
  assert nonspace(f)==b'['
  while True:
   b=nonspace(f)
   if b==b']':break
   assert b==b'{'; got={}
   while True:
    b=nonspace(f)
    if b==b'}':break
    assert b==b'"';key=string(f,True);assert nonspace(f)==b':'
    b=nonspace(f)
    if key in ('source_group','role'):
     assert b==b'"' and key not in got;got[key]=string(f,True)
    elif b==b'"':string(f)
    elif b in (b'{',b'['):skip_compound(f,b)
    else:scalar(f,b)
    b=nonspace(f)
    if b==b'}':break
    assert b==b','
   assert set(got)=={'source_group','role'}
   assert got['role']==expected;g=norm(got['source_group']);assert g
   rows.append(g)
   b=nonspace(f)
   if b==b']':break
   assert b==b','
 return rows
rows=scan('REPLAY_FIT.json','REPLAY_FIT')
checks=set(scan('REPLAY_CHECK.json','REPLAY_CHECK'));assert len(checks)==32
counts=collections.Counter(rows);assert len(rows)==192 and len(counts)==64 and set(counts.values())=={3}
print(json.dumps({'records':len(rows),'groups':len(counts),'protected_overlap_groups':len(set(counts)&reserve),'panel_overlap_groups':len(set(counts)&panel),'check_overlap_groups':len(set(counts)&checks),'other_historical_overlap_groups':len(set(counts)&historic),'surviving_groups':len(set(counts)-reserve-panel-checks-historic),'qa_decoded':0,'scope':'identity_only','content_admitted':False}))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],input=script,text=True,capture_output=True,timeout=60)
if r.returncode: raise RuntimeError('identity audit failed: '+r.stderr[-1000:])
x=json.loads(r.stdout);print(json.dumps(x))
