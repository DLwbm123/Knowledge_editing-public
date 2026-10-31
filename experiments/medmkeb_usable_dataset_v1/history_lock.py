"""Read selected V18 identity metadata only; preserve unknown status."""
import collections,json,os,sys
from pathlib import Path
root=Path(os.environ['TASK_REPO'])
sys.path.insert(0,str(root/'experiments/medmkeb_usable_dataset_v1'))
from metadata_inventory import records,Groups
import stream_extract as s
import score_index as si

v15=root/'.private/medmkeb_usable_dataset_v1_20261031/IDENTITY_QUARANTINE_V15.json'
v18=root/'.private/medmkeb_expanded_v18_20261019/CANDIDATES_FROZEN.json'
state=json.loads((root/'.private/medmkeb_independent_admission_v15_20261019/STATE.json').read_text())
data=Path(state['local_data_root']); paths=[data/'raw/train_data.json',data/'raw/eval_data_threehop_final.json']
if any(not p.is_file() or p.stat().st_size>64*1024**2 for p in paths): raise ValueError('source missing or oversized')
rows=dict(zip(('train','eval'),map(records,paths)))
if [len(rows[x]) for x in ('train','eval')] != [4490,2497]: raise ValueError('source count mismatch')
old=json.loads(v15.read_text()); oldrows=old['records']
if old['status']!='IDENTITY_ONLY_NOT_ADMITTED' or len(oldrows)!=6987: raise ValueError('prior lock mismatch')
wanted={'native_component_roles','PMC_roles','actual_locality_image_roles','historical_image_keys','historical_PMC','excluded_native_components'}; meta={}
with v18.open('rb') as f:
 if s.nonspace(f)!=b'{': raise ValueError('root')
 while True:
  t=s.nonspace(f)
  if t==b'}': break
  if t!=b'"': raise ValueError('key')
  key=s.string(f,True)
  if s.nonspace(f)!=b':': raise ValueError('separator')
  t=s.nonspace(f); start=f.tell()-1; si.skip(f,t); end=f.tell()
  if key in wanted:
   if key in meta or end-start>10*1024*1024: raise ValueError('metadata duplicate or oversized')
   f.seek(start); meta[key]=json.loads(f.read(end-start)); f.seek(end)
  t=s.nonspace(f)
  if t==b'}': break
  if t!=b',': raise ValueError('delimiter')
 if f.read().strip(): raise ValueError('trailing')
if set(meta)!=wanted: raise ValueError('missing metadata')
for k in ('native_component_roles','PMC_roles','actual_locality_image_roles'):
 if not isinstance(meta[k],dict) or not all(isinstance(a,str) and isinstance(b,str) for a,b in meta[k].items()): raise ValueError('role type')
for k in ('historical_image_keys','historical_PMC','excluded_native_components'):
 if not isinstance(meta[k],list) or not all(isinstance(x,str) for x in meta[k]): raise ValueError('identity type')
g=Groups()
for rs in rows.values():
 for a,b,c in rs:g.union(a,b)
hist=set(meta['historical_image_keys']); hist_groups={g.root(x) for x in hist if x in g.parent}; hist_groups.update(meta['excluded_native_components'])
locked=[]; counts=collections.Counter(); n=0
for split, rs in rows.items():
 for i,(a,b,c) in enumerate(rs):
  oldrow=oldrows[n];n+=1; group=g.root(a)
  if (oldrow['official_split'],oldrow['index'],oldrow['edit_image_group'],oldrow['locality_article']) != (split,i,group,c): raise ValueError('previous identity lock mismatch')
  flags=list(oldrow['reasons'])
  if group in meta['native_component_roles']:flags.append('V18_FROZEN_EDIT_GROUP')
  if c in meta['PMC_roles']:flags.append('V18_FROZEN_LOCALITY_ARTICLE')
  if group in hist_groups:flags.append('V18_HISTORY_EDIT_GROUP')
  if c in meta['historical_PMC']:flags.append('V18_HISTORY_LOCALITY_ARTICLE')
  status='QUARANTINE_KNOWN' if flags else 'UNKNOWN';counts[status]+=1
  for flag in flags:counts[flag]+=1
  locked.append({'official_split':split,'index':i,'edit_image_group':group,'locality_article':c,'status':status,'reasons':flags})
out=Path(os.environ['TASK_PRIVATE_OUT']);fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:json.dump({'status':'IDENTITY_ONLY_NOT_ADMITTED','records':locked},f,separators=(',',':'))
print(json.dumps({'records':n,'counts':dict(counts),'v18_locality_image_role_count':len(meta['actual_locality_image_roles']),'qa_decoded':0,'pixels':0,'models':0}))
