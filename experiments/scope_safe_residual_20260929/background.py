"""Freeze train-release acquisition facts by source, before Base scoring."""
import hashlib,collections
from pathlib import Path
from p0 import read,write,digest
from audit_data import ident

def main(old,out,data):
 tasks=read(old/'private/TASKS_R2_LOCKED.json')['tasks'];aux=read(old/'private/AUXILIARY_POOL.json')
 used=aux+[t['native'] for t in tasks]+[r for t in tasks for k in ['evaluation','official_evaluation_full','U_fit','U_new','U_expanded'] for r in t[k]]
 for n in ['PRESSURE_VALIDATION_FROZEN.json','PRESSURE_TEST_FROZEN.json']:used+=read(old/'private'/n)['rows']
 historical_cal=[]
 def collect(x):
  if isinstance(x,dict):
   if 'source_group' in x and 'question' in x:historical_cal.append(x)
   else:
    for v in x.values():collect(v)
  elif isinstance(x,list):
   for v in x:collect(v)
 for p in old.parents[1].glob('textjoint-r*/run/private/CAL*.json'):collect(read(p))
 used+=historical_cal
 groups={r['source_group'] for r in used};images={r.get('image_sha256') for r in used};forbidden={ident(r) for r in used}
 for split in ['validation','test']:groups|={'SLAKE:'+r['img_name'].split('/')[0] for r in read(data/(split+'.json'))}
 pools=collections.defaultdict(list);seen=set();candidates=0
 for x in sorted(read(data/'train.json'),key=lambda x:(x['img_id'],x['qid'])):
  if x['q_lang']!='en' or x['content_type'] not in ['Modality','Plane']:continue
  candidates+=1;group='SLAKE:'+x['img_name'].split('/')[0]
  if group in groups:continue
  image=data/'imgs'/x['img_name'];sha=hashlib.sha256(image.read_bytes()).hexdigest();row=dict(question=x['question'],reference=str(x['answer']),image_path=str(image),image_sha256=sha,source_group=group,dataset='SLAKE',source_qid=x['qid'],source_role='train',category=x['content_type'])
  if sha in images or ident(row) in forbidden or ident(row) in seen:continue
  seen.add(ident(row));split=int(digest([group,'scope-stage-20260929'])[:8],16)%10
  purpose='U_bg' if split<6 else 'CAL_NEG' if split==6 else 'CHECK_NEG' if split==7 else 'LOCALITY_STRESS_HOLDOUT'
  row.update(purpose=purpose,scope='negative',scope_basis='Acquisition modality/plane is source-image-specific; excludes every known edited/fit/formal/calibration source and image. No label from answer or embedding distance.')
  pools[purpose].append(row)
 for purpose in ['U_bg','CAL_NEG','CHECK_NEG','LOCALITY_STRESS_HOLDOUT']:
  write(out/'private'/f'{purpose}.json',dict(rows=pools[purpose],frozen_before_base_scores=True,selection='Source hash partition, train-release acquisition facts only'))
 sets=[{x['source_group'] for x in pools[k]} for k in pools]
 assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
 summary={k:dict(inputs=len(v),sources=len({x['source_group'] for x in v})) for k,v in pools.items()}
 write(out/'public/BACKGROUND_POOL_AUDIT.json',dict(candidates=candidates,historical_cal_rows_excluded=len(historical_cal),pools=summary,formal_and_edited_sources_excluded=True,cross_partition_source_overlap=0,Base_accuracy=None,Base_correct_holdout=None,holdout_status='CANDIDATES_FROZEN_AWAITING_LEASE_AND_NEW_SCORING_RESERVATION',limitation='Only acquisition facts certified, no claim of full difficult same-image factual coverage'))
 manifest=read(out/'private/SCOPE_ROLE_MANIFEST.json');manifest['new_source_pools']={k:[dict(input_hash=digest(ident(x)),source=x['source_group'],scope=x['scope'],purpose=k) for x in v] for k,v in pools.items()};write(out/'private/SCOPE_ROLE_MANIFEST.json',manifest)
 audit=read(out/'public/DATA_AUDIT.json');audit.update(new_background_pools=summary,U_bg_status='RESTRICTED_ACQUISITION_FACTS',old_aux_background_not_used=True,holdout_candidates_frozen=True);write(out/'public/DATA_AUDIT.json',audit)
if __name__=='__main__':
 import sys
 main(*(Path(x) for x in sys.argv[1:]))
