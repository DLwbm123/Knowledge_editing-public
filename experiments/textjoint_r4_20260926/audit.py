"""R4 upfront assets, connected historical exposure and CAL_PLUS qualification."""
import os,json,hashlib,collections,shutil
from pathlib import Path
import torch
ROOT=Path(os.environ['RUN_ROOT']);R3=ROOT.parents[1]/'textjoint-r3-20260926/run';R2=ROOT.parents[1]/'textjoint-r2-20260925/run';OUT=ROOT.parents[1]
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def norm(x):return ' '.join(str(x).casefold().split())
class Join:
 def __init__(self):self.parent={}
 def find(self,k):
  self.parent.setdefault(k,k)
  if self.parent[k]!=k:self.parent[k]=self.find(self.parent[k])
  return self.parent[k]
 def union(self,a,b):self.parent[self.find(a)]=self.find(b)
def main():
 ts=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];expected={(x['order'],x['writer']):x for x in read(R3/'public/RETAINED_BANKS.json')['experts']};assets=[]
 for t in ts:
  for arm in ['P','P+S']:
   rel=Path('runs/s20260925')/arm/f'e{t["order"]:03d}'/'private/edits'/f'e{t["order"]:03d}'/'C_NO_H/step-80.pt'
   p=R3/rel if (R3/rel).exists() else R2/rel;s=torch.load(p,map_location='cpu',weights_only=True)
   assert s['step']==80 and s['seed']==t['seed']+1 and s['canonical_edit_id']==t['canonical_edit_id']
   assert all(int(v['step'])==80 for v in s['optimizer']['state'].values())
   h=sha(p);assert h==expected[t['order'],arm]['sha256']
   initial=R2/'runs/s20260925/P'/f'e{t["order"]:03d}';w=initial/'private/edits'/f'e{t["order"]:03d}'/'initial/W0_COMPLETE.pt';p0=initial/'STEP0.pt'
   a=dict(order=t['order'],arm=arm,path=str(p),sha256=h,actual_optimizer_steps=80,seed=s['seed'],P_W0_sha256=sha(w),P_STEP0_sha256=sha(p0),resume_fields=[k for k in ['optimizer','torch_rng','cuda_rng','python_rng'] if k in s])
   assets.append(a)
 write(ROOT/'private/CHECKPOINT_INVENTORY.json',assets)
 write(ROOT/'public/CHECKPOINT_AUDIT.json',dict(status='PASS',experts=len(assets),actual_optimizer_steps=80,matched_R3_hashes=True,shared_P_W0_and_STEP0=True,seed=20260925))
 # Join every supplied native/variant identity to its source case/image/relation.
 base=OUT/'m3bench_task_specific_core9_data/20260903T013245Z'
 inventory=[json.loads(s) for s in (base/'base_inventory/BASE_QUERY_INVENTORY_CORE9.jsonl').read_text().splitlines()]
 data=Path('/remote-home/wangbomin/DataP/knowledge_editing/data/m3bench')
 vqa=read(data/'VQA-RAD/VQA_RAD Dataset Public.json');cases={s['image_name']:s['image_case_url'] for s in vqa}
 join=Join();byid={r['query_id']:r for r in inventory}
 for r in inventory:
  q='q:'+r['query_id'];join.union(q,'image:'+r['image_sha256'])
  source=(r['relative_image_path'].split('/')[0] if r['dataset']=='SLAKE' else cases.get(Path(r['image_path']).name,Path(r['image_path']).name))
  join.union(q,'case:'+r['dataset']+':'+source)
  for lin in r.get('lineage',[]):
   if lin.get('relation_id'):join.union(q,'relation:'+lin['relation_id'])
 events=read(OUT/'medtrace_stage17_20260912_r01/formal/private/CANDIDATE_POOL.json')['events'];method_ids=set()
 for e in events:
  method_ids.add(e['edit_query_id']);method_ids.update(e['all_probe_query_ids'])
  for q in e['all_probe_query_ids']:join.union('q:'+e['edit_query_id'],'q:'+q)
 for t in ts:
  for row in [t['native'],*t['official_evaluation_full']]:
   if row.get('query_id'):method_ids.add(row['query_id'])
 base_ids=set()
 files=[]
 for n in ['BASE_PREDICTIONS_CORE9.jsonl','BASE_NEW_PREDICTIONS.jsonl']:
  p=base/'base_predictions'/n;files.append(dict(path=str(p),bytes=p.stat().st_size))
  for line in p.read_text().splitlines():
   row=json.loads(line)
   if row.get('query_id'):base_ids.add(row['query_id'])
 exposed=method_ids|base_ids;components={join.find('q:'+q) for q in exposed};unexposed=[r for r in inventory if join.find('q:'+r['query_id']) not in components]
 fresh=[r for r in unexposed if r.get('source_task')=='T0']
 new_audit=dict(inventory_rows=len(inventory),source_task_counts=dict(collections.Counter(r.get('source_task','UNKNOWN') for r in inventory)),historical_method_probe_ids=len(method_ids),historical_Base_exposed_ids=len(base_ids),unexposed_connected_rows=len(unexposed),unexposed_T0_rows=len(fresh),historical_audit_complete=False,independent_CONFIRM=False,source='existing authorized core9 lineage, Stage17 candidate probes, R2/R3 role panels, historical Base outputs',decision='R2_VERIFY24_REGRESSION',reason='No certified new 24-edit queue with isolated T2G/T2L panels and full-history provenance; Base exposure and broader campaign history cannot be relabeled as untouched holdout')
 write(ROOT/'public/NEW_QUEUE_AUDIT.json',new_audit);write(ROOT/'private/NEW_QUEUE_AUDIT.json',dict(new_audit,evidence_files=files,unexposed_candidate_ids=[r['query_id'] for r in unexposed]))
 # CAL qualification uses source annotations, never generated image labels.
 used=[r for t in ts for k in ['official_evaluation_full','evaluation','U_fit','U_new','U_expanded'] for r in t[k]]+[t['native'] for t in ts]
 used+=read(ROOT/'private/AUXILIARY_POOL.json');groups={r['source_group'] for r in used};hashes={r['image_sha256'] for r in used}
 catalog={split:read(data/'SLAKE'/f'{split}.json') for split in ['train','validation','test']}
 groups|={'SLAKE:'+r['img_name'].split('/')[0] for split in ['validation','test'] for r in catalog[split]}
 def image_row(s):
  path=data/'SLAKE/imgs'/s['img_name'];return dict(dataset='SLAKE',question=s['question'],reference=str(s['answer']),image_path=str(path),image_sha256=sha(path),source_group='SLAKE:'+s['img_name'].split('/')[0],source_qid=s['qid'],content_type=s['content_type'],source_split='train')
 pool=[r for r in catalog['train'] if 'SLAKE:'+r['img_name'].split('/')[0] not in groups and r['q_lang']=='en']
 positives=[];bcounts=[]
 for t in ts[:24]:
  # Whole-image disease-set questions establish both relation and annotated entity
  # set. Relative-location/function references lack cross-image scope proof here.
  if t['native']['dataset']!='SLAKE' or norm(t['native']['question'])!='what diseases are included in the picture?':
   bcounts.append(dict(order=t['order'],qualified=0,reason='cross-image entity/relation scope not independently established'));continue
  rows=[];sources=set()
  for s in pool:
   if s['content_type']!='Abnormality' or norm(s['question'])!=norm(t['native']['question']) or norm(s['answer'])!=norm(t['native']['reference']):continue
   row=image_row(s)
   if row['image_sha256'] in hashes or row['source_group'] in sources:continue
   sources.add(row['source_group']);rows.append(dict(row,role='B_cross_image',order=t['order'],associated_edit=t['canonical_edit_id'],proof='same whole-image disease-set relation and full entity set from independent source QA annotations; distinct source image'))
   if len(rows)==2:break
  positives+=rows;bcounts.append(dict(order=t['order'],qualified=len(rows),reason='source-annotated whole-image diagnosis relation'))
 target_answers={norm(t['native']['reference']) for t in ts};b_sources={r['source_group'] for r in positives};negatives=[];n_sources=collections.Counter()
 for kind in ['C_hard','D_diverse']:
  for s in pool:
   if s['content_type'] in ['Modality','Plane'] or norm(s['answer']) in target_answers:continue
   hard=s['content_type']=='Abnormality' and norm(s['question'])=='what diseases are included in the picture?'
   if (kind=='C_hard')!=hard:continue
   row=image_row(s)
   if row['image_sha256'] in hashes or row['source_group'] in b_sources or n_sources[row['source_group']]>=2:continue
   negatives.append(dict(row,role=kind,proof='isolated source image; different annotated diagnosis/entity/relation; no formal input or training support'))
   n_sources[row['source_group']]+=1
   if sum(r['role']==kind for r in negatives)>=60:break
 a=read(R3/'private/CAL_ROUTE.json')['positives']
 for row in a:row['role']='A_'+row['role'];row['historical_role']='R3_CAL_ONLY'
 rows=a+positives+negatives;part=Join()
 for row in rows:
  group='source:'+row['source_group'];part.find(group)
  if row.get('associated_edit'):part.union(group,'edit:'+row['associated_edit'])
 for row in rows:
  group=part.find('source:'+row['source_group']);row['partition']='CAL_CHECK' if int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%3==0 else 'CAL_FIT'
 covered=sum(c['qualified']>=2 for c in bcounts);qualified=covered>=12 and sum(r['role']=='C_hard' for r in negatives)>=60 and len(n_sources)>=20
 # Reused R3 A is explicitly exposed; insufficient B alone prohibits threshold/gate training.
 summary=dict(status='SUPPORTED' if qualified else 'INCOMPLETE',A_reused_R3_CAL=len(a),B_cross_image_rows=len(positives),B_edits_with_two=covered,B_target_edits=12,B_sources=len(b_sources),C_hard_rows=sum(r['role']=='C_hard' for r in negatives),D_diverse_rows=sum(r['role']=='D_diverse' for r in negatives),negative_sources=len(n_sources),partition_counts=dict(collections.Counter(r['partition'] for r in rows)),blind_spots='relative anatomy/functions and rare VQA-RAD relations lack sufficient audited cross-image supports',Rcal='PENDING' if qualified else 'UNSUPPORTED_CALIBRATION',Rscope='CONDITIONAL_ONLY_IF_Rcal_INFEASIBLE' if qualified else 'NOT_AUTHORIZED_BY_QUALIFICATION',no_threshold_infeasibility_claim=not qualified)
 write(ROOT/'private/CAL_PLUS.json',dict(rows=rows,coverage=bcounts,summary=summary,frozen_before_thresholds=True));summary['sha256']=sha(ROOT/'private/CAL_PLUS.json');write(ROOT/'public/CAL_PLUS_AUDIT.json',summary)
 if not qualified:
  write(ROOT/'ROUTER_LOCK.json',dict(status='R0_FALLBACK_INCOMPLETE_CAL_PLUS',kappa=1.,mu=0.,Rcal='UNSUPPORTED',Rscope='UNSUPPORTED',cal_sha256=summary['sha256'],reason=summary['blind_spots'],frozen_before_R4_outputs=True))
  write(ROOT/'public/ROUTER_LOCK.json',read(ROOT/'ROUTER_LOCK.json'))
 write(ROOT/'AUDIT.json',dict(status='PASS_WITH_DATA_LIMITATIONS',new_queue=new_audit['decision'],CAL_PLUS=summary['status'],checkpoint_experts=len(assets)))
 print(new_audit);print(summary)
if __name__=='__main__':main()
