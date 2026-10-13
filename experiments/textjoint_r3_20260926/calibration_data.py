"""Freeze outcome-blind, image-preserving CAL_ROUTE from isolated SLAKE train."""
import os,json,hashlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT'])
def norm(x):return ' '.join(x.casefold().split()).rstrip('?.。？')
def main():
 read=lambda p:json.loads(p.read_text())
 tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];phrases=read(ROOT/'private/CAL_NATIVE_PARAPHRASES.json')
 used=[r for t in tasks for k in ['evaluation','official_evaluation_full','U_fit','U_new','U_expanded'] for r in t[k]]+[t['native'] for t in tasks]
 used+=read(ROOT/'private/AUXILIARY_POOL.json')
 groups={r['source_group'] for r in used};hashes={r['image_sha256'] for r in used}
 data=Path('/remote-home/wangbomin/DataP/knowledge_editing/data/m3bench/SLAKE')
 groups|={'SLAKE:'+r['img_name'].split('/')[0] for split in ['validation','test'] for r in read(data/(split+'.json'))}
 negatives=[];seen=set()
 for r in sorted(read(data/'train.json'),key=lambda x:(x['img_id'],x['qid'])):
  group='SLAKE:'+r['img_name'].split('/')[0]
  if group in groups or r['q_lang']!='en' or r['content_type'] not in ['Modality','Plane']:continue
  path=data/'imgs'/r['img_name'];h=hashlib.sha256(path.read_bytes()).hexdigest()
  if h in hashes or (h,norm(r['question'])) in seen:continue
  # Acquisition modality/plane is image-specific; disjoint images and source cases
  # make these non-target facts for every native edit in the whole 48-edit library.
  seen.add((h,norm(r['question'])))
  negatives.append(dict(question=r['question'],reference=str(r['answer']),image_path=str(path),image_sha256=h,source_group=group,dataset='SLAKE',source_qid=r['qid'],role='whole_library_non_target',reason='isolated acquisition modality/plane on a different source image',modality='image+text'))
  if len(negatives)>=60 and len({x['source_group'] for x in negatives})>=20:break
 positives=[]
 for t,qs in zip(tasks[:24],phrases,strict=True):
  forbidden={norm(q) for q in t['semantic_fit_questions']+t['fit_questions']+[r['question'] for r in t['official_evaluation_full']]}
  assert len(set(map(norm,qs)))==2 and all(norm(q) not in forbidden for q in qs),t['order']
  for kind,q in [('native',t['native']['question']),*[("new_paraphrase",q) for q in qs]]:
   positives.append(dict(t['native'],question=q,role=kind,associated_edit=t['canonical_edit_id'],order=t['order'],reference=''))
 obj=dict(positives=positives,negatives=negatives,frozen_before_thresholds=True,selection='native-only human-authored paraphrases; source-isolated SLAKE train acquisition questions, metadata order, no outcomes',support_status='SUPPORTED' if len(negatives)>=60 and len({r['source_group'] for r in negatives})>=20 else 'UNSUPPORTED')
 p=ROOT/'private/CAL_ROUTE.json';assert not p.exists();p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
 summary=dict(status=obj['support_status'],native=24,new_paraphrases=len(positives)-24,negative_unique_inputs=len(negatives),negative_source_groups=len({r['source_group'] for r in negatives}),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),no_formal_probes=True,whole_library_roles=True,image_preserved=True)
 (ROOT/'public/CAL_ROUTE_AUDIT.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
