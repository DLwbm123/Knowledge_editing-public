"""Freeze source-only character groups, then a finite pre-edit locality pool."""
import inspect,json,subprocess
from pathlib import Path

def remote_freeze(payload):
 import json,time,re,shutil
 from pathlib import Path
 outputs=Path('/data/bmw/Knowledge_editing/outputs');parent=outputs/'full-context-summary8-v1-20261018/run';prior=outputs/'vigor-editperf8-v1-20261019/attempt2';root=outputs/'vigor-factF3-v1-20261020'
 if root.exists():raise FileExistsError('Preparation already frozen, no replay')
 if shutil.disk_usage(outputs).free<9*1024**3:raise OSError('Storage reserve')
 root.mkdir();(root/'private').mkdir();(root/'public').mkdir()
 def write(name,v):(root/name).write_text(json.dumps(v,indent=2)+'\n')
 probe=root/'private/.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
 write('private/STORAGE_BASELINE.json',dict(existed_before=False,baseline_bytes=0,total_limit=268435456,free_bytes=shutil.disk_usage(outputs).free))
 m=json.loads((prior/'private/MANIFEST.json').read_text())
 # This annotation step sees only the three original edit sources, before evaluation rows.
 char_groups=payload['private_source_groups'];facts=[]
 for task,groups in zip(m['tasks'][:3],char_groups):
  source=task['record'];target=source['target'];spans=[]
  for start,end in groups:
   assert 0<=start<end<=len(target);spans.append(dict(start=start,end=end))
  assert all(spans[i]['end']<=spans[i+1]['start'] for i in range(len(spans)-1))
  facts.append(dict(alias=task['alias'],source_record=source,source_target=target,source_question=source['question'],groups=spans))
 write('private/SOURCE_FACT_GROUPS.json',dict(frozen_epoch=time.time(),annotation_basis='original_edit_source_only',facts=facts))
 write('public/SOURCE_FACT_GROUPS.json',dict(frozen=True,annotation_basis='original_edit_source_only',extra_semantic_annotation=True,groups=[dict(sample=x['alias'],group_count=len(x['groups'])) for x in facts],groups_changed_after_evaluation=False,implementation_commit=payload['commit']))
 # Separate evaluation-only lane: cannot alter source groups after this point.
 ledger=json.loads((parent/'private/EVAL_LEDGER.json').read_text());queries=ledger['queries'];queue=json.loads((parent/'private/BENCHMARK146_QUEUE.json').read_text())['tasks'][:3]
 cal=json.loads((outputs/'vigor-medical-preparation-v1-20261018/run/attempt2/private/manifest.json').read_text());excluded=set(cal['calibration_source_groups']);source_groups={t['native']['source_group'] for t in queue}
 propagation=set()
 for t in m['tasks'][:3]:
  for entry in t['panel']:
   if entry['task'] in ['T0','T1G','T2G']:
    q=m['queries'][entry['query']];propagation.add((q['image_sha256'],q['question'].strip().casefold()))
 allow=re.compile(r'\b(modality|plane|view|orientation|imaging|scan|mri|ct|organ|body part|anatomical|tissue)\b',re.I)
 deny=re.compile(r'\b(disease|abnormal|finding|lesion|edema|tumou?r|function|enzyme|digest|nutrient|cause|treat|diagnos|patholog)\w*',re.I)
 picked=[];seen=set();near_counts={};unrelated_groups=set()
 def mapped(path):
  if '/derived_inputs/' in path:return str(parent/'private/derived_inputs'/path.split('/derived_inputs/',1)[1])
  if '/DataP/' in path:return str(Path('/data/bmw/DataP')/path.split('/DataP/',1)[1])
  return path
 def append(qid,q,category,anchor):
  key=(q['image_sha256'],q['question'].strip().casefold())
  if key in seen or key in propagation:return False
  if q['source_group'] in excluded:return False
  seen.add(key);row=dict(q,image_path=mapped(q['image_path']));assert Path(row['image_path']).is_file()
  picked.append(dict(alias='L'+str(len(picked)+1).zfill(3),original_query_id=qid,category=category,anchor_case=anchor,query=row));return True
 for task,source in zip(m['tasks'][:3],queue):
  count=0;sg=source['native']['source_group']
  for qid,q in sorted(queries.items()):
   if q['source_group']!=sg or not allow.search(q['question']) or deny.search(q['question']):continue
   if append(qid,q,'near',task['alias']):count+=1
   if count>=4:break
  near_counts[task['alias']]=count
 for qid,q in sorted(queries.items()):
  sg=q['source_group']
  if sg in source_groups or sg in unrelated_groups or sg in excluded:continue
  if ledger['Base_correctness'].get(qid) is not True:continue
  if append(qid,q,'unrelated',None):unrelated_groups.add(sg)
  if len(unrelated_groups)>=12:break
 assert len(picked)<=24
 write('private/LOCALITY_CANDIDATES.json',dict(frozen_epoch=time.time(),selection_uses_post_edit_outputs=False,rows=picked))
 public=dict(status='SOURCE_AND_CANDIDATES_FROZEN',candidate_count=len(picked),near_candidates=near_counts,unrelated_candidates=len(unrelated_groups),near_target=6,unrelated_target=6,current_Base_scoring_pending=True,calibration_source_excluded=True,final_selection='first_two_Base_correct_per_anchor_and_first_six_unrelated',rows=[dict(alias=x['alias'],category=x['category'],anchor_case=x['anchor_case']) for x in picked])
 write('public/PREPARATION.json',public);return public

def prepare():
 r=Path(__file__).resolve().parents[2]
 # Annotation offsets are private, source strings never serialized into public code.
 annotations=json.loads((r/'.local_run/factF3_source_groups.json').read_text())
 payload=dict(private_source_groups=annotations,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip())
 code=inspect.getsource(remote_freeze)+'\nprint(json.dumps(remote_freeze(json.loads(sys.stdin.readline()))))'
 return subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',capture_output=True,text=True,timeout=60)
if __name__=='__main__':
 x=prepare();print(x.stdout);print(x.stderr);raise SystemExit(x.returncode)
