"""Reuse only R3 R0 consumers with full saved execution/input/bank bindings."""
import os,json,hashlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);PREV=ROOT.parents[1]/'textjoint-r3-20260926/run'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def main():
 tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];assert sha(ROOT/'private/TASKS_R2_LOCKED.json')==sha(PREV/'private/TASKS_R2_LOCKED.json')
 assets={(a['order'],a['arm']):a for a in read(ROOT/'private/CHECKPOINT_INVENTORY.json')};byedit={t['canonical_edit_id']:t for t in tasks};evidence=[];count=0
 for job in (PREV/'jobs').iterdir():
  if not job.name.startswith(('dev-','verify-')) or not (job/'JOB.json').exists():continue
  j=read(job/'JOB.json')
  if j['mode'] not in ['single','sequential']:continue
  assert read(job/'STATUS.json')['status']=='GPU_COMPLETE'
  for f in job.glob('*/p*/CONSUMERS.json'):
   rows=read(f)
   if not rows or rows[0]['mode']=='diagnostic' or not rows[0]['arm'].endswith('_R0'):continue
   for row in rows:
    saved=read(PREV/'private/generations'/f"{row['execution_id']}.json");b=saved['execution_binding']
    assert saved['output']==row['output'] and b['router']==dict(kappa=1.,mu=0.) and b['prefix']==row['prefix'] and not b['forced_diagnostic']
    t=byedit[row['edit']];query=next(q for q in t['evaluation'] if q['task']==row['task'] and q['query_id']==row['query_id'])
    assert b['input']['question']==query['question'] and b['input']['image_source_sha256']==query['image_sha256']
    assert b['input']['generation']['max_new_tokens']==1024
    arm=j['writer']
    for expert in b['bank']:
     et=byedit[expert['edit']];assert expert['sha256']==assets[et['order'],arm]['sha256'] and expert['actual_optimizer_steps']==80
    assert len(b['bank'])==(1 if row['mode']=='single' else row['prefix'])
    record=read(ROOT/'private/judge/pending'/f"{row['judge_key']}.json")['record']
    assert record['question']==query['question'] and record['gold_answer']==query['reference'] and record['raw_base_answer']==row['output']['raw_answer']
   dest=ROOT/'jobs'/('reuse-'+job.name)/f.relative_to(job);write(dest,rows)
   write(ROOT/'jobs'/('reuse-'+job.name)/'STATUS.json',dict(status='GPU_COMPLETE',job=j,reused_from='R3',new_GPU_seconds=0))
   evidence.append(dict(source=str(f),destination=str(dest),source_sha256=sha(f),consumers=len(rows)));count+=len(rows)
 write(ROOT/'private/REFERENCE_REUSE_PROOF.json',evidence);write(ROOT/'public/REFERENCE_REUSE_AUDIT.json',dict(status='PASS',consumers=count,full_execution_bindings_verified=True,checkpoint_hashes_verified=True,identical_panel_and_Judge_input=True,new_generation=0))
 print('REUSE_VERIFIED',count)
if __name__=='__main__':main()
