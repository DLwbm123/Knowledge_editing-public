"""Run only after source review. Evaluation strings never reach author/reviewer.

Runs a read-only deterministic check on pro5000; returns anonymous counts only.
It must never send candidate repairs or evaluation text back to the author.
"""
import json
import os
from pathlib import Path
import subprocess
import hashlib
W=Path(os.environ['WORKSPACE']);S=W/'.local_run'
REPORT_DIR=W/'reports/purew_cpol146_v1_20261013'

def main():
    candidates=json.loads((S/'CANDIDATES.json').read_text())
    source=json.loads((S/'SOURCE_QUALIFICATION.json').read_text())
    code='''import json,sys,unicodedata,collections
from pathlib import Path
R=Path('/data/bmw/Knowledge_editing/outputs/purew-original146-v1-20261012/run')
rows=json.loads(sys.stdin.readline())
q=json.loads((R/'private/BENCHMARK146_QUEUE.json').read_text())['tasks']
l=json.loads((R/'private/EVAL_LEDGER.json').read_text())
def norm(x):return ''.join(c for c in unicodedata.normalize('NFKC',x).casefold() if c.isalnum())
evals=collections.defaultdict(list)
for t in q:
 for e in t['events']:
  if e['task']=='T0':continue
  for k in e['all_probe_query_ids']:
   x=l['queries'][k];evals[e['task']].append((x['image_sha256'],x['question']))
counts=collections.Counter();failed=set()
for r in rows:
 t=q[r['order']-1]
 for style in ('syntax','terminology','interrogative','context'):
  a=r[style]
  for role,rs in evals.items():
   exact=any(a==b for _,b in rs)
   normalized=any(norm(a)==norm(b) for _,b in rs)
   full=any(im==t['native']['image_sha256'] and norm(a)==norm(b) for im,b in rs)
   if exact:counts[role+'_exact_text']+=1
   if normalized:counts[role+'_normalized_text']+=1;failed.add((r['order'],style))
   if full:counts[role+'_normalized_full_input']+=1
print(json.dumps(dict(candidate_count=len(rows)*4,collision_candidates=len(failed),counts=dict(counts),screen_all_eligible_and_noneligible=True,source_and_answer_not_changed=True,evaluation_labels_read_for_training=False)))
'''
    entry="/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"
    request=json.dumps(code)+'\n'+json.dumps(candidates)+'\n'
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',entry],input=request,text=True,capture_output=True,check=True,timeout=180)
    check=json.loads(p.stdout)
    result=dict(source_review_status=source['status'],semantic_or_style_failures=source['failed_candidates'],collision_screen=check,
        candidate_binding=source['candidate_binding'],source_binding=source['source_binding'],review_binding=source['semantic_review_binding'],
        status='PASS' if source['failed_candidates']==0 and check['collision_candidates']==0 else 'BLOCKED_DATA_QUALIFICATION',
        candidates_repaired=0,evaluation_examples_disclosed_to_author=0,training_launched=False)
    (S/'QUALIFICATION_FINAL.json').write_text(json.dumps(result,indent=2))
    (REPORT_DIR/'QUALIFICATION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
