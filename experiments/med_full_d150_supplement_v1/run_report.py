"""Publish same-support scoring aggregates after all supplement guards exit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

w=Path(os.environ['FIXED_WORKTREE']);root=Path(os.environ['FIXED_SCORE_ROOT'])
deadline=float(os.environ['EXPERIMENT_DEADLINE'])
if time.time()>=deadline-60:raise TimeoutError('original deadline')
if json.loads((root/'GUARD_EXIT.json').read_text())['code']!=0:raise ValueError('score guard failed')
script="""import json\nfrom pathlib import Path\np=Path(%r)\na=json.loads((p/'SPAN_LOCK.private.json').read_text())['records'];z=json.loads((p/'SOURCE_IDENTITY_LOCK.private.json').read_text())['unknown_owners']\nprint(json.dumps({'matched':[x['order'] for x in a if x['status']=='WEAK_TOKEN_MATCH'],'unmatched':[x['order'] for x in a if x['status']!='WEAK_TOKEN_MATCH'],'unresolved':z}))\n""" % os.environ['EXPERIMENT_ROOT']
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],input=script,text=True,capture_output=True,timeout=30)
if p.returncode:raise RuntimeError('frozen coverage identity unavailable')
coh=json.loads(p.stdout)
if [len(coh[k]) for k in ('matched','unmatched','unresolved')]!=[25,76,49] or len(set().union(*(set(x) for x in coh.values())))!=150:
    raise ValueError('frozen cohort identities differ')
sys.path.insert(0,str(w/'experiments/med_full_d150_supplement_v1'))
import report
result=report.build(root,coh)
dest=w/'reports/med_full_d150_supplement_v1_20261102';report.write_report(result,dest)
completion=json.loads((root/'COMPLETE.json').read_text());admit=json.loads((root/'PAIRING_ADMISSION.json').read_text())
audit={'status':result['status'],'source_experiment':'f7801657 and 34d10808; original reports unchanged',
       'frozen_events':2148,'candidate_generation':0,'candidate_training':0,
       'original_valid_reused':1376,'requested_targets':53,'history_unverified':48,'conflict_adjudications':5,
       'actual_new_attempt_items':completion['new_attempt_items'],'remaining_unknown':completion['unknown'],
       'score_worker_exit':json.loads((root/'WORKER_EXIT.json').read_text())['code'],
       'score_guard_exit':json.loads((root/'GUARD_EXIT.json').read_text())['code'],
       'method_model':'gpt-6-astra','reasoning':'medium','cli_version':'0.166.0-alpha.5.1','immutable_snapshot':'unknown',
       'original_clock_hard_end_utc':'2026-11-03T06:39:53Z','no_automatic_retry':True,
       'comparison_input_admission':admit,
       'report_only_failure':'Initial report import collided by module name; fixed import and recomputed frozen judgments without training, generation or scoring retry.',
       'limitations':['48 historical attempt statuses cannot be verified; historical repeats cannot be ruled out',
          'five historical conflicts receive one independent adjudication; original discordance retained, sensitivity reported',
          '25 MED+D and 125 original MED experts','no OFF support; exposed development data',
          'BalancEdit and MED+D use different numerical precision']}
with (dest/'EXECUTION_AUDIT.json').open('x') as f:json.dump(audit,f,ensure_ascii=False,indent=2)
(root/'REPORT_COMPLETE.json').write_text(json.dumps({'status':result['status'],'unknown':completion['unknown']}))
print(json.dumps({'status':result['status'],'unknown':completion['unknown'],'attempts':completion['new_attempt_items']}))
