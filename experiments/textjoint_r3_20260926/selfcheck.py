"""Small runnable checks for rejection, missing pairs and atomic reservations."""
import os,sys,tempfile,threading
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(r),str(r/'source')]
from router_r3 import selfcheck
selfcheck()
import report_r3 as report
row=dict(edit='e',task='T2L',query_id='q',source_group='s',base_judge_key='b')
rs=[dict(row,arm='A',judge_key='a'),dict(row,arm='B',judge_key='x')]
x=report.paired(rs,{'b':True,'a':True},['A','B'],[-1,1])
assert x['status']=='MISSING' and x['metrics']['T2L']['delta_edit_macro'] is None
assert x['metrics']['T2L']['missing_identification_bounds']==[-1,0]
x=report.paired(rs,{'b':True,'a':False,'x':True},['A','B'],[-1,1]);assert x['metrics']['T2L']['delta_edit_macro']==1
import budget
with tempfile.TemporaryDirectory(dir=r/'tmp') as d:
 budget.ROOT=Path(d);budget.write(Path(d)/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0,gpu_seconds_limit=100,historical_gpu_seconds=0,historical_judge_attempt_items=0,judge_submission_attempt_items=0,judge_submission_attempt_items_limit=6000,reservations=[]))
 out=[]
 def reserve(i):
  try:budget.reserve(str(i),'fake',80);out.append(True)
  except TimeoutError:out.append(False)
 ts=[threading.Thread(target=reserve,args=(i,)) for i in range(2)]
 for t in ts:t.start()
 for t in ts:t.join()
 assert sorted(out)==[False,True]
print('PASS: R1, missing paired bounds, concurrent budget reservations')
