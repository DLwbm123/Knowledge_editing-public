import json,os,sys,time,sqlite3,inspect,collections
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);env=json.loads((R/'private/LAUNCH_ENV.json').read_text());os.environ.update(env);sys.path.insert(0,str(R/'private/tools'))
import factorial_protocol as f
f.admit(R)
import factorial_runtime as w
assert Path(w.w.__file__).parent==R/'private/tools'
assert 'partitions' in inspect.signature(w.w.train).parameters
p=w.w.packet();assert w.w.contract.source_admission(p)['status']=='PASS'
assert len(w.w.tasks()[0::2])==len(w.w.tasks()[1::2])==73
for t in w.w.tasks():
 for row in w.w.trajectory.source_rows(p,t['order']):assert Path(row['image_path']).is_file()
from m13_only_continue import complete
for arm in ['C','M13']:
 for owner in range(1,147):complete(R,arm,owner)
sys.path.insert(0,str(R/'private/tools'))
import selected_report146 as report
O=Path(f.read(R/'private/FACTORIAL_ADMISSION1.json')['evaluation_parent']);records={};scores={}
with sqlite3.connect('file:'+str(O/'private/judge_cm146_astra_medium/queue.sqlite')+'?mode=ro',uri=True) as db:
 for path,status,correct in db.execute('select c.path,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
  d=f.read(Path(path))
  if d['mode'] not in ['base','single']:continue
  k=tuple(d[x] for x in ['method','mode','prefix','owner'])+(d['binding']['input']['query_id'],);records[k]=d;scores[k]=correct if status=='FORMAT_VALID' else None
result,_=report.calculate(f.read(R/'private/ORIGINAL146_TASKS.json'),f.read(R/'private/ORIGINAL_BASE_MASK146.json'),records,scores,methods=('C','M13'),single_only=True)
old=f.read(O/'public/CM_RESULTS.json')
for m in ['C','M13']:assert result['panels'][m]['single_1']==old['panels'][m]['single_1']
receipt=dict(status='PASS',source_admission=True,original_completed_states=292,single_report_exact=True,partitions=[73,73],consumers=f.CONSUMERS,new_generation=5912,epoch=time.time())
w.c.write(R/'private/FACTORIAL_CPU_CHECK1.json',receipt);print(json.dumps(receipt))
