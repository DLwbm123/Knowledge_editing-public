"""Two disjoint workers share the frozen V92 budget and completion barrier."""
import json,os,time,traceback
from pathlib import Path
r=Path(os.environ['V81_ROOT'])
def read(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
try:
 while not all((r/f'EXIT_worker_{p}.json').exists() for p in range(2)):
  if (r/'STOP').exists():raise RuntimeError('Worker/guard STOP')
  assert time.time()<read(r/'RESOURCE_LIMITS.json')['hard_end']-300
  time.sleep(5)
 exits=[read(r/f'EXIT_worker_{p}.json') for p in range(2)]
 assert all(x['exit_code']==0 for x in exits)
 assert all((r/f'EVAL_COMPLETE_{p}.json').exists() for p in range(2))
 counts={k:sum(x['counts'][k] for x in exits) for k in exits[0]['counts']}
 assert len(list((r/'outputs').glob('*/*/*.json')))==775
 write(r/'MAIN_COMPLETE.json',dict(status='MODEL_CONSUMERS_COMPLETE_SCORING_PENDING',outputs=775,counts=counts,time=time.time()))
 import sys
 sys.path.insert(0,str(r/'tools'));import review
 review.evaluation_pack(r,775)
except BaseException as e:
 write(r/'FAILURE_controller.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
 write(r/'STOP',dict(reason='CONTROLLER_FAILURE',time=time.time()));raise
finally:
 write(r/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
