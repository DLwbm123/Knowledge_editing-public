"""Close the original consumer set using one frozen ledger plus one continued actor."""
import json,os,time,traceback
from pathlib import Path
r=Path(os.environ['V81_ROOT'])
def read(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
try:
 while not (r/'EXIT_worker_1.json').exists():
  assert not (r/'STOP').exists() and time.time()<read(r/'RESOURCE_LIMITS.json')['hard_end']-300
  time.sleep(3)
 active=read(r/'EXIT_worker_1.json');assert active['exit_code']==0
 frozen=read(r/'ACTOR_COUNTS_FROZEN_0.json')['counts']
 counts={k:active['counts'][k]+frozen[k] for k in frozen}
 assert len(list((r/'outputs').glob('*/*/*.json')))==775
 write(r/'MAIN_COMPLETE.json',dict(status='MODEL_CONSUMERS_COMPLETE_SCORING_PENDING',outputs=775,counts=counts,time=time.time(),historical_failed_actor_preserved=True))
 import sys
 sys.path.insert(0,str(r/'tools'));import review
 review.evaluation_pack(r,775)
except BaseException as e:
 write(r/'FAILURE_controller.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));write(r/'STOP',dict(reason='RECOVERY2_FAILURE',time=time.time()));raise
finally:write(r/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))
