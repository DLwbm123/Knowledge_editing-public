"""The user waiver removes time/total GPU caps, never STOP or lease checks."""
import json,os,tempfile,time,importlib
from pathlib import Path

def main():
 with tempfile.TemporaryDirectory() as td:
  os.environ['RUN_ROOT']=td
  r=importlib.import_module('resources');r.ROOT=Path(td)
  def put(name,data):(r.ROOT/name).write_text(json.dumps(data))
  put('EXPERIMENT_LOCK.json',dict(deadline_at='2000-01-01T00:00:00+00:00',no_new_training_after='2000-01-01T00:00:00+00:00',gpu_seconds_limit=1))
  put('RESOURCE_LEDGER.json',dict(current_gpu_seconds=100,gpu_sessions=[]))
  try:r.check()
  except TimeoutError:pass
  else:raise AssertionError('Original deadline must hold without waiver')
  put('USER_LIMIT_WAIVER.json',dict(wallclock_gpu_limits_waived=True))
  r.check();r.check(training=True)
  (r.ROOT/'STOP').touch()
  try:r.check()
  except TimeoutError:pass
  else:raise AssertionError('User STOP must remain effective')
  (r.ROOT/'STOP').unlink()
  put('RESOURCE_LEDGER.json',dict(current_gpu_seconds=100,gpu_sessions=[dict(pid=os.getpid(),started_epoch=time.time()-10,reserved_seconds=1)]))
  try:r.check()
  except TimeoutError as e:assert 'lease' in str(e)
  else:raise AssertionError('Worker lease must remain effective')
 print('PASS: original caps, authorized waiver, user STOP and lease')
if __name__=='__main__':main()
