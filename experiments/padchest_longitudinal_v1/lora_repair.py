"""Zero-edit PEFT dependency repair; no completed trajectory or score replay."""
import json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['LONG_ROOT']);child=None
cfg=json.loads((ROOT/'CONFIG.private.json').read_text());clock=json.loads((ROOT/'CLOCK.json').read_text())
def write(name,x):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def stop(*_):
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 raise InterruptedError('original-clock dependency repair stopped')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def main():
 job=ROOT/'LORA';archive=job/'engineering_repair1';archive.mkdir()
 failure=json.loads((job/'FAILURE.json').read_text());assert "No module named 'peft'" in failure['error']
 assert not list(job.glob('T*/training/*.json')) and not list(job.glob('T*/outputs/*.json')) and not (job/'METHOD_CONFIG.json').exists()
 dependency=os.environ['LONG_PEFT_PATH'];sys.path.insert(0,dependency)
 import peft
 assert peft.__version__=='0.15.2'
 write('LORA_REPAIR1_ADMISSION.private.json',dict(time=time.time(),original_clock=clock,prior_completed_edits=0,prior_generations=0,
    dependency=dependency,PEFT=peft.__version__,read_only_existing_dependency=True,other_methods_unchanged=True,old_failure_preserved=True))
 for name in ['FAILURE.json','EXIT.json','WORKER_ONCE','runtime']:
  p=job/name
  if p.exists():p.rename(archive/name)
 gpu=cfg['gpu'];row=subprocess.check_output(['nvidia-smi','-i',gpu['index'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).split(',')
 assert row[0].strip()==gpu['uuid'] and int(row[1])>=60000
 env=dict(os.environ,PYTHONPATH=dependency+os.pathsep+os.environ.get('PYTHONPATH',''),COMPARE_ROOT=str(ROOT),COMPARE_METHOD='LORA',
    COMPARE_HEALTH_FILE='LORA_REPAIR1_HEALTH.json',CURRENT_SOURCE=str(ROOT/'tools/comparator.py'))
 global child
 with (ROOT/'LORA_REPAIR1.log').open('x') as log:
  child=subprocess.Popen([cfg['python'],str(Path(cfg['neutral'])/'w.py')],env=env,cwd='/data/bmw',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
    preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3)))
 write('LORA_REPAIR1_IDENTITY.private.json',dict(pid=child.pid,time=time.time()))
 while child.poll() is None:
  used=0
  for base in [ROOT,Path(cfg['neutral'])]:
   for directory,dirs,files in os.walk(base,followlinks=False):
    for name in files:
     p=Path(directory)/name
     try:
      if not p.is_symlink():used+=p.stat().st_blocks*512
     except FileNotFoundError:pass
  fs=os.statvfs(ROOT)
  write('LORA_REPAIR1_HEALTH.json',dict(time=time.time(),phase='LORA_DEPENDENCY_REPAIR',bytes=used,hard_end=clock['hard_end']))
  assert used+2*1024**3<12_000_000_000 and fs.f_bavail*fs.f_frsize>8*1024**3 and time.time()<clock['hard_end']-1200
  time.sleep(5)
 write('LORA_REPAIR1_EXIT.json',dict(code=child.wait(),time=time.time(),clock=clock))
 assert child.returncode==0
 assert (ROOT/'LORA/T5/GENERATION_COMPLETE.json').exists()
 write('GENERATION_COMPLETE.json',dict(status='ALL_GENERATION_COMPLETE_WITH_RECORDED_DEPENDENCY_REPAIR',methods=['MDU','BALANCEDIT','GRACE','LORA'],original_failure_preserved=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  write('LORA_REPAIR1_FAILURE.private.json',dict(time=time.time(),error=traceback.format_exc()))
  if child and child.poll() is None:stop()
  raise
