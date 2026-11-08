"""Finite original-clock supervisor; only this experiment's process groups."""
import json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['LONG_ROOT']);active=None
cfg=json.loads((ROOT/'CONFIG.private.json').read_text());clock=json.loads((ROOT/'CLOCK.json').read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x));t.replace(p)
def health(phase):
 size=0
 for base in [ROOT,Path(cfg['neutral'])]+[Path(p) for p in cfg.get('additional_owned_roots',[])]:
  for directory,dirs,files in os.walk(base,followlinks=False):
   for name in files:
    p=Path(directory)/name
    try:
     if not p.is_symlink():size+=p.stat().st_blocks*512
    except FileNotFoundError:pass
 x=dict(time=time.time(),phase=phase,bytes=size,hard_end=clock['hard_end'])
 for p in [ROOT/'HEALTH.json',ROOT/'MDU/TRAIN_GUARD_HEALTH.json',ROOT/'MDU/EVAL_GUARD_HEALTH.json']:
  if p.parent.exists():write(p,x)
 assert size+2*1024**3<12_000_000_000 and shutil.disk_usage(ROOT).free>8*1024**3
 assert time.time()<clock['hard_end']-1200 and not (ROOT/'STOP.json').exists()
def terminate(*_):
 if active and active.poll() is None:
  os.killpg(active.pid,signal.SIGTERM)
  try:active.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
 raise InterruptedError('finite supervisor stop')
signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
def run(phase,name,extra=None,gpu=True):
 global active
 while gpu:
  row=subprocess.check_output(['nvidia-smi','-i',cfg['gpu']['index'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
  assert row[0].strip()==cfg['gpu']['uuid'];health('WAIT_'+phase)
  if int(row[1])>=60000:break
  time.sleep(10)
 health(phase)
 env=dict(os.environ,CURRENT_SOURCE=str(ROOT/'tools'/name),**(extra or {}))
 if cfg.get('peft_path'):env['PYTHONPATH']=cfg['peft_path']+os.pathsep+env.get('PYTHONPATH','')
 with (ROOT/(phase+'.log')).open('x') as log:
  active=subprocess.Popen([cfg['python'],str(Path(cfg['neutral'])/'w.py')],env=env,cwd='/data/bmw',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
   preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3)))
 write(ROOT/(phase+'_IDENTITY.private.json'),dict(pid=active.pid,time=time.time()))
 while active.poll() is None:health(phase);time.sleep(5)
 write(ROOT/(phase+'_EXIT.json'),dict(code=active.returncode,time=time.time()));assert active.returncode==0,phase

def main():
 (ROOT/'GUARD_ONCE').touch(exist_ok=False)
 run('INPUT','pixel_preflight.py',gpu=False)
 run('BASE','base.py')
 while not (ROOT/'BASE_MASK_FROZEN.private.json').exists():health('WAIT_CORE_BASE');time.sleep(10)
 run('SETUP','setup_train.py',gpu=False)
 if json.loads((ROOT/'COHORT_FROZEN.json').read_text())['admitted']==0:
  write(ROOT/'COMPLETE.json',dict(status='NO_CONFIRMED_BASE_ERRORS'));return
 extra=dict(EXPERIMENT_ROOT=str(ROOT/'MDU'),T5_REMOTE_ROOT=str(ROOT/'MDU'))
 run('MDU_TRAIN','initialize.py',extra)
 run('MDU_EVAL','bank.py',extra)
 write(ROOT/'MDU/GENERATION_COMPLETE.json',dict(status='COMPLETE'))
 for method in ['BALANCEDIT','GRACE','LORA']:
  (ROOT/method).mkdir();run(method,'comparator.py',dict(COMPARE_ROOT=str(ROOT),COMPARE_METHOD=method))
 write(ROOT/'GENERATION_COMPLETE.json',dict(status='ALL_GENERATION_COMPLETE',methods=['MDU','BALANCEDIT','GRACE','LORA']))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/'GUARD_FAILURE.private.json',dict(time=time.time(),error=traceback.format_exc()))
  if active and active.poll() is None:terminate()
  raise
