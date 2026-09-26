"""Finite dependency queue, no background research expansion or clock resets."""
import os,sys,subprocess,time,signal,traceback
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(ROOT))
from budget import read,write,reserve,close
from datetime import datetime

def epoch(s):return datetime.fromisoformat(s).timestamp()
def gpu(index):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
 row=next(r.split(', ') for r in rows.splitlines() if r.split(', ')[0]==str(index))
 if int(row[2])>=100:raise RuntimeError(f'GPU {index} is not idle; do not disturb other tasks')
 return row[1]
def command(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return ''
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False

def stop_own(pid):
 if command(pid).strip() in ['/root/anaconda3/bin/python /tmp/r4x.py','/root/anaconda3/bin/python /tmp/r4d.py']:os.kill(pid,signal.SIGTERM)

def launch(job,index,seconds):
 jid=job['id'];folder=ROOT/'jobs'/jid
 if (folder/'STATUS.json').exists():
  state=read(folder/'STATUS.json');assert state['job']==job,'Job binding changed'
  if state['status']=='GPU_COMPLETE':return None
  if state['status']=='RUNNING' and (folder/'PROCESS.json').exists():
   receipt=read(folder/'PROCESS.json')
   assert alive(receipt['pid']) and command(receipt['pid']).strip()=='/root/anaconda3/bin/python '+job.get('entry','/tmp/r4x.py')
   return receipt
  raise RuntimeError('Prior failure requires diagnosis, not automatic re-execution')
 uuid=gpu(index);write(folder/'JOB.json',job);reserve(jid,uuid,seconds)
 env=os.environ.copy();env.update(RUN_JOB_JSON=str(folder/'JOB.json'),JOB_ID=jid,PINNED_GPU_UUID=uuid,CUDA_VISIBLE_DEVICES=uuid,TMPDIR=str(ROOT/'tmp'),PYTHONUNBUFFERED='1')
 with (ROOT/'logs'/f'{jid}.log').open('ab') as out:p=subprocess.Popen(['/root/anaconda3/bin/python',job.get('entry','/tmp/r4x.py')],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=p.pid,job=jid,gpu_uuid=uuid,started_epoch=time.time(),reserved_seconds=seconds)
 write(ROOT/f'ACTIVE_GPU{index}.json',receipt)
 time.sleep(.3);assert command(p.pid).strip()=='/root/anaconda3/bin/python '+job.get('entry','/tmp/r4x.py')
 receipt['argv']=command(p.pid).strip();receipt['cwd']=str(Path(f'/proc/{p.pid}/cwd').resolve());write(folder/'PROCESS.json',receipt)
 return receipt

def wait(receipts):
 active=[r for r in receipts if r]
 while active:
  now=time.time();manifest=read(ROOT/'RUN_MANIFEST.json')
  stop=(ROOT/'STOP').exists() or now>=epoch(manifest['deadline_at']) or (ROOT/'JUDGE_FAILURE.json').exists()
  for r in list(active):
   if stop or now-r['started_epoch']>=r['reserved_seconds']-5:
    stop_own(r['pid']);time.sleep(2)
    if alive(r['pid']):raise RuntimeError('Worker did not exit after stop; do not mark ledger closed')
    close(r['job']);raise TimeoutError('Stop, Judge failure, or residency reservation reached')
   if not alive(r['pid']):
    close(r['job']);status=read(ROOT/'jobs'/r['job']/'STATUS.json')
    assert status['status']=='GPU_COMPLETE',status
    active.remove(r)
  if active:time.sleep(5)

def score_coverage():
 p=ROOT/'private/judge';pending={x.stem for x in (p/'pending').glob('*.json')};scored={x.stem for x in (p/'scores').glob('*.json')};failed=set(read(p/'JUDGE_MISSING_LOCK.json')['keys'])-scored
 ledger=read(ROOT/'RESOURCE_LEDGER.json');flight={k for a in ledger['judge_attempts'] if a['status']=='RESERVED' for k in a.get('keys',[])}-scored-failed
 c=dict(expected=len(pending),scored=len(pending&scored),FAILED_NO_RETRY=len(pending&failed),IN_FLIGHT=len(pending&flight),UNSUBMITTED=len(pending-scored-failed-flight),MISSING=len(pending-scored))
 write(ROOT/'public/JUDGE_COVERAGE.json',c);return c

def wait_scores(canary=False):
 deadline=epoch(read(ROOT/'RUN_MANIFEST.json')['deadline_at'])-600
 while time.time()<deadline:
  if (ROOT/'STOP').exists() or (ROOT/'JUDGE_FAILURE.json').exists():raise RuntimeError('Scorer blocked; preserve current results')
  c=score_coverage()
  if not c['UNSUBMITTED'] and not c['IN_FLIGHT']:
   if canary and c['MISSING']:raise RuntimeError('Canary Judge loop has missing decisions')
   return c
  time.sleep(10)
 raise TimeoutError('Scoring hard deadline reserve reached')

