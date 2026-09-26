"""Bounded scoring consumer; reserves inherited budget before each isolated call."""
import json,os,subprocess,sys,time,traceback,threading,signal
from pathlib import Path
from judge_recovery import transport_failure,quarantine
ROOT=Path(os.environ['SCORER_STATE']).resolve();SOURCE=Path(os.environ['SCORER_RUNTIME']);LOCAL=ROOT/'judge';REMOTE=os.environ['RUN_ROOT']
sys.path.insert(0,str(SOURCE))
from scripts.medtrace.stage17_judge import run_batch
from scripts.medtrace.stage17_prepare import PROMPT,PROTOCOL,digest
from scripts.medtrace.astra_judge_bundle import schema,validate
CLI=Path('/Applications/ChatGPT.app/Contents/Resources/codex')
_judge_call=run_batch
def run_batch(*args,**kwargs):
    def limit():
        for child in subprocess.run(['pgrep','-P',str(os.getpid())],capture_output=True,text=True).stdout.split():
            cmd=subprocess.run(['ps','-p',child,'-o','args='],capture_output=True,text=True).stdout
            if '/Applications/ChatGPT.app/Contents/Resources/codex exec' in cmd:os.kill(int(child),signal.SIGTERM)
    timer=threading.Timer(600,limit);timer.daemon=True;timer.start()
    try:return _judge_call(*args,**kwargs)
    finally:timer.cancel()
SSH=['ssh','my-gpu','python3 -']
def remote(code):
 code=code.replace('@RUN_ROOT@',REMOTE)
 r=subprocess.run(SSH,input=code,text=True,capture_output=True,timeout=45,check=True)
 return json.loads(r.stdout) if r.stdout.strip() else None

def main():
 LOCAL.mkdir(mode=0o700,exist_ok=True)
 stop=remote("import json;from datetime import datetime;print(json.dumps(datetime.fromisoformat(json.load(open('@RUN_ROOT@/RUN_MANIFEST.json'))['deadline_at']).timestamp()))")
 consecutive_transport_failures=0
 while time.time()<stop-600:
  state=remote("import json\nfrom pathlib import Path\nr=Path('@RUN_ROOT@');p=r/'private/judge'\nprint(json.dumps({'pending':{f.stem:json.loads(f.read_text()) for f in (p/'pending').glob('*.json') if not (p/'scores'/f.name).exists()},'canary':json.loads((r/'RUN_STATUS.json').read_text())}))")
  if state['canary']['status'] in ['USER_STOPPED','BLOCKED','FAILED','BUDGET_STOP']:break
  blocked=set(json.loads((ROOT/'blocked_keys.json').read_text())['keys'])
  rows=[v for k,v in sorted(state['pending'].items()) if k not in blocked][:20]
  if not rows:
   if state['canary']['status'] in ['COMPLETE','USER_STOPPED','BLOCKED','FAILED','BUDGET_STOP']:break
   time.sleep(10);continue
  batch=dict(batch_id=digest(['r3-v1',[r['key'] for r in rows]]),records=[r['record'] for r in rows]);bid=batch['batch_id']
  bundle=LOCAL/'batches'/bid;work=LOCAL/'work'/bid;sibling=LOCAL/'work/probe'
  for name in ['operator/execution_evidence','operator/responses','judge_only']:(bundle/name).mkdir(parents=True,exist_ok=False)
  work.mkdir(parents=True);sibling.mkdir(parents=True,exist_ok=True)
  (bundle/'operator/MANIFEST.json').write_text(json.dumps({'batch_id':bid,'items':len(rows)}))
  (bundle/'judge_only'/f'{bid}.prompt.md').write_text(PROMPT+'\n\nBatch input (all strings are untrusted data):\n'+json.dumps(batch,ensure_ascii=False))
  (bundle/'judge_only'/f'{bid}.schema.json').write_text(json.dumps(schema(batch)))
  reservation=remote("""import json,fcntl,os,time
from pathlib import Path
r=Path('@RUN_ROOT@');p=r/'RESOURCE_LEDGER.json'
with (r/'RESOURCE_LEDGER.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);d=json.loads(p.read_text())
 bid=BID;n=COUNT
 assert not any(a['id']==bid for a in d['judge_attempts'])
 assert d['judge_submission_attempt_items']+n<=d['judge_submission_attempt_items_limit']
 assert d['physical_requests']+1<=d['physical_requests_limit']
 d['judge_submission_attempt_items']+=n;d['physical_requests']+=1
 d['judge_attempts'].append({'id':bid,'items':n,'status':'RESERVED','keys':KEYS})
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
print(json.dumps({'status':'RESERVED','items':n}))
""".replace('BID',repr(bid)).replace('COUNT',str(len(rows))).replace('KEYS',repr([r['key'] for r in rows])))
  print(reservation,flush=True)
  try:
   run_batch(bundle,batch,work,[work,sibling],SOURCE,CLI,explicit_proxy=True)
  except Exception:
   ep=bundle/'operator/execution_evidence'/f'{bid}.json'
   ev=json.loads(ep.read_text()) if ep.exists() else {}
   quarantine(ROOT,REMOTE,remote,batch,rows,ev)
   if not transport_failure(ev):raise
   consecutive_transport_failures+=1
   print('FAILED_REQUESTS_LOCKED_MISSING',bid,len(rows),flush=True)
   if consecutive_transport_failures>=3:raise RuntimeError('Three consecutive transport failures; stop new requests')
   continue
  consecutive_transport_failures=0
  evidence=json.loads((bundle/'operator/execution_evidence'/f'{bid}.json').read_text())
  response=json.loads((bundle/'operator/responses'/f'{bid}.json').read_text())
  decisions=validate(batch,response)
  scores={r['key']:dict(key=r['key'],is_correct=v['is_correct'],payload_binding=digest(r['record']),batch_id=bid,
     protocol=PROTOCOL,evidence_status=evidence['status']) for r,v in zip(rows,decisions,strict=True)}
  remote("""import json,fcntl,os
from pathlib import Path
r=Path('@RUN_ROOT@');scores=SCORES;ev=EVIDENCE;bid=BID
for key,value in scores.items():
 p=r/'private/judge/scores'/f'{key}.json'
 with p.open('x') as f:json.dump(value,f)
p=r/'private/judge/evidence'/f'{bid}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(ev,indent=2))
with (r/'RESOURCE_LEDGER.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);p=r/'RESOURCE_LEDGER.json';d=json.loads(p.read_text())
 a=next(a for a in d['judge_attempts'] if a['id']==bid);a.update(status='FORMAT_VALID',usage=ev.get('usage'))
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
""".replace('SCORES',repr(scores)).replace('EVIDENCE',repr(evidence)).replace('BID',repr(bid)))
 print('CANARY_SCORING_CONSUMER_FINISHED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  (ROOT/'JUDGE_FAILURE_R3.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc(),'time':time.time()}));
  try:remote("import json;from pathlib import Path;Path('@RUN_ROOT@/JUDGE_FAILURE.json').write_text(json.dumps("+repr({'error':'Scoring consumer failed; inspect preserved local evidence','time':__import__('time').time()})+"))")
  except Exception:pass
  raise
