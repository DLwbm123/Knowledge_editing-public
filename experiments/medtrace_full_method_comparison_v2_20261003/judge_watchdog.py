"""Current-session guard for a launcher that rewrites the CLI executable path."""
import os,json,time,subprocess,signal
from pathlib import Path
from datetime import datetime
root=Path(os.environ['SCORER_STATE']);parent=int(os.environ['SCORER_PARENT']);start=os.environ['SCORER_LSTART']
while True:
 p=subprocess.run(['ps','-p',str(parent),'-o','lstart='],capture_output=True,text=True)
 if p.returncode or p.stdout.strip()!=start:break
 for child in subprocess.run(['pgrep','-P',str(parent)],capture_output=True,text=True).stdout.split():
  cmd=subprocess.run(['ps','-p',child,'-o','args='],capture_output=True,text=True).stdout
  if '--model gpt-6-astra' not in cmd or '--output-last-message '+str(root/'work')+'/' not in cmd:continue
  for ep in (root/'batches').glob('*/operator/execution_evidence/*.json'):
   try:ev=json.loads(ep.read_text())
   except Exception:continue
   if ev.get('status')!='STARTING':continue
   final=root/'work'/ev['batch_id']/'final.json'
   if '--output-last-message '+str(final) not in cmd:continue
   if time.time()-datetime.fromisoformat(ev['started_at_utc']).timestamp()>=600:
    os.kill(int(child),signal.SIGTERM)
    (root/'TIMEOUT_RECEIPT.json').write_text(json.dumps(dict(pid=int(child),batch_id=ev['batch_id'],epoch=time.time(),limit=600)))
 time.sleep(5)
