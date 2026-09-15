"""Finite incremental copy of the last rented-host phase; no GPU actions."""
import json,os,shlex,subprocess,time
from pathlib import Path
cfg=json.loads(Path(os.environ['JOB_CONFIG']).read_text());local=Path(cfg['local']);status=local/'EVACUATION.json'
def state(**kw):
 kw['at']=time.time();tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(kw,indent=2)+'\n');tmp.replace(status)
ssh=cfg['source_ssh'];dst=cfg['destination_ssh'];phase='private/grace_sequential'
tries=0
while True:
 try:
  program='from pathlib import Path\nimport json\nr=Path('+repr(cfg['source_root'])+')\nprint(json.dumps(dict(complete=(r/"private/grace_sequential/COMPLETE.json").exists(),cleanup=(r/"private/grace_sequential/CLEANUP.json").exists())))\n'
  observed=json.loads(subprocess.check_output([*ssh,'python3 -'],input=program,text=True,timeout=30))
  args=['rsync','-az','--partial','--timeout=90','--include=*/','--include=*.json','--include=*.jsonl','--exclude=*']
  subprocess.run([*args,'-e',shlex.join(ssh[:-1]),ssh[-1]+':'+cfg['source_root']+'/'+phase+'/',str(local/'runs/job-20260914-baselines'/phase)+'/'],check=True)
  subprocess.run([*args,'-e',shlex.join(dst[:-1]),str(local/'runs/job-20260914-baselines'/phase)+'/',dst[-1]+':'+cfg['destination_root']+'/'+phase+'/'],check=True)
  state(status='LATEST_OUTPUTS_COPIED',source_complete=observed['complete'],source_cleanup=observed['cleanup'],destination=cfg['destination_root'])
  if observed['complete'] and observed['cleanup']:
   program='''from pathlib import Path
import json,sys,time
r=Path(ROOT);sys.path.insert(0,SOURCE)
from scripts.medtrace.stage17_external import validate_phase
from scripts.medtrace.stage17_campaign import prefixes,query_ids
cfg=json.loads((r/'private/DISPATCH.json').read_text());done=validate_phase(r,cfg,'sequential',method='grace')
ledger=json.loads(Path(LEDGER).read_text());tasks={t['edit_id']:t for t in ledger['tasks']}
counts=dict(native=0,panel=0)
for i,ident in enumerate(ledger['main_T0'],1):
 p=r/'private/grace_sequential'/f'e{i:03d}'
 assert (p/'COMPLETE.json').is_file() and (p/'TRAINING.json').is_file()
 native=list((p/'native').glob('*.json'));panel=list((p/'panel').glob('*.json'))
 expect=len(set(q for eid in ledger['main_T0'][:i] for q in query_ids(tasks[eid]))) if i in prefixes(cfg['N']) else 0
 assert len(native)==1 and len(panel)==expect
 counts['native']+=len(native);counts['panel']+=len(panel)
assert counts==dict(native=146,panel=3213)
receipt=dict(status='FINAL_TRANSFER_VALIDATED',counts=counts,phase=done,at=time.time(),snapshot_checkpoint_cleanup='PENDING_STRICT_LIFECYCLE_CHECK')
Path(RECEIPT).write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(dict(status=receipt['status'],counts=counts)))
'''
   values={'ROOT':cfg['destination_root'],'SOURCE':'/tmp/job-4921/source','LEDGER':cfg['handoff_root']+'/runs/medtrace_stage17_20260913_r01/private/COHORT_AND_SUPPORT_LEDGER.json','RECEIPT':cfg['handoff_root']+'/GRACE_FINAL_TRANSFER.json'}
   for key,value in values.items():program=program.replace(key,repr(value))
   result=json.loads(subprocess.check_output([*dst,cfg['destination_python']+' -'],input=program,text=True,timeout=60))
   state(status='FINAL_TRANSFER_VALIDATED',counts=result['counts'],snapshot_checkpoint_cleanup='PENDING_STRICT_LIFECYCLE_CHECK')
   break
  tries=0
 except Exception as error:
  tries+=1;state(status='RETRY_WAIT' if tries<3 else 'SOURCE_OR_TRANSFER_UNAVAILABLE',error=repr(error),consecutive_failures=tries,last_snapshot_preserved=True)
  if tries>=3:raise
 for _ in range(5):time.sleep(60)
