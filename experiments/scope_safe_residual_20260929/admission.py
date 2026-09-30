"""One-shot, fail-closed lease/budget check; never modifies old owners or starts GPUs."""
import json
from pathlib import Path

def status(old,new):
 read=lambda p:json.loads(p.read_text())
 reasons=[]
 q=read(old/'QUEUE.json')
 if any(j['status']!='COMPLETE' for j in q):reasons.append('E3_E4_QUEUE_NOT_RELEASED')
 if not (old/'E4_ACTIVATION.json').exists():reasons.append('E4_RESERVATION_NOT_YET_CONSUMED')
 d=read(old/'RESOURCE_LEDGER.json')
 if any(not x.get('ended_epoch') for x in d['gpu_sessions']):reasons.append('OLD_GPU_LEASE_ACTIVE')
 if not (new/'NEW_JUDGE_AUTHORIZATION.json').exists():reasons.append('NO_NEW_STAGE_JUDGE_AUTHORIZATION')
 if not (new/'DATA_READY.json').exists():reasons.append('DATA_AND_COMPLETE_PANEL_NOT_READY')
 if (old/'STOP').exists() or (new/'STOP').exists():reasons.append('USER_STOP')
 return dict(status='READY_WAITING_FOR_LEASE' if reasons else 'READY_FOR_SEPARATE_ATOMIC_LEASE_ACQUISITION',blockers=reasons,gpu_started=False,old_budget_never_transferred=True)

def authorize_input(row,purpose,forbidden):
 if purpose not in ['fit','calibration','check','regression','holdout']:raise ValueError('Unknown purpose')
 if purpose in ['fit','calibration','check'] and row['input_hash'] in forbidden:raise ValueError('Formal input cannot enter development support')
 if purpose!=row['purpose']:raise ValueError('Purpose mismatch')
 if row.get('scope')=='unknown' and purpose in ['fit','calibration']:raise ValueError('Unknown scope excluded')
 return True

if __name__=='__main__':
 row=dict(input_hash='x',purpose='fit',scope='negative');assert authorize_input(row,'fit',set())
 try:authorize_input(row,'fit',{'x'});raise AssertionError()
 except ValueError:pass
 print('PASS: purpose isolation and formal-input exclusion')
