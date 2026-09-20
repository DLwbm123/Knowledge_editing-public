"""Single frozen protocol, global inherited ledger, exact reuse, no semantic retry."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs
from reports.medtrace_stage22_20260919.scoring import judge
p=Path(__file__).parent/'private/run';common=ROOT/'reports/medtrace_stage22_20260919/private/run'
base=read(p/'private/BASE.json')['records'];assert len(base)==52
judge(common,base,'24D_BASE52',category='24D')
if os.environ.get('BASE_ONLY')!='1':
 rows=outputs(p)
 for n,expected in ((19,210),(45,298)):
  if os.environ.get('PREFIX_ONLY') and int(os.environ['PREFIX_ONLY'])!=n:continue
  rr=[r for r in rows if r['N']==n];assert len(rr)==expected
  judge(common,rr,f'24D_DIAGNOSTIC_{n}',category='24D')
 if not os.environ.get('PREFIX_ONLY'):
  assert read(p/'public/EXIT_24D.json')['exit_code']==0 and len(rows)==508
  write(p/'public/SCORING_STATUS.json',dict(status='COMPLETE',base=52,diagnostic_outputs=508,formal_method_result=False))
print('SCORING_COMPLETED',flush=True)
