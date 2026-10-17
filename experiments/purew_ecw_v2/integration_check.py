"""Disposable CPU production-import/metadata check; no model or CUDA calls."""
import json,os,sys,tempfile,shutil
from pathlib import Path
x=json.loads(sys.stdin.readline());r=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'));tools=r/'private/tools';tools.mkdir(parents=True)
p=Path('/data/bmw/Knowledge_editing/outputs/purew-fs3-v1-20261014/run/private')
for path in (p/'tools').glob('*.py'):shutil.copy2(path,tools/path.name)
for name,text in x['files'].items():(tools/name).write_text(text)
(r/'private/FROZEN_PROTOCOL.json').write_text(json.dumps(x['protocol']))
os.environ.update(RUN_ROOT=str(r),ACTION='CPU_INTEGRATION',ARM='X',TMPDIR=str(r))
sys.path.insert(0,str(p/'source'));sys.path.insert(0,str(tools))
import controller,queue_revision,report_revision,worker
from panels import expected_ids
from core import Memory
import torch
q=json.loads((p/'BENCHMARK146_QUEUE.json').read_text())['tasks'];l=json.loads((p/'EVAL_LEDGER.json').read_text())
counts=[len(expected_ids(q,l,'bank_R0',0,z)) for z in [1,3]]+[sum(len(expected_ids(q,l,'insertion_R0',z,z)) for z in [1,2,3]),len(expected_ids(q,l,'protect100_R0',0,100))]
assert counts==[46,64,15,36]
w=torch.randn(40,36).half();risk=Memory(w,True);risk.consolidate(torch.randn(4,40,36,dtype=torch.float64)*.01);risk.prepare();g=torch.randn(40,36);g*=5/g.norm();d,event=risk.solve(g)
actual=(w.float()+d.reshape_as(w).float()).half().double()-w.double();assert risk.inspect(actual)['native_risk_bound']>=0
resumed=Memory.restore(risk.state(),'cpu');dd,_=resumed.solve(g);assert torch.equal(dd,d)
assert x['protocol']['budgets']['maximum_generations']==2*(3+sum(counts)+2)
print(json.dumps(dict(status='PASS',production_imports=True,scoring_metadata_counts=counts,restricted_native_complete_risk=True,restricted_restore_next_exact=True,CUDA_calls=0,model_calls=0)))
