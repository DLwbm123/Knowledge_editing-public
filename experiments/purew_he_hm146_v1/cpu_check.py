"""Run production tensor paths in a disposable CPU-only sandbox."""
import contextlib,copy,json,os,shutil,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
payload=json.loads(sys.stdin.readline());started=time.monotonic()
root=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'));tools=root/'private/tools';tools.mkdir(parents=True)
parent=Path('/data/bmw/Knowledge_editing/outputs/purew-objective-scope-v1-20261015/run/private')
for p in (parent/'tools').glob('*.py'):shutil.copy2(p,tools/p.name)
for n,s in payload['files'].items():(tools/n).write_text(s)
os.environ.update(RUN_ROOT=str(root),ARM='HE',ACTION='CPU_PREPARATION',CUDA_VISIBLE_DEVICES='',DATA_ROOT=str(root),SCOPE_DIAGNOSTICS='0')
sys.path[:0]=[str(tools),str(parent/'source')]
(root/'private/FROZEN_PROTOCOL.json').write_text(json.dumps(payload['protocol']))
import torch,numpy as np
from configuration import *
import controller,worker,source_loss,medical,queue_revision,report_revision
import astra_report as report_engine
from core import Memory
torch.set_num_threads(2)
def forbidden(*a,**k):raise AssertionError('No CUDA/model/Judge launch permitted in CPU check')
torch.cuda.init=torch.cuda._lazy_init=forbidden;torch.cuda.is_available=lambda:False
validate_protocol(payload['protocol'])
try:require_execution(payload['protocol'],payload['authorization'],payload['hardware'])
except AssertionError:pass
else:raise AssertionError('Preparation authorization admitted execution')
trace=[];controller.run_plan(trace.append)
assert len([x for x in trace if x['action']=='learn'])==292
assert [x['owner'] for x in trace if x['arm']=='HE' and x['action']=='learn']==list(range(1,147))
assert [x['action'] for x in trace[:4]]==['base','admit','learn','insertion']
for arm in ARMS:
 assert len([x for x in trace if x['arm']==arm and x['action']=='bank'])==4
assert source_gate(.1,[1.]*10,True)['failed_scientific_gate']
assert source_gate(.01,[1.]*9+[0.],True)['continue_queue']
assert source_gate(.01,[],False)['evidence_pass'] is None
for k,cap in PER_ARM.items():
 check_counts({k:cap},{k:2*cap})
 try:check_counts({k:cap+1},{k:cap+1})
 except AssertionError:pass
 else:raise AssertionError('Own cap borrowed from second arm')
assert list(report_engine.r.score_bounds(report_engine.r.combine([{'shared':1},{'shared':-1}]),{}))==[0,0]
assert list(report_engine.r.score_bounds({'a':1,'b':-1},{'a':None,'b':1}))==[-1,0]
# Exact objective difference and both-side gradients, using the production loss selector.
y=torch.tensor([0,1]);z=torch.tensor([[.7,.2,-.1],[.1,.6,-.2]],dtype=torch.float64,requires_grad=True)
zr=(z.detach()*.9).requires_grad_();prefix=torch.zeros((1,3),dtype=torch.float64)
anchor=source_loss.teacher(z.detach()+.4,y,prefix)
m=SimpleNamespace(_projection_cache=[anchor]*5,_evidence_indices=[[[0],[1]]]*5)
with patch.object(source_loss,'position_logits',return_value=(z,y,prefix)),patch.object(worker.b,'logits',return_value=(zr,y)):
 source_loss.ARM='HE';he=source_loss.loss(m,[0]*5,[1]*5,[None]*5,1)[0]
 source_loss.ARM='HM';hm=source_loss.loss(m,[0]*5,[1]*5,[None]*5,1)[0]
 e,_=source_loss.evidence_components(z,y,zr,y,[[0],[1]])
 assert torch.allclose(he-hm,e,atol=1e-12,rtol=0)
 g=torch.autograd.grad(he-hm,(z,zr),retain_graph=True);oracle=torch.autograd.grad(e,(z,zr))
 assert all(torch.allclose(a,b,atol=1e-12,rtol=0) for a,b in zip(g,oracle))
 for arm in ARMS:
  source_loss.ARM=arm
  value=source_loss.loss(m,[0]*5,[],[],1)[0]
  assert torch.equal(value,source_loss.components(z,y,prefix,anchor)['odds_hinge'])
# More than old46 rows with exact finite-support covariance and state restore.
mem=Memory(torch.zeros((2,3),dtype=torch.float16),'J');a=torch.randn((50,6),dtype=torch.float64)
mem.consolidate(a);assert len(mem.blocks['history'].B)==50
assert torch.allclose(mem.blocks['history'].B.T@mem.blocks['history'].B,a.T@a,atol=1e-10,rtol=0)
assert len(Memory.restore(mem.state()).blocks['history'].B)==50
# Actual learner traverses all146 owners per arm and restores native/master/statistics.
c,b=worker.c,worker.b
c.rng=lambda:dict(torch_rng=torch.get_rng_state());c.restore_rng=lambda x:torch.set_rng_state(x['torch_rng'])
original_read=Path.read_text

def guarded_read(p,*a,**kw):
 if p in (root/'private/EVAL_LEDGER.json',root/'private/operator/source.json'):raise PermissionError('CPU source boundary')
 return original_read(p,*a,**kw)

c.write(root/'RUN_MANIFEST.json',dict(owned_weight_limit_bytes=1024**3))
c.write(root/'RESOURCE_LEDGER.json',dict(weights_observed_peak_bytes=0))
# Production save creates these only in the disposable sandbox.
summary=[]
for arm in ARMS:
 worker.ARM=source_loss.ARM=arm;worker.WINDOW=root/'private/window'/arm;worker.CONTROL=root/'private/control'/arm
 for kind in ('REAL_VLM_ADMISSION','BASE_IDENTITY'):c.write(root/'public'/(kind+'_'+arm+'.json'),dict(status='PASS'))
 module=torch.nn.Linear(3,2,bias=False,dtype=torch.float16)
 base=torch.tensor([[2.,0.,-1.],[0.,2.,-1.]],dtype=torch.float16)
 with torch.no_grad():module.weight.copy_(base)
 module.weight.requires_grad_(False);module.eval()
 model=SimpleNamespace(model=module,device=torch.device('cpu'),get_module=lambda _:module,adapter=SimpleNamespace(tokenizer=None))
 (root/'private/calibration').mkdir(exist_ok=True)
 torch.save(Memory(module.weight,'J').state(),root/'private/calibration/Base_plain.pt')
 b.COUNTS=dict(forwards=0,backwards=0,committed_updates=0,generation_attempts=0);b.STOP_REQUESTED=False
 def position(_,k):return module.weight.double(),y,prefix
 # Equal full/removed => evidence fails, demonstrating continuation of failed sources.
 def logits(_,k):return module.weight.double(),y
 def prepare(m,items):
  assert torch.equal(module.weight,base),'Teacher was formed after previous edit loaded'
  m._projection_cache=[source_loss.teacher(module.weight.detach().double(),y,prefix)]*5
 for owner in range(1,147):
  os.environ.update(OWNER=str(owner),ACTION='learn_'+arm+'_'+str(owner))
  with torch.no_grad():module.weight.copy_(base)
  packet=dict(quick_owner=owner,gate=owner in ELIGIBLE,task=dict(order=owner,seed=310+owner,native=dict(reference='fixture')))
  c.write(worker.WINDOW/'current.json',packet)
  full=list(range(5));removed=list(range(5,10)) if owner in ELIGIBLE else []
  with contextlib.ExitStack() as stack:
   for obj,name,value in [(worker,'load_learner',lambda:model),(worker,'track',lambda _:contextlib.ExitStack()),(worker,'install',lambda *a:None),(worker,'batches',lambda *a:(full,removed)),(worker,'peak_memory',lambda:{}),(source_loss,'prepare',prepare),(source_loss,'position_logits',position),(b,'logits',logits),(b,'counts',lambda:None),(b,'call',lambda *a:None),(c,'budget',lambda:None),(medical,'source_concepts',lambda _:['a','b']),(medical,'concept_indices',lambda *a:[[0],[1]])]:stack.enter_context(patch.object(obj,name,value))
   stack.enter_context(patch.object(Path,'read_text',guarded_read));stack.enter_context(patch.object(torch.cuda,'manual_seed_all',lambda *a:None))
   worker.learner()
  state=worker.validate_state(torch.load(root/'private/active'/arm/'state.pt',map_location='cpu',weights_only=True))
  assert state['owner']==owner and state['boundary'] and state['slot']==20
  rec=c.read(root/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'))
  if owner in ELIGIBLE:assert rec['gate']['failed_scientific_gate'] and not rec['birth']
 assert len(list((root/'private/steps'/arm).glob('*/*.json')))==2920
 assert b.COUNTS['committed_updates']>0, 'Fixture must exercise native writes'
 assert not torch.equal(state['native'],base), 'Sequential state must carry edits'
 assert len(state['memory']['blocks']['removed']['B'])==4*len(ELIGIBLE), 'Prewrite statistics lost across sources'
 assert (root/'private/final'/arm/'native.pt').exists()
 summary.append(dict(arm=arm,owners=146,slots=2920,failed_sources_continued=9))
assert not torch.cuda.is_initialized()
shutil.rmtree(root)
print(json.dumps(dict(status='PASS',actual_learner_trajectories=summary,CPU_seconds=time.monotonic()-started,
 real_model_calls=0,GPU_calls=0,Judge_calls=0,install_calls=0,production_budget_guards=True,
 original_Base_teacher_before_state=True,shared_loss_oracle=True,no_evidence_HM_HE_equal=True,
 complete_support_over46=True,shared_missing_bounds=True,preparation_launch_denied=True)))
