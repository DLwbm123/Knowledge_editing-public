"""CPU-only bit-exact continuation through the real serialized learner body."""
import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
payload=json.loads(sys.stdin.readline());os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
P=Path(payload['run_root'])
with tempfile.TemporaryDirectory(prefix='e.',dir='/data/bmw/tmp') as folder:
 R=Path(folder);tools=R/'private/tools';tools.mkdir(parents=True);(R/'public').mkdir()
 for p in (P/'private/tools').glob('*.py'):shutil.copy2(p,tools/p.name)
 (tools/'source_stage.py').write_text(payload['source_stage'])
 os.environ.update(RUN_ROOT=str(R),ACTION='cpu_serial_test',ARM='F',SOURCE_OWNER='2',DATA_ROOT=folder);sys.path.insert(0,str(tools))
 import torch,numpy as np
 torch.set_num_threads(1)
 import common as c,learner,checkpoint,source_stage
 from geometry import Risk
 c.write(R/'private/FROZEN_PROTOCOL.json',dict(budgets=dict(maximum_committed_updates=1000,maximum_backwards=3000,maximum_generations=1000,maximum_forward_calls=10000)))
 c.write(R/'RUN_MANIFEST.json',dict(starting_epoch=0,deadline_epoch=9999999999,GPU_seconds_limit=99999,min_free_bytes=1,owned_weight_limit_bytes=99999999))
 c.write(R/'STAGE_CAP.json',dict(GPU_seconds_limit=99999));c.write(R/'RESOURCE_LEDGER.json',dict(gpu_sessions=[],gpu_seconds_used=0,weights_observed_peak_bytes=0));c.write(R/'private/GPU_SOURCE_VERSION.json',dict(commit='CPU_ONLY'))
 class Toy(torch.nn.Module):
  def __init__(self,w):super().__init__();self.weight=torch.nn.Parameter(w.clone(),requires_grad=False)
  def forward(self,inputs_embeds,**kw):return SimpleNamespace(logits=torch.nn.functional.linear(inputs_embeds,self.weight))
 torch.manual_seed(15);initial=torch.randn(7,3).half();x=torch.randn(1,3,3).half();batches={1:(dict(inputs_embeds=x),torch.tensor([[-100,2,4]])),2:(dict(inputs_embeds=x),torch.tensor([[-100,3,5]]))}
 model=Toy(initial);runtime=SimpleNamespace(model=model,device=torch.device('cpu'),get_module=lambda _:model)
 weight=model.weight;master=initial.float();risk=Risk(weight);weight.requires_grad_(True)
 for slot in range(1,161):master,event=learner.optimize(runtime,weight,master,risk,[batches[1]]*5,[],None,slot)
 rows=learner.probe_rows(runtime,weight,[batches[1]]*5,[],20260913);risk.consolidate(rows)
 event.update(owner=1,slot=160,completed_slots=160,arm='F')
 learner.old.save(master,weight,risk,1,160,160,event,initial.float(),boundary=True)
 saved=checkpoint.read(R/'private/active/F/state.pt');c.write(R/'private/units/F/1.json',dict(completed_slots=160))
 for slot in range(1,161):master,event=learner.optimize(runtime,weight,master,risk,[batches[2]]*5,[],None,slot)
 risk.consolidate(learner.probe_rows(runtime,weight,[batches[2]]*5,[],20260914))
 expected=dict(master=master.clone(),native=weight.detach().clone(),risk=risk.state())
 fresh=Toy(initial);test_runtime=SimpleNamespace(model=fresh,device=torch.device('cpu'),get_module=lambda _:fresh)
 c.write(R/'private/window/F/current.json',dict(task=dict(order=2,seed=20260914,fit_questions=['a','b','c','d'],native={}),gate=False))
 source=source_stage.build_source(Path(learner.old.__file__).read_text());namespace={'__name__':'cpu_serialized'};exec(compile(source,'cpu_serialized','exec'),namespace)
 namespace.update(Risk=learner.old.Risk,probe_rows=learner.old.probe_rows,optimize=learner.old.optimize,validate_state=learner.old.validate_state,
     load_learner=lambda:test_runtime,track=lambda _:SimpleNamespace(close=lambda:None),lease=lambda:__import__('contextlib').nullcontext(),
     peak_memory=lambda:None,install=lambda *args:None,batches=lambda *_:([batches[2]]*5,[]))
 # The existing real read-boundary selfcheck is separate; here only the execution/restore body is tested.
 oldread=Path.read_text
 def read_text(path,*a,**k):
  if path in (R/'private/EVAL_LEDGER.json',R/'private/operator/source.json'):raise PermissionError('Fixture denied historical access')
  return oldread(path,*a,**k)
 Path.read_text=read_text
 learner.old.b.guard=lambda *args:None;learner.old.b.schema=lambda model:[('weight',list(model.weight.shape),str(model.weight.dtype))]
 namespace['main']()
 result=checkpoint.read(R/'private/active/F/state.pt')
 assert result['owner']==2 and result['slot']==160 and result['completed_slots']==320 and result['boundary']
 assert torch.equal(result['master'],expected['master']) and torch.equal(result['native'],expected['native'])
 assert checkpoint.exact(result['risk'],expected['risk'])
 assert (R/'private/control/F/EVALUATE.json').exists() and not (R/'private/units/F/2.json').exists()
 print(json.dumps(dict(status='PASS',actual_serialized_body=True,bit_exact_master_native_risk=True,boundary_clock_exact=True,evaluation_handoff_before_exit=True,GPU_updates=0,Judge_requests=0)))
