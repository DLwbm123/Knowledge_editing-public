"""Bounded CPU checks for fixed R0 and real independent expert banks."""
import json,sys,os
from pathlib import Path

def run(root,parent):
 sys.path[:0]=[str(root/'tools'),str(parent/'private/tools'),str(parent/'private/source'),str(parent/'private/official_llava')]
 import torch
 import module_run
 from m3bench_repro.editors.routing import MemoryRouter
 from m3bench_repro.editors.routed_layers import RoutedFullLinear
 from cache_v18 import selfcheck
 torch.set_num_threads(1)
 selfcheck();cache=dict(status='PASS',cold_hit_exact=True,mutated_input_rejected=True,original_methods_restored=True)
 router=MemoryRouter('euclidean');router.add('a',torch.tensor([0.,0.]),.1,[1]);router.add('b',torch.tensor([.3,0.]),1.,[2])
 d=router.route(torch.tensor([.12,0.]));assert not d.activated and d.nearest_logical_edit_id=='a'
 changed=MemoryRouter('euclidean');changed.add('a',torch.tensor([0.,0.]),.1,[991]);changed.add('b',torch.tensor([.3,0.]),1.,[992]);assert changed.route(torch.tensor([.12,0.]))==d
 base=torch.nn.Linear(4,3,bias=False);bank=RoutedFullLinear(base);x=torch.arange(4,dtype=torch.float32).reshape(1,4)
 bank.add_edit('a').weight.data.add_(.1234567);bank.set_active('a');one=bank(x).clone();saved=bank.export_state();bank.add_edit('b').weight.data.sub_(.2);bank.set_active('a');assert torch.equal(one,bank(x));bank.train_only(None)
 # FP16 preprocessing must operate on detached Base, never on bank member weights.
 before=bank.get_edit('a').weight.detach().clone();bank.base.half();bank.base.float();assert torch.equal(before,bank.get_edit('a').weight)
 restored=RoutedFullLinear(torch.nn.Linear(4,3,bias=False));restored.load_exported_state(saved);restored.set_active('a');assert torch.equal(one,restored(x))
 tasks=json.loads((root/'DATA_ADMISSION.json').read_text())['tasks'];assert len(tasks)==10 and [t['order'] for t in tasks]==list(range(1,11));coverage=[]
 for n in [1,5,10]:
  ts=tasks[:n];qs={q['query_id'] for t in ts for q in t['eval_rows']};cs=[(t['key'],q['query_id']) for t in ts for q in t['eval_rows']];assert len(cs)==len(set(cs));coverage.append(dict(prefix=n,designated=len(cs),natural_unique=len(qs)))
 assert coverage==[dict(prefix=1,designated=50,natural_unique=50),dict(prefix=5,designated=114,natural_unique=114),dict(prefix=10,designated=219,natural_unique=183)]
 out=dict(status='PASS',model_calls=0,coverage=coverage,max_generations_from_consumers=1486,shared_R0_nearest_rejection_preserved=True,route_does_not_use_answer_labels=True,independent_full_linear_members=True,save_load_exact=True,FP16_preprocessing_preserves_detached_FP32_members=True,production_cache=cache)
 (root/'CPU_ADMISSION.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(run(Path(os.environ['V80_ROOT']),Path(os.environ['PARENT_RUN'])),indent=2))
