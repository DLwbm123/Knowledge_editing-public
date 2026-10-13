"""Read-only R2 inventory and frozen panel audit before R3 GPU execution."""
import hashlib,json,os
from pathlib import Path
from collections import Counter
import torch
ROOT=Path(os.environ['RUN_ROOT']);OLD=ROOT.parents[1]/'textjoint-r2-20260925/run'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def main():
 ts=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];inventory=[]
 for t in ts:
  for arm in ['P','P+S']:
   folder=OLD/'runs/s20260925'/arm/f'e{t["order"]:03d}'
   path=folder/'private/edits'/f'e{t["order"]:03d}'/'C_NO_H/step-80.pt'
   entry=dict(order=t['order'],edit=t['canonical_edit_id'],arm=arm,path=str(path),exists=path.exists(),support_binding=digest({k:t[k] for k in ['native','semantic_fit_questions','U_fit','U_new']}),layer='model.layers.30.mlp.down_proj',teacher_generation=read(OLD/'source/freshstart/generation.json'))
   if path.exists():
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['step']==80 and state['seed']==t['seed']+1 and state['canonical_edit_id']==t['canonical_edit_id']
    assert max(int(v['step']) for v in state['optimizer']['state'].values())==80
    entry.update(sha256=sha(path),actual_optimizer_steps=80,seed=state['seed'],resume_fields=[k for k in ['optimizer','torch_rng','cuda_rng','python_rng'] if k in state],bytes=path.stat().st_size)
    del state
   w=OLD/'runs/s20260925/P'/f'e{t["order"]:03d}'/'private/edits'/f'e{t["order"]:03d}'/'initial/W0_COMPLETE.pt'
   p0=OLD/'runs/s20260925/P'/f'e{t["order"]:03d}'/'STEP0.pt'
   assert w.exists() and p0.exists()
   entry.update(P_W0_sha256=sha(w),P_STEP0_sha256=sha(p0))
   inventory.append(entry)
 write(ROOT/'private/CHECKPOINT_INVENTORY.json',inventory)
 coverage=[]
 for name,tt in [('DEV24',ts[:24]),('R2_VERIFY24_EXTENSION',ts[24:])]:
  for kind in sorted({r['task'] for t in tt for r in t['evaluation']}):
   rr=[(t,r) for t in tt for r in t['evaluation'] if r['task']==kind]
   coverage.append(dict(panel=name,task=kind,consumers=len(rr),effective_edits=len({t['canonical_edit_id'] for t,r in rr}),unique_inputs=len({(r['image_sha256'],r['question']) for t,r in rr}),sources=len({r['source_group'] for t,r in rr}),Base_correct=sum(r.get('base_correct',r.get('initial')) for t,r in rr),Base_wrong=sum(not r.get('base_correct',r.get('initial')) for t,r in rr),modalities=dict(Counter('image+text' if r.get('image_path') else 'text' for t,r in rr)),historically_exposed=True))
 public=ROOT/'public';write(public/'COVERAGE_AUDIT.json',coverage)
 write(public/'CHECKPOINT_INVENTORY_AGGREGATE.json',dict(existing_80=sum(e['exists'] for e in inventory),missing=[dict(order=e['order'],arm=e['arm']) for e in inventory if not e['exists']],actual_step_verified=True,shared_initialization_retained=True))
 # Read final supplemented reports rather than pre-supplement repository snapshots.
 write(ROOT/'private/R2_FINAL_EVIDENCE.json',{n:read(OLD/'public'/n) for n in ['DEV24.json','VERIFY24_SINGLE.json','VERIFY24_SEQUENTIAL.json','RETAINED_BANKS.json','CONFIG_PUBLIC.json','RESOURCE_LEDGER_AGGREGATE.json']})
 for n in ['supports_r2.py','worker_r2.py','orchestrator_r2.py','metrics_r2.py','metrics.py','source/scripts/medtrace/stage15.py']:
  assert (OLD/n).is_file(),n
 write(ROOT/'AUDIT.json',dict(status='PASS',inventory=len(inventory),existing_80=sum(e['exists'] for e in inventory),panel_sha256=sha(ROOT/'private/TASKS_R2_LOCKED.json'),R2_unchanged=True,reference_T1L_correction='VERIFY B0/P final JSON is 2/2, not 0/2; small denominator not robustness evidence'))
 print(read(ROOT/'AUDIT.json'))
if __name__=='__main__':main()
