"""Freeze one positive support guard before new formal output."""
import io,shutil,time,os,sys
from pathlib import Path
import torch
from resources import ROOT,read,write
from storage import Store
sys.path[:0]=[str(ROOT/'source_patch'),str(ROOT/'source')]
def main():
 old=Path(os.environ['PR9_ROOT'])
 assert read(old/'RUN_STATUS.json')['status']=='CLOSED_NOT_ADMITTED'
 if not (ROOT/'private/ROUTER_FEATURES.pt').exists():
  f=torch.load(old/'private/ROUTER_FEATURES.pt',map_location='cpu',weights_only=False)
  def attach(e):
   z=e['positive'].float();assert z.shape[0]==5
   d=torch.cdist(z,z,p=2);d.fill_diagonal_(float('inf'));r=d.min(dim=1).values
   assert torch.isfinite(r).all() and (r>=0).all()
   e['guard_radii']=r
  for e in f['entries']:attach(e)
  for bank in f['banks'].values():
   for e in bank:attach(e)
  for e in f['singles'].values():attach(e)
  audit=dict(status='FROZEN_PRE_OUTPUT',rule='per-anchor closest OTHER native/S_fit distance, inclusive union; NEG0 tau0 OR guard, only inside original R0 closest-winner radius',anchors_per_edit=5,edits=48,formal_inputs_outcomes_in_construction=0,U_bg_used_to_fit_guard=0,cross_image_annotations_added=0,zero_distance_radii_retained=True,epoch=time.time())
  write(ROOT/'public/POSITIVE_GUARD_FREEZE.json',audit)
  b=io.BytesIO();torch.save(f,b);Store(ROOT).write('private/ROUTER_FEATURES.pt',b.getvalue())
  shutil.copy2(old/'private/H_WEIGHT_BINDINGS.json',ROOT/'private/H_WEIGHT_BINDINGS.json')
  for name in ['INSERTION_HAZARD_AUDIT.json','INSERTION_HAZARD_AUDIT_ZH.md','CAL_SCOPE_DATA_AUDIT.json']:shutil.copy2(old/'public'/name,ROOT/'public'/name)
 else:
  assert read(ROOT/'public/POSITIVE_GUARD_FREEZE.json')['status']=='FROZEN_PRE_OUTPUT'
  f=torch.load(ROOT/'private/ROUTER_FEATURES.pt',map_location='cpu',weights_only=False)
 from routers import ScopeRouter
 eligible=outside=0
 for e in f['entries']:
  a=ScopeRouter([e],'R0');b=ScopeRouter([e],'PLOO')
  for z in e['positive']:
   if a.route(z).logical_edit_id==e['edit']:assert b.route(z).logical_edit_id==e['edit'];eligible+=1
   else:assert not b.route(z).activated;outside+=1
 write(ROOT/'public/ROUTER_MECHANICAL_TESTS.json',dict(status='GUARD_FROZEN_MODEL_PARITY_PENDING',guard_membership_preserves_all_R0_eligible_legal_supports=True,eligible_supports=eligible,outside_original_R0_supports=outside,prototype_origins_native_S_fit_U_bg_only=True,no_formal_guard_fit=True,training_steps=0))
 write(ROOT/'RUN_STATUS.json',dict(status='READY_FOR_MODEL_MECHANICAL',phase='P3',epoch=time.time()))
 print('GUARD_FROZEN',eligible,outside)
if __name__=='__main__':main()
