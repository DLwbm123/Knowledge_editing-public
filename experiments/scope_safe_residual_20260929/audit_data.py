"""Outcome-blind data and initialization inventory; no implicit scope labels."""
import json,hashlib,collections,os
from pathlib import Path
from p0 import read,write,digest

def norm(s):return ' '.join(s.casefold().split()).rstrip('?.。？')
def ident(x):return (x.get('image_sha256'),norm(x['question']))
def main(old,out,source):
 tasks=read(old/'private/TASKS_R2_LOCKED.json')['tasks'];aux=read(old/'private/AUXILIARY_POOL.json')
 formal=[r for t in tasks for r in t['official_evaluation_full']]
 for n in ['PRESSURE_VALIDATION_FROZEN.json','PRESSURE_TEST_FROZEN.json']:formal+=read(old/'private'/n)['rows']
 forbidden={ident(x) for x in formal};forbidden_groups={x['source_group'] for x in formal};edited_groups={t['native']['source_group'] for t in tasks}
 policy=dict(forbidden_for_fit_calibration_and_mining=['formal T0/T1G/T2G/T1L/T2L','old pressure validation/test','their per-item scores'],allowed_fit=['native','S_fit','inherited U_old/U_new','audited U_bg'],calibration='Disjoint source groups and inputs from fit; no formal outcomes',check='Independent CHECK only for lambda choice',regression='Exposed DEV/REG evaluation only',holdout='New source groups only; never rename DEV/REG CONFIRM',enforce='Purpose enum and input/source exclusion, explicit scope evidence required',future_edit_support_prohibited=True)
 write(out/'public/DATA_ACCESS_POLICY.json',policy)
 known=[x for x in aux if ident(x) not in forbidden and x['source_group'] not in forbidden_groups|edited_groups and x.get('source_role')=='train' and x['source_annotation'].get('content_type') in ['Modality','Plane']]
 unknown=[x for x in aux if x not in known]
 roles=[]
 for x in known:roles.append(dict(input_hash=digest(ident(x)),source=x['source_group'],role='U_bg',scope='negative',evidence='train-only acquisition modality/plane on source disjoint from every edited fact and formal panels',data=x))
 for x in unknown:roles.append(dict(input_hash=digest(ident(x)),source=x['source_group'],role='EXCLUDED',scope='unknown',evidence='No certified fact-scope separation or overlaps existing protected sources'))
 write(out/'private/SCOPE_ROLE_MANIFEST.json',dict(rows=roles,formal_forbidden_hashes=sorted(digest(x) for x in forbidden)))
 write(out/'public/SCOPE_ROLE_MANIFEST.json',dict(candidates=len(aux),confirmed_background=len(known),scope_unknown_excluded=len(unknown),private_manifest=True))
 # Stable source/type stratification; no scores or model outputs enter selection.
 strata=collections.defaultdict(list)
 for t in tasks[:24]:strata[t['native'].get('dataset','unknown')].append(t)
 for k in strata:strata[k].sort(key=lambda t:digest([t['native']['source_group'],t['canonical_edit_id'],'PILOT_DEV8']))
 chosen=[];seen=set()
 while len(chosen)<8:
  changed=False
  for k in sorted(strata):
   candidates=[x for x in strata[k] if x['canonical_edit_id'] not in {t['canonical_edit_id'] for t in chosen} and x['native']['source_group'] not in seen]
   if candidates and len(chosen)<8:chosen.append(candidates[0]);seen.add(candidates[0]['native']['source_group']);changed=True
  if not changed:break
 chosen.sort(key=lambda t:t['order']);assert len(chosen)==8
 write(out/'private/PILOT_SELECTION.json',dict(orders=[t['order'] for t in chosen],edit_ids=[t['canonical_edit_id'] for t in chosen],rule='round-robin dataset strata, stable source/edit hash, unique source; no outcomes',seed=20260929))
 n=sum(len(t['evaluation']) for t in chosen);n4=sum(len(t['evaluation']) for t in chosen[:4]);minimum=6*(2*n+n4)
 queue=[dict(id=f'P1-{t["order"]}-{a}',order=t['order'],branch=a,extra_steps=80,status='BLOCKED_DATA_AND_NEW_JUDGE_RESERVATION',init='INIT_POST80',requires=['DATA_READY','NEW_JUDGE_RESERVATION','E3_E4_RELEASED']) for t in chosen for a in ['A0','AK','AU','AH','AHS_001','AHS_01']]
 write(out/'QUEUE.json',queue)
 write(out/'public/JUDGE_BUDGET_REQUIREMENT.json',dict(new_stage_authorized_attempt_items=0,old_ceiling_not_transferable=9539,pilot_branches=6,formal_single_items=6*n,formal_bank4_8_items=6*(n4+n),formal_upper_bound_without_dedup=minimum,additional_CHECK_CAL_and_forced_items=None,status='NOT_ADMISSIBLE_UNTIL_COMPLETE_PANEL_COUNTS_FROZEN',note='Actual account Codex quota not inferred from internal ceiling'))
 import torch
 init=[]
 for t in tasks[:24]:
  p=old/f'adapters/s20260927/M1/e{t["order"]:03d}.pt';x=torch.load(p,map_location='cpu',weights_only=False);b=x['binding'];assert x['step']==b['actual_step']==80 and b['method']=='M1' and b['seed']==20260927 and b['structure']=='CP'
  assert b['edit']==t['canonical_edit_id'] and x['kind']=='LR';assert set(x['expert'])=={'A','B'}
  init.append(dict(order=t['order'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),origin='INIT_POST80',binding=b,shape={k:list(v.shape) for k,v in x['expert'].items()}))
 write(out/'private/INIT_BINDINGS.json',init);write(out/'public/INIT_BINDINGS.json',dict(verified=24,origin='E1/E2 DEV M1 post-continuation80',new_steps=80,common_start_per_edit=True,import_mode='Read-only references; no historical weight mutation',old_W0_reuse=False))
 modules=['experiments/decomp_24h_20260927/structures.py','experiments/decomp_appendix_20260929/training.py','experiments/decomp_24h_20260927/evaluation.py','experiments/decomp_appendix_20260929/report.py','experiments/textjoint_r3_20260926/router_r3.py','experiments/textjoint_r3_20260926/calibration_data.py','experiments/textjoint_r4_20260926/README.md']
 write(out/'public/INHERITANCE_AUDIT.json',dict(modules=[dict(relative_path=n,sha256=hashlib.sha256((source/n).read_bytes()).hexdigest()) for n in modules],R1='Existing kappa/mu rejection control, not a novel method',R4='R0 frozen due to deficient cross-image CAL_PLUS; not evidence R1 ineffective',U_KL='Already forced-on teacher-forced; new work changes negative coverage/sampling/residual/gate'))
 audit=dict(auxiliary_candidates=len(aux),confirmed_U_bg=len(known),U_bg_sources=len({x['source_group'] for x in known}),scope_unknown=len(unknown),formal_input_collisions=sum(ident(x) in forbidden for x in aux),certified_CAL_cross_image_positives=0,certified_CAL_same_image_paraphrases=0,Rneg_status='UNSUPPORTED_CAL_SCOPE',new_holdout_certified=0,new_holdout_target_inputs=200,new_holdout_target_sources=50,independent_CONFIRM=False,pilot_edits=8,pilot_sources=len(seen),cached_feature_calibration='Blocked: insufficient certified CAL positives; no threshold fit to formal data',expert_line='Data preparation incomplete; do not claim GPU training ready')
 write(out/'public/DATA_AUDIT.json',audit)
 write(out/'public/SCIENCE_LOCK.json',dict(status='CPU_PREPARATION_NOT_GPU_ADMITTED',main='CP -> free rank4 INIT_POST80',layer='model.layers.30.mlp.down_proj',precision='float16 backbone, FP32 master',branches=['A0','AK','AU','AH','AHS'],steps=80,lambda_res_pilot=[.01,.1],tau_grid=[-.2,0,.1,.2,.3],seed=20260929,no_E3_E4_outcome_selection=True,fit_input_role_policy=policy,remaining_unfrozen=['new U_bg pool audit','CHECK/CAL data','complete scoring reservation']))
 write(out/'RUN_STATUS.json',dict(status='BLOCKED_DATA_AND_JUDGE_RESERVATION',gpu_status='READY_WAITING_FOR_LEASE_NOT_ADMITTED',gpu_started=False,blockers=['E3/E4 own all allowed GPU leases','no new-stage Judge allocation','independent CHECK/CAL incomplete','U_bg restricted'],no_polling_or_training=True))
if __name__=='__main__':
 import sys
 main(*(Path(x) for x in sys.argv[1:]))
