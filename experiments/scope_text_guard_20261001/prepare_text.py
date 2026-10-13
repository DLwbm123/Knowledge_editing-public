"""Freeze legal text supports before new formal routing or outcomes."""
import os,sys,io,time,shutil
from pathlib import Path
import torch
from resources import ROOT,read,write
from storage import Store
sys.path[:0]=[str(ROOT/'source_patch'),str(ROOT/'source')]
from routers import canon,ScopeRouter,selfcheck
def main():
 selfcheck();old=Path(os.environ['PR9_ROOT']);assert read(old/'RUN_STATUS.json')['status']=='CLOSED_NOT_ADMITTED'
 assert not (ROOT/'private/ROUTER_FEATURES.pt').exists(),'Definitions are immutable'
 f=torch.load(old/'private/ROUTER_FEATURES.pt',map_location='cpu',weights_only=False);tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];byedit={t['canonical_edit_id']:t for t in tasks};bg={canon(r['question']) for r in read(ROOT/'private/U_bg.json')['rows']}
 def attach(e):
  t=byedit[e['edit']];allq={canon(q) for q in [t['native']['question']]+t['semantic_fit_questions']};e['text_guard_questions']=sorted(allq-bg);e.pop('guard_radii',None);assert not set(e['text_guard_questions'])&bg
 for e in f['entries']:attach(e)
 for b in f['banks'].values():
  for e in b:attach(e)
 for e in f['singles'].values():attach(e)
 records=[dict(edit_order=e['order'],positive_question_origins=5,unique_positive_questions=len({canon(q) for q in [byedit[e['edit']]['native']['question']]+byedit[e['edit']]['semantic_fit_questions']}),unconflicted_text_supports=len(e['text_guard_questions'])) for e in f['entries']]
 write(ROOT/'public/TEXT_GUARD_FREEZE.json',dict(status='FROZEN_PRE_OUTPUT',normalization='Unicode NFKC, casefold, whitespace collapse; no punctuation removal or fuzzy matching',U_bg_unique_questions=len(bg),edit_support_counts=records,formal_inputs_answers_or_correctness_in_construction=0,all_U_bg_question_collisions_excluded=True,new_verified_cross_image_annotations=0,epoch=time.time()))
 b=io.BytesIO();torch.save(f,b);Store(ROOT).write('private/ROUTER_FEATURES.pt',b.getvalue());shutil.copy2(old/'private/H_WEIGHT_BINDINGS.json',ROOT/'private/H_WEIGHT_BINDINGS.json')
 for n in ['INSERTION_HAZARD_AUDIT.json','INSERTION_HAZARD_AUDIT_ZH.md','CAL_SCOPE_DATA_AUDIT.json']:shutil.copy2(old/'public'/n,ROOT/'public'/n)
 eligible=0
 for e in f['entries']:
  t=byedit[e['edit']];questions=[t['native']['question']]+t['semantic_fit_questions'];a=ScopeRouter([e],'R0');b=ScopeRouter([e],'TXT')
  for z,q in zip(e['positive'],questions,strict=True):
   if a.route(z).logical_edit_id==e['edit']:assert b.diagnostic(z,question=q)[0].logical_edit_id==e['edit'];eligible+=1
 write(ROOT/'public/ROUTER_MECHANICAL_TESTS.json',dict(status='TEXT_FROZEN_MODEL_PARITY_PENDING',legal_supports_preserved=eligible,text_origins_native_plus_four_S_fit_only=True,all_U_bg_exact_question_collisions_excluded=True,no_formal_guard_fit=True,training_steps=0))
 write(ROOT/'RUN_STATUS.json',dict(status='READY_FOR_MODEL_MECHANICAL',phase='P3',epoch=time.time()));print('TEXT_SUPPORTS_FROZEN',sum(r['unconflicted_text_supports'] for r in records),eligible)
if __name__=='__main__':main()
