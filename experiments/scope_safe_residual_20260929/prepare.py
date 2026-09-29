"""CPU-only independent namespace preparation; never edits the predecessor."""
import os,json,shutil,hashlib,time,sys
from pathlib import Path
from p0 import read,write,digest
from audit_data import ident

def main(old,out):
 assert old.resolve()!=out.resolve()
 for name in ['source','source_patch']:
  if not (out/name).exists():shutil.copytree((old/name).resolve(),out/name,ignore=shutil.ignore_patterns('__pycache__','.git'))
 if not (out/'models').exists():(out/'models').symlink_to((old/'models').resolve(),target_is_directory=True)
 for n in ['resources.py','storage.py','worker_v3.py','router_r3.py','structures.py','training.py','evaluation.py','judge_io.py','diagnostics.py']:
  shutil.copy2(old/n,out/n)
 for n in ['CONFIG_LOCK.json','GPU_BINDINGS.json','STORAGE_POLICY.json','RUN_MANIFEST.json','EXPERIMENT_LOCK.json','USER_LIMIT_WAIVER.json']:
  shutil.copy2(old/n,out/n)
 for n in ['TASKS_R2_LOCKED.json','G_SUPPORTS.json','BASE_MASKS.json']:
  shutil.copy2(old/'private'/n,out/'private'/n)
 if (old/'SUPPORT_REPAIR.json').exists():
  a=read(old/'SUPPORT_REPAIR.json');write(out/'SUPPORT_REPAIR.json',a);(out/a['repaired_path']).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/a['repaired_path'],out/a['repaired_path'])
 for name in ['logs','jobs','private/base','private/keys','private/black','private/judge/pending','private/judge/scores']:(out/name).mkdir(parents=True,exist_ok=True)
 # Cache tensors/outputs are copied, not shared writable links.
 for name in ['base','keys']:
  for p in (old/'private'/name).glob('*'):
   if p.is_file() and p.suffix in ['.pt','.json'] and not (out/'private'/name/p.name).exists():shutil.copy2(p,out/'private'/name/p.name)
 tasks=read(out/'private/TASKS_R2_LOCKED.json')['tasks'];formal=[x for t in tasks for x in t['official_evaluation_full']]
 for n in ['PRESSURE_VALIDATION_FROZEN.json','PRESSURE_TEST_FROZEN.json']:formal+=read(old/'private'/n)['rows']
 fit=[t['native'] for t in tasks]+[dict(t['native'],question=q) for t in tasks for q in t['semantic_fit_questions']+t['fit_questions']]+[x for t in tasks for k in ['U_fit','U_new'] for x in t[k]]
 forbidden={ident(x) for x in formal+fit};cal=read(old.parents[1]/'textjoint-r3-20260926/run/private/CAL_ROUTE.json')['positives'];check={}
 for t in tasks[:24]:
  rows=[]
  for x in cal:
   if x['order']==t['order'] and x['role']=='new_paraphrase' and ident(x) not in forbidden:
    assert x['image_sha256']==t['native']['image_sha256']
    rows.append(dict(x,image_path=t['native']['image_path'],reference=t['native']['reference'],task='CHECK_POS',query_id='check-'+digest(ident(x)),purpose='check',scope='positive'))
  check[str(t['order'])]=rows
 assert all(len(v)==2 for v in check.values()),'CHECK support shortfall; do not replace editors'
 write(out/'private/CHECK_POS.json',check)
 # Verify imported tensor identity against the immutable data/protocol and retained provenance.
 import torch
 prior=read(out/'private/INIT_BINDINGS.json');ledger=read(old/'STORAGE_LEDGER.json');lock=read(old/'SCIENCE_LOCK.json');g=read(old/'private/G_SUPPORTS.json')
 if (old/'SUPPORT_REPAIR.json').exists():g=read(old/read(old/'SUPPORT_REPAIR.json')['repaired_path'])
 gs={x['order']:x for x in g};audit=[]
 for t,entry in zip(tasks[:24],prior,strict=True):
  src=Path(entry['path']);x=torch.load(src,map_location='cpu',weights_only=False);b=x['binding'];assert b==entry['binding'] and hashlib.sha256(src.read_bytes()).hexdigest()==entry['sha256']
  support=hashlib.sha256(json.dumps(dict(native=t['native'],P=t['semantic_fit_questions'],Uold=t['U_fit'],Unew=t['U_new'],G=gs[t['order']]),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
  assert b['input_target_support']==support and b['science_id']==lock['id'] and b['backend']==lock['backend']
  assert b['initialization_W0_hash']==ledger['artifacts'][f'checkpoints/W0/s20260927/CP/e{t["order"]:03d}.pt']['hash']
  assert x['step']==80 and b['actual_step']==80 and b['method']=='M1' and x['kind']=='LR'
  dst=out/f'initializers/e{t["order"]:03d}.pt';dst.parent.mkdir(exist_ok=True)
  if not dst.exists():shutil.copy2(src,dst)
  audit.append(dict(order=t['order'],sha256=entry['sha256'],binding=b,relative_path=str(dst.relative_to(out))))
 write(out/'private/INITIALIZERS.json',audit)
 selection=read(out/'private/PILOT_SELECTION.json')['orders'];neg=read(out/'private/CHECK_NEG.json')['rows'];n=sum(len(t['evaluation']) for t in tasks if t['order'] in selection);n4=sum(len(t['evaluation']) for t in tasks if t['order'] in selection[:4])
 budget=7*(2*n+n4+len(selection)*len(neg)+sum(len(check[str(o)]) for o in selection))+len(neg)+n
 write(out/'public/JUDGE_BUDGET_REQUIREMENT.json',dict(status='AUTHORIZED_NO_INTERNAL_CAP_FINITE_PILOT',new_stage_items_upper_bound=budget,arms_including_E_orig=7,formal_and_CHECK_forced=True,unknown_payload_dedup_discount=0,authorization='User: 不用担心额度，我额度够，没有限制，你直接运行即可',account_quota_not_asserted=True))
 write(out/'NEW_JUDGE_AUTHORIZATION.json',dict(epoch=time.time(),user_text='不用担心额度，我额度够，没有限制，你直接运行即可',scope='Attached finite P0-P5 protocol with its data and reproducible-gain conditions; no arbitrary search',internal_attempt_cap_waived=True,finite_pilot_reservation=budget,old_ledger_unchanged=True))
 science=dict(parent_science=lock['id'],backend=lock['backend'],model_assets=lock['model_assets'],recipe=read(out/'public/SCIENCE_LOCK.json'),initializers=[dict(order=x['order'],sha256=x['sha256']) for x in audit],data={n:hashlib.sha256((out/'private'/n).read_bytes()).hexdigest() for n in ['TASKS_R2_LOCKED.json','U_bg.json','CHECK_POS.json','CHECK_NEG.json','PILOT_SELECTION.json']})
 science['id']=digest(science);write(out/'SCIENCE_LOCK.json',science)
 write(out/'DATA_READY.json',dict(status='EXPERT_PILOT_SUPPORTED_RESTRICTED_BACKGROUND',CHECK_positive=48,CHECK_negative=len(neg),CHECK_provenance='Inherited R3 same-image paraphrases, input-disjoint from all fit/formal; previously exposed calibration, not independent confirmation',Rneg='UNSUPPORTED_CAL_SCOPE'))
 write(out/'public/INIT_BINDINGS.json',dict(verified=24,origin='INIT_POST80',tensor_step=80,new_steps=80,common_start_per_edit=True,checks=['payload SHA','input/support identity','science and backend','retained original W0 provenance','kind and actual step'],historical_code_bindings_preserved=True))
 write(out/'PREDECESSOR.json',dict(root=str(old),release_requires=['E4 activation','all queue complete','no live controller/worker','all old GPU sessions closed','SCORER_DONE']))
 for n in ['STORAGE_LEDGER.json']:
  if not (out/n).exists():write(out/n,dict(artifacts={},peak_bytes=0,deleted_bytes=0,deleted_count=0))
 arms=['A0','AK','AU','AH','AHS_001','AHS_01'];queue=[]
 for o in selection:queue.append(dict(id=f'P1-{o}',mode='train',order=o,methods=['E_orig']+arms,status='PENDING',requires=[]))
 for m in ['E_orig']+arms:queue.append(dict(id=f'P1-bank-{m}',mode='bank',method=m,orders=selection,prefixes=[4,8],status='PENDING',requires=[f'adapters/s20260929/{m}/e{o:03d}.pt' for o in selection]))
 write(out/'QUEUE.json',queue)
 write(out/'RUN_STATUS.json',dict(status='READY_WAITING_FOR_LEASE',gpu_started=False,phase='P1_DEV8',continuations=48,queue_jobs=len(queue),budget_blocker=False))
 print(json.dumps(dict(CHECK_positive=48,CHECK_negative=len(neg),init_verified=24,pilot_budget=budget,jobs=len(queue))))
if __name__=='__main__':main(*(Path(x) for x in sys.argv[1:]))
