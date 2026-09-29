"""Scientific identity is distinct from implementation/report revisions."""
import json,hashlib
from functools import lru_cache
from pathlib import Path
from storage import Store,digest_file
from resources import ROOT,read

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def g_supports():
 path=ROOT/'private/G_SUPPORTS.json';amendment=ROOT/'SUPPORT_REPAIR.json'
 if amendment.exists():
  a=read(amendment);assert digest_file(path)==a['original_sha256'];path=ROOT/a['repaired_path'];assert digest_file(path)==a['repaired_sha256']
 return {x['order']:x for x in read(path)}

def state_digest(expert):
 h=hashlib.sha256()
 for name,tensor in expert.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

@lru_cache(maxsize=256)
def initialization(edit,seed,kind):
 import torch
 from structures import TuckerC4,LR4
 from methods.medtrace import AsymmetricCPExpert
 from scripts.medtrace.run_dev16 import derive_seed
 actual_seed=derive_seed(edit,seed)
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(actual_seed)
  expert={'CP':lambda:AsymmetricCPExpert(14336,4096,4),'TK':TuckerC4,'LR':LR4}[kind]()
  value=state_digest(expert)
 return dict(kind=kind,run_seed=seed,derived_seed=actual_seed,state_sha256=value)

def context(t,seed,kind):
 g=g_supports()[t['order']];lock=read(ROOT/'SCIENCE_LOCK.json')
 return dict(science_id=lock['id'],edit=t['canonical_edit_id'],seed=seed,structure=kind,input_target_support=digest(dict(native=t['native'],P=t['semantic_fit_questions'],Uold=t['U_fit'],Unew=t['U_new'],G=g)),initializer=initialization(t['canonical_edit_id'],seed,kind),implementation={name:digest_file(ROOT/name) for name in ['structures.py','training.py']},optimizer='native/A2 AdamW1e-3; W0/continuation Adam input1e-4 output1e-3; clip1',protocol='original locked stage counts/stops, masks and full-vocabulary KL',backend=lock['backend'])
def expected(t,seed,kind,method,stage,step):
 imported=read(ROOT/'IMPORTED_BINDINGS.json').get(f'adapters/s{seed}/{method}/e{t["order"]:03d}.pt')
 if imported:
  assert method=='M0' and stage=='continuation' and step==80
  assert digest_file(ROOT/imported['path'])==imported['sha256']
  return imported['parent_expected']
 c=context(t,seed,kind);c.update(method=method,stage=stage,actual_step=step,loss=dict(native=1 if stage=='native' else .5,P=0 if stage=='native' else .25 if method in ['M2','M3','M5','M7'] and stage=='continuation' else .5,G=.25 if method in ['M2','M3','M5','M7'] and stage=='continuation' else 0,U=.01 if stage=='continuation' else 0,D=.10 if method in ['M3','M5','M7'] and stage=='continuation' else 0))
 if '_E3_' in method:c['gauge_protocol']=dict(variant=method.split('_E3_')[1],implementation=digest_file(ROOT/'e3.py'),rank=4,fresh_optimizer=True)
 if stage=='continuation':
  wbase='W0_E3' if '_E3_' in method else 'W0';rel=f'checkpoints/{wbase}/s{seed}/{kind}/e{t["order"]:03d}.pt';ledger=read(ROOT/'STORAGE_LEDGER.json');entry=ledger['artifacts'].get(rel);assert entry and entry.get('hash'),'Missing W0 provenance';c['initialization_W0_hash']=entry['hash']
  # A later repair can rebuild W0 while valid old siblings keep their original provenance.
  migration=ROOT/'fix/pr5_v1/LEGACY_BINDING_MIGRATION.json';final=f'adapters/s{seed}/{method}/e{t["order"]:03d}.pt'
  if migration.exists():
   prior=read(migration).get('artifacts',{}).get(final)
   if prior and (ROOT/final).exists() and digest_file(ROOT/final)==prior['sha256'] and all(prior['expected'].get(k)==v for k,v in c.items() if k!='initialization_W0_hash'):c['initialization_W0_hash']=prior['expected']['initialization_W0_hash']
  if method in ['M3','M5','M7']:
   teacher=f'adapters/s{seed}/M0/e{t["order"]:03d}.pt';c['teacher_hash']=digest_file(ROOT/teacher)
 return c

def check_payload(x,rel,want):
 imported=read(ROOT/'IMPORTED_BINDINGS.json').get(rel)
 if imported:
  assert digest_file(ROOT/rel)==imported['sha256'],'Imported teacher changed'
  assert want==imported['parent_expected'] and x['binding']==imported['binding'],'Imported teacher identity mismatch'
  assert x['step']==want['actual_step']==80 and x.get('expert'),'Imported teacher step/payload mismatch'
  return
 if not isinstance(x.get('expert'),dict) or not x['expert']:raise ValueError('Empty expert payload')
 if x.get('step')!=want['actual_step']:raise ValueError('Actual step mismatch')
 if x.get('binding')==want:return
 migration=read(ROOT/'fix/pr5_v1/LEGACY_BINDING_MIGRATION.json') if (ROOT/'fix/pr5_v1/LEGACY_BINDING_MIGRATION.json').exists() else {}
 allowed=migration.get('artifacts',{}).get(rel)
 if not allowed or allowed['sha256']!=digest_file(ROOT/rel) or allowed['expected']!=want:raise ValueError('Binding mismatch; quarantine instead of overwriting: '+rel)

def load_expected(rel,want):return Store(ROOT).load_checkpoint(rel,lambda x,path:check_payload(x,path,want))
def rolling(rel,t,seed,kind,method=None):
 if not (ROOT/rel).exists():return None
 store=Store(ROOT);x=store.load_checkpoint(rel,fallback=True);loaded_rel=store.last_loaded_path;b=x['binding']
 if b.get('edit')!=t['canonical_edit_id'] or b.get('method')!=(method or kind):return None
 stage=b.get('stage');want=expected(t,seed,kind,method or kind,stage,x['step']);check_payload(x,loaded_rel,want)
 if 'optimizer' in x:
  ids=[v for g in x['optimizer']['param_groups'] for v in g['params']];assert len(ids)==len(set(ids)) and set(x['optimizer']['state'])<=set(ids)
 return x

def verify_science_files():
 lock=read(ROOT/'SCIENCE_LOCK.json')
 for p,info in lock['model_assets'].items():assert Path(p).stat().st_size==info['bytes'],'Model size changed'
 for n,info in lock['imports'].items():assert digest_file(info['path'])==info['sha256'],'Scientific dependency changed: '+n
 for name,h in lock['roles'].items():assert digest_file(ROOT/'private'/name)==h,'Frozen role content changed'
 return lock['id']
