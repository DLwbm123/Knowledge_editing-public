"""New continuations bind INIT_POST80, never relabel it as W0."""
import os,json,hashlib
from pathlib import Path
from resources import ROOT,read
from storage import Store,digest_file

def expected(t,seed,kind,method,stage,step):
 assert stage=='continuation' and kind=='LR'
 init=next(x for x in read(ROOT/'private/INITIALIZERS.json') if x['order']==t['order'])
 return dict(science_id=read(ROOT/'SCIENCE_LOCK.json')['id'],edit=t['canonical_edit_id'],seed=seed,method=method,stage=stage,actual_step=step,origin='INIT_POST80',initialization_sha256=init['sha256'],kind=kind,implementation={n:digest_file(ROOT/n) for n in ['training.py','protection.py','mechanisms.py','scope_worker.py']},fresh_optimizer=True)
def load_expected(rel,want):
 def validate(x,p):
  assert x['binding']==want and x['step']==want['actual_step'] and x['expert'],'Checkpoint binding mismatch'
 return Store(ROOT).load_checkpoint(rel,validate)
def rolling(rel,t,seed,kind,method):
 if not (ROOT/rel).exists():return None
 x=Store(ROOT).load_checkpoint(rel,fallback=True)
 if x['binding']['edit']!=t['canonical_edit_id'] or x['binding']['method']!=method:return None
 assert x['binding']==expected(t,seed,kind,method,'continuation',x['step'])
 return x
