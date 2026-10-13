"""New namespace binds functional anchor, initialization and immutable old controls."""
import os,json,hashlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def expected(t,seed,kind,method,stage,step):
    assert stage=='continuation' and kind=='LR' and seed==20260929
    init=next(x for x in read(ROOT/'private/INITIALIZERS.json') if x['order']==t['order'])
    files=['training.py','anchor.py','protection.py','mechanisms.py','scope_worker.py']
    return dict(science_id=read(ROOT/'SCIENCE_LOCK.json')['id'],edit=t['canonical_edit_id'],seed=seed,method=method,stage=stage,actual_step=step,origin='INIT_POST80',initialization_sha256=init['origin_sha256'],carrier_sha256=init['sha256'],kind=kind,implementation={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},fresh_optimizer=True)
def load_expected(rel,want):
    from storage import Store
    def validate(x,p):
        assert x['binding']==want and x['step']==want['actual_step'] and x['expert'],'Checkpoint binding mismatch'
    return Store(ROOT).load_checkpoint(rel,validate)
def rolling(rel,t,seed,kind,method):
    if not (ROOT/rel).exists():return None
    from storage import Store
    x=Store(ROOT).load_checkpoint(rel,fallback=False)
    if x['binding']['edit']!=t['canonical_edit_id'] or x['binding']['method']!=method:return None
    assert x['binding']==expected(t,seed,kind,method,'continuation',x['step'])
    return x
