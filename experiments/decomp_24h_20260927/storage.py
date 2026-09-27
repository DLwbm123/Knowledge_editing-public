"""Recoverable, owner-bound file transactions with bounded control/log reserves."""
import os,json,fcntl,time,hashlib,io,contextlib,shutil,uuid,copy
from pathlib import Path

def digest_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def owner():
 p=Path('/proc/self/stat');return dict(pid=os.getpid(),start=p.read_text().split()[21] if p.exists() else None)
def alive(w):
 try:
  p=Path('/proc')/str(w['pid']);return p.exists() and p.joinpath('stat').read_text().split()[2]!='Z' and (w.get('start') is None or p.joinpath('stat').read_text().split()[21]==w['start'])
 except FileNotFoundError:return False

def atomic_control(p,data,cap=16*1024**2):
 if len(data)>cap:raise RuntimeError('CONTROL_FILE_LIMIT')
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.control.tmp')
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

class Store:
 def __init__(self,root):
  self.root=Path(root).resolve();self.db=self.root/'STORAGE_LEDGER.json';self.policy=json.loads((self.root/'STORAGE_POLICY.json').read_text());self.limits=self.policy.get('quotas_enabled',True)
 def path(self,rel):
  p=self.root/rel;real=p.resolve()
  if not real.is_relative_to(self.root) or real==self.root:raise ValueError('Foreign or symlink-protected artifact')
  return p
 @contextlib.contextmanager
 def lock(self):
  with (self.root/'STORAGE.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);d=json.loads(self.db.read_text()) if self.db.exists() else dict(artifacts={},peak_bytes=0,deleted_bytes=0,deleted_count=0)
   yield d
   atomic_control(self.db,json.dumps(d).encode())
 def reserve(self,rel,size,kind,consumers=(),pin=False,expected_hash=None):
  p=self.path(rel)
  if size<0:raise ValueError('Negative reservation')
  with self.lock() as d:
   old=d['artifacts'].get(rel)
   if old and (old['readers'] or old['status']=='WRITING'):raise RuntimeError('Active reader/writer; reconcile only verified dead owner')
   rows=[x for x in d['artifacts'].values() if x['status']!='DELETED'];managed=sum(x['bytes']+x.get('previous_bytes',0) for x in rows)
   total=managed+self.policy.get('environment_reserved_bytes',0)+self.policy.get('control_reserved_bytes',0)
   emergency=0 if kind=='checkpoint' else self.policy.get('emergency_bytes',0)
   if (self.limits and total+size+emergency>self.policy['hard_bytes']) or shutil.disk_usage(self.root).free-size<self.policy['min_free_bytes']:raise RuntimeError('STORAGE_BLOCKED: total/free limit')
   category=sum(x['bytes']+x.get('previous_bytes',0) for x in rows if x['kind']==kind)
   if self.limits and kind in ['checkpoint','teacher'] and category+size>self.policy[kind+'_bytes']:raise RuntimeError('STORAGE_BLOCKED: category limit')
   durable=sum(a['bytes'] for name,a in d['artifacts'].items() if a['status']!='DELETED' and (a['kind'] not in ['checkpoint','teacher'] or name.startswith('adapters/')))
   if self.limits and (kind not in ['checkpoint','teacher'] or rel.startswith('adapters/')) and durable+size>self.policy.get('final_total_bytes',self.policy['hard_bytes']):raise RuntimeError('FINAL_DELIVERY_STORAGE_BLOCKED')
   if self.limits and rel.startswith('adapters/') and sum(a['bytes'] for name,a in d['artifacts'].items() if name.startswith('adapters/') and a['status']!='DELETED')+size>self.policy.get('final_model_bytes',self.policy['hard_bytes']):raise RuntimeError('FINAL_MODEL_STORAGE_BLOCKED')
   tx=uuid.uuid4().hex;d['peak_bytes']=max(d['peak_bytes'],total+size)
   d['artifacts'][rel]=dict(real_path=str(p),owner_run=str(self.root),bytes=size,kind=kind,consumers=list(consumers),readers=0,pin=pin,status='WRITING',hash=None,verified_at=None,previous_bytes=old['bytes'] if old and old['status']!='DELETED' else 0,old_ready=copy.deepcopy(old) if old and old['status']=='READY' else None,writer=owner(),transaction=tx,tmp_path=rel+'.txn-'+tx+'.tmp',expected_hash=expected_hash)
   return tx
 def _finish(self,d,rel,a):
  p=self.path(rel);assert p.stat().st_size<=a['bytes'];a.update(bytes=p.stat().st_size,status='READY',hash=digest_file(p),verified_at=time.time(),previous_bytes=0)
  for k in ['old_ready','writer','transaction','tmp_path','expected_hash']:a.pop(k,None)
 def reconcile(self,rel,own_transaction=None):
  with self.lock() as d:
   a=d['artifacts'][rel]
   if a['status']!='WRITING':return a['status']
   if a['readers']:raise RuntimeError('Active reader')
   own=own_transaction==a.get('transaction') and a.get('writer')==owner()
   if not own and (not a.get('writer') or alive(a['writer'])):raise RuntimeError('Writer not proven dead')
   p=self.path(rel);tmp=self.path(a['tmp_path']);expected=a.get('expected_hash');valid=None
   for candidate in [p,tmp]:
    if expected and candidate.is_file() and digest_file(candidate)==expected and (candidate==p or not own or a.get('verified')):
     if a['kind']=='checkpoint':
      import torch
      x=torch.load(candidate,map_location='cpu',weights_only=True);assert 'expert' in x and 'step' in x and 'seed' in x
     valid=candidate;break
   if valid:
    if valid==tmp:
     old=a.get('old_ready')
     if old and p.is_file() and digest_file(p)==old['hash'] and p.name=='latest.pt':
      extra=p.stat().st_size;total=sum(x['bytes']+x.get('previous_bytes',0) for x in d['artifacts'].values() if x['status']!='DELETED')+self.policy.get('environment_reserved_bytes',0)+self.policy.get('control_reserved_bytes',0);assert not self.limits or total+extra<=self.policy['hard_bytes'],'Recovery copy reservation exceeds hard cap'
      prev=p.with_name('previous.pt');data=p.read_bytes();atomic_control(prev,data,cap=self.policy['checkpoint_bytes']);prevrel=str(prev.relative_to(self.root));d['artifacts'][prevrel]=dict(old,real_path=str(prev),consumers=[],pin=True)
     os.replace(tmp,p)
    self._finish(d,rel,a);result='NEW_COMMITTED'
   else:
    old=a.get('old_ready')
    if old and p.is_file() and digest_file(p)==old['hash']:d['artifacts'][rel]=old;result='OLD_RESTORED'
    elif old:raise RuntimeError('No verified old or new generation; preserve all evidence')
    else:a.update(status='DELETED',bytes=0,previous_bytes=0);result='EMPTY_ABORTED'
   # Only this transaction's invalid temporary is reclaimable after owner proof.
   if tmp.exists():tmp.unlink()
   d.setdefault('recoveries',[]).append(dict(path=rel,transaction=a.get('transaction'),result=result,epoch=time.time()))
   return result
 def commit(self,rel):
  with self.lock() as d:self._finish(d,rel,d['artifacts'][rel])
 def _write(self,rel,data,kind,consumers,pin,verify=None,fault=None,exclusive=False):
  # Serialize the entire transaction for a shared artifact, not just its reservation.
  self.path(rel);locks=self.root/'WRITE_LOCKS';locks.mkdir(exist_ok=True)
  with (locks/(hashlib.sha256(rel.encode()).hexdigest()+'.lock')).open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   with self.lock() as d:pending=d['artifacts'].get(rel,{}).get('status')=='WRITING'
   if pending:self.reconcile(rel)
   return self._write_locked(rel,data,kind,consumers,pin,verify,fault,exclusive)
 def _write_locked(self,rel,data,kind,consumers,pin,verify=None,fault=None,exclusive=False):
  tx=self.reserve(rel,len(data),kind,consumers,pin,hashlib.sha256(data).hexdigest())
  with self.lock() as d:tmp=self.path(d['artifacts'][rel]['tmp_path'])
  p=self.path(rel);p.parent.mkdir(parents=True,exist_ok=True)
  try:
   if fault:fault('reserve')
   with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
   if fault:fault('fsync')
   if verify:verify(tmp)
   with self.lock() as d:d['artifacts'][rel]['verified']=True
   if fault:fault('verify')
   if p.exists() and kind=='checkpoint' and p.name=='latest.pt':self.write(str(p.with_name('previous.pt').relative_to(self.root)),p.read_bytes(),'checkpoint',pin=True)
   if exclusive:os.link(tmp,p);tmp.unlink()
   else:os.replace(tmp,p)
   if fault:fault('replace')
   self.commit(rel)
  except BaseException:
   self.reconcile(rel,own_transaction=tx)
   raise
 def write(self,rel,data,kind='evidence',consumers=(),pin=False,exclusive=False):self._write(rel,data,kind,consumers,pin,exclusive=exclusive)
 @contextlib.contextmanager
 def reader(self,rel):
  with self.lock() as d:
   a=d['artifacts'][rel];assert a['status']=='READY';a['readers']+=1;a['accessed']=time.time()
  try:yield self.path(rel)
  finally:
   with self.lock() as d:d['artifacts'][rel]['readers']-=1
 def load_checkpoint(self,rel,validator=None,fallback=False):
  import torch
  failures=[]
  for name in [rel]+([str(Path(rel).with_name('previous.pt'))] if fallback else []):
   try:
    with self.lock() as d:a=copy.deepcopy(d['artifacts'][name])
    if a['status']=='WRITING':self.reconcile(name)
    with self.reader(name) as p:
     with self.lock() as d:expected=d['artifacts'][name]['hash']
     if digest_file(p)!=expected:raise OSError('Checkpoint digest mismatch')
     x=torch.load(p,map_location='cpu',weights_only=True)
    if validator:validator(x,name)
    if failures:
     with self.lock() as d:d.setdefault('fallbacks',[]).append(dict(requested=rel,used=name,failures=failures,step=x['step'],epoch=time.time()))
    self.last_loaded_path=name;return x
   except (OSError,EOFError,KeyError,RuntimeError) as e:failures.append(dict(path=name,error=str(e)))
  raise RuntimeError('No valid checkpoint generation: '+json.dumps(failures))
 def consumed(self,rel,consumer):
  with self.lock() as d:d['artifacts'][rel]['consumers'].remove(consumer)
 def delete(self,rel,rebuildable_teacher=False):
  p=self.path(rel)
  with self.lock() as d:
   a=d['artifacts'][rel]
   if a['status']!='READY' or a['pin'] or a['readers'] or (a['consumers'] and not (rebuildable_teacher and a['kind']=='teacher')):raise RuntimeError('Artifact has live dependencies')
   receipt=dict(path=rel,bytes=a['bytes'],hash=a['hash'],epoch=time.time(),reason='regenerable teacher eviction' if rebuildable_teacher else 'all binary consumers completed')
   atomic_control(self.root/'CLEANUP_PENDING.json',json.dumps(receipt).encode());p.unlink();a.update(status='DELETED',previous_bytes=0);d['deleted_bytes']+=a['bytes'];d['deleted_count']+=1
   rp=self.root/'CLEANUP_RECEIPT.jsonl';prior=rp.read_bytes() if rp.exists() else b'';atomic_control(rp,prior+json.dumps(receipt).encode()+b'\n')
 def save(self,rel,payload,consumers=(),pin=False,fault=None):
  import torch
  allowed={'expert','optimizer','step','seed','stage','torch_rng','cuda_rng','python_rng','sampler','curve','edit','binding','cursor','forwards','tokens','kind'}
  assert isinstance(payload,dict) and set(payload)<=allowed and 'expert' in payload,'Checkpoint whitelist'
  def plain(x):
   if torch.is_tensor(x):assert x.numel()<=1000000,'Dense/Base tensor forbidden';return x
   if isinstance(x,str):return str(x)
   if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
   if isinstance(x,list):return [plain(v) for v in x]
   if isinstance(x,tuple):return tuple(plain(v) for v in x)
   if x is None or type(x) in (int,float,bool):return x
   raise TypeError('Nonprimitive checkpoint metadata: '+type(x).__name__)
  payload=plain(payload);b=io.BytesIO();torch.save(payload,b)
  def verify(tmp):
   check=torch.load(tmp,map_location='cpu',weights_only=True);assert check.get('step')==payload.get('step') and check.get('seed')==payload.get('seed');assert all(torch.equal(v.detach().cpu(),check['expert'][k]) for k,v in payload['expert'].items())
  self._write(rel,b.getvalue(),'checkpoint',consumers,pin,verify,fault)
 def evict_teacher(self,needed=0):
  with self.lock() as d:rows=sorted((a.get('accessed',0),r) for r,a in d['artifacts'].items() if a['kind']=='teacher' and not a['readers'] and not a['pin'] and a['status']=='READY')
  freed=0
  for _,rel in rows:
   size=self.path(rel).stat().st_size;self.delete(rel,True);freed+=size
   if freed>=needed:break
  return freed
 def disk_audit(self):
  # Registered payloads plus explicitly bounded control/log and external pools.
  with self.lock() as d:
   untracked=logs=0
   for p in self.root.rglob('*'):
    if not p.is_file() or p.is_symlink():continue
    rel=str(p.relative_to(self.root))
    if rel.startswith('logs/'):logs+=p.stat().st_size;continue
    if rel in ['STORAGE_LEDGER.json','STORAGE_LEDGER.control.tmp','STORAGE.lock','CLEANUP_PENDING.json','CLEANUP_RECEIPT.jsonl'] or '.txn-' in rel or rel.endswith('.lock'):continue
    if rel not in d['artifacts']:
     size=p.stat().st_size;untracked+=size;d['artifacts'][rel]=dict(real_path=str(p),owner_run=str(self.root),bytes=size,kind='evidence',consumers=[],readers=0,pin=True,status='READY',hash=digest_file(p),verified_at=time.time(),adopted_no_delete=True)
   managed=sum(a['bytes']+a.get('previous_bytes',0) for a in d['artifacts'].values() if a['status']!='DELETED');total=managed+self.policy.get('environment_reserved_bytes',0)+self.policy.get('control_reserved_bytes',0)
   import subprocess
   external=sum(int(subprocess.check_output(['du','-sk',p],text=True).split()[0])*1024 for p in self.policy.get('external_owned_roots',[]))
   assert not self.limits or external<=self.policy.get('environment_reserved_bytes',0),'EXTERNAL_POOL_LIMIT'
   controls=sum(p.stat().st_size for p in self.root.glob('STORAGE*') if p.is_file())
   assert not self.limits or logs+controls<=self.policy.get('control_reserved_bytes',0),'LOG_POOL_LIMIT'
   actual=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())
   assert not self.limits or actual+external+self.policy.get('local_scorer_reserved_bytes',0)<=self.policy['hard_bytes'],'ACTUAL_STORAGE_LIMIT'
   assert not self.limits or total<=self.policy['hard_bytes'],'REAL_STORAGE_LIMIT'
   return dict(managed_bytes=managed,actual_owned_bytes=actual,logs_bytes=logs,external_actual_bytes=external,newly_accounted=untracked,reserved_total=total,quotas_enabled=self.limits,soft_reached=self.limits and total>=self.policy['soft_bytes'])
