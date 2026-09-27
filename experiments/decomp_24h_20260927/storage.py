"""Locked artifact reservations, two-generation saves and dependency-safe reclamation."""
import os,json,fcntl,time,hashlib,io,contextlib,shutil
from pathlib import Path
class Store:
 def __init__(self,root):
  self.root=Path(root).resolve();self.db=self.root/'STORAGE_LEDGER.json';self.policy=json.loads((self.root/'STORAGE_POLICY.json').read_text())
 def path(self,rel):
  p=self.root/rel;real=p.resolve()
  if not real.is_relative_to(self.root) or real==self.root:raise ValueError('Foreign or symlink-protected artifact')
  return p
 @contextlib.contextmanager
 def lock(self):
  with (self.root/'STORAGE.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);d=json.loads(self.db.read_text()) if self.db.exists() else dict(artifacts={},peak_bytes=0,deleted_bytes=0,deleted_count=0)
   yield d
   p=self.db.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));os.replace(p,self.db)
 def reserve(self,rel,size,kind,consumers=(),pin=False):
  self.path(rel)
  if size<0:raise ValueError('Negative reservation')
  with self.lock() as d:
   old=d['artifacts'].get(rel)
   if old and (old['readers'] or old['status']=='WRITING'):raise RuntimeError('Active reader/writer')
   total=sum(x['bytes']+(x.get('previous_bytes',0) if x['status']=='WRITING' else 0) for x in d['artifacts'].values() if x['status']!='DELETED')+self.policy.get('environment_reserved_bytes',0)
   # Existing bytes coexist with the new temporary during replacement.
   if total+size>self.policy['hard_bytes'] or shutil.disk_usage(self.root).free-size<self.policy['min_free_bytes']:raise RuntimeError('STORAGE_BLOCKED: total/free limit')
   category=sum(x['bytes']+(x.get('previous_bytes',0) if x['status']=='WRITING' else 0) for x in d['artifacts'].values() if x['kind']==kind and x['status']!='DELETED')
   if kind in ['checkpoint','teacher'] and category+size>self.policy[kind+'_bytes']:raise RuntimeError('STORAGE_BLOCKED: category limit')
   d['peak_bytes']=max(d['peak_bytes'],total+size)
   d['artifacts'][rel]=dict(real_path=str(self.path(rel)),owner_run=str(self.root),bytes=size,kind=kind,consumers=list(consumers),readers=0,pin=pin,status='WRITING',hash=None,verified_at=None,previous_bytes=old['bytes'] if old else 0)
 def commit(self,rel):
  p=self.path(rel)
  with self.lock() as d:
   a=d['artifacts'][rel];assert p.stat().st_size<=a['bytes'];a.update(bytes=p.stat().st_size,status='READY',hash=hashlib.sha256(p.read_bytes()).hexdigest(),verified_at=time.time())
 def write(self,rel,data,kind='evidence',consumers=(),pin=False):
  self.reserve(rel,len(data),kind,consumers,pin);p=self.path(rel);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
  with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p);self.commit(rel)
 @contextlib.contextmanager
 def reader(self,rel):
  with self.lock() as d:
   a=d['artifacts'][rel];assert a['status']=='READY';a['readers']+=1;a['accessed']=time.time()
  try:yield self.path(rel)
  finally:
   with self.lock() as d:d['artifacts'][rel]['readers']-=1
 def consumed(self,rel,consumer):
  with self.lock() as d:
   a=d['artifacts'][rel];a['consumers'].remove(consumer)
 def delete(self,rel,rebuildable_teacher=False):
  p=self.path(rel)
  with self.lock() as d:
   a=d['artifacts'][rel]
   if a['status']!='READY' or a['pin'] or a['readers'] or (a['consumers'] and not (rebuildable_teacher and a['kind']=='teacher')):raise RuntimeError('Artifact has live dependencies')
   receipt=dict(path=rel,bytes=a['bytes'],hash=a['hash'],epoch=time.time(),reason='regenerable teacher eviction' if rebuildable_teacher else 'all registered binary consumers completed')
   plan=self.root/'CLEANUP_PENDING.json';plan.write_text(json.dumps(receipt));p.unlink();a['status']='DELETED';d['deleted_bytes']+=a['bytes'];d['deleted_count']+=1
   with (self.root/'CLEANUP_RECEIPT.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
 def save(self,rel,payload,consumers=(),pin=False):
  import torch
  allowed={'expert','optimizer','step','seed','stage','torch_rng','cuda_rng','python_rng','sampler','curve','edit','binding','cursor','forwards','tokens','kind'}
  assert isinstance(payload,dict) and set(payload)<=allowed and 'expert' in payload,'Checkpoint whitelist'
  def inspect(x):
   if torch.is_tensor(x):assert x.numel()<=1000000,'Dense/Base tensor forbidden'
   elif isinstance(x,dict):
    for v in x.values():inspect(v)
   elif isinstance(x,(list,tuple)):
    for v in x:inspect(v)
  def plain(x):
   if isinstance(x,str):return str(x) # TorchVersion is a string subclass rejected by weights_only.
   if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
   if isinstance(x,list):return [plain(v) for v in x]
   if isinstance(x,tuple):return tuple(plain(v) for v in x)
   if x is None or type(x) in (int,float,bool) or torch.is_tensor(x):return x
   raise TypeError('Nonprimitive checkpoint metadata: '+type(x).__name__)
  payload=plain(payload);inspect(payload);b=io.BytesIO();torch.save(payload,b);data=b.getvalue();self.reserve(rel,len(data),'checkpoint',consumers,pin)
  p=self.path(rel);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
  with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  check=torch.load(tmp,map_location='cpu',weights_only=True)
  assert check.get('step')==payload.get('step') and check.get('seed')==payload.get('seed')
  assert all(torch.equal(v.detach().cpu(),check['expert'][k]) for k,v in payload['expert'].items())
  if p.exists():
   prev=p.with_name('previous.pt');prevrel=str(prev.relative_to(self.root));olddata=p.read_bytes();self.write(prevrel,olddata,'checkpoint',pin=True)
  os.replace(tmp,p);self.commit(rel)
 def evict_teacher(self,needed=0):
  with self.lock() as d:rows=sorted([(a.get('accessed',0),r) for r,a in d['artifacts'].items() if a['kind']=='teacher' and not a['readers'] and not a['pin'] and a['status']=='READY'])
  freed=0
  for _,rel in rows:
   size=self.path(rel).stat().st_size;self.delete(rel,True);freed+=size
   if freed>=needed:break
  return freed
