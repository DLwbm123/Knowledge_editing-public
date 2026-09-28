"""One runnable preflight: concurrent quotas, protected artifacts and rolling recovery."""
import json,tempfile,multiprocessing,torch,os
from pathlib import Path
from storage import Store
from structures import LR4,TuckerC4

def contender(root,i,q):
 try:Store(root).write(str(i),b'x'*600,kind='evidence');q.put(True)
 except RuntimeError:q.put(False)
def main():
 root=Path(tempfile.mkdtemp(dir=os.environ.get('TMPDIR')));policy=dict(hard_bytes=1000,soft_bytes=800,checkpoint_bytes=900,teacher_bytes=900,min_free_bytes=0)
 (root/'STORAGE_POLICY.json').write_text(json.dumps(policy));q=multiprocessing.Queue();ps=[multiprocessing.Process(target=contender,args=(root,i,q)) for i in range(3)]
 for p in ps:p.start()
 for p in ps:p.join()
 assert sum(q.get() for _ in ps)==1
 policy.update(hard_bytes=20*1024**2,checkpoint_bytes=10*1024**2,teacher_bytes=10*1024**2);(root/'STORAGE_POLICY.json').write_text(json.dumps(policy));s=Store(root)
 s.write('pin',b'x',pin=True)
 try:s.delete('pin');raise AssertionError('pin deleted')
 except RuntimeError:pass
 (root/'foreign').symlink_to('/tmp')
 try:s.write('foreign/never',b'x');raise AssertionError('symlink escaped')
 except ValueError:pass
 s.write('teacher',b'x',kind='teacher',consumers=['future'])
 with s.reader('teacher'):assert s.evict_teacher(1)==0
 assert s.evict_teacher(1)==1
 s.write('dependency',b'x',consumers=['consumer'])
 try:s.delete('dependency');raise AssertionError('dependency deleted')
 except RuntimeError:pass
 s.consumed('dependency','consumer');s.delete('dependency')
 x=LR4(16,8);opt=torch.optim.Adam(x.parameters());torch.manual_seed(9)
 payload=dict(expert=x.state_dict(),optimizer=opt.state_dict(),step=20,seed=9,torch_rng=torch.get_rng_state(),binding={'torch':torch.__version__})
 s.save('rolling/latest.pt',payload,pin=True);payload['step']=40;s.save('rolling/latest.pt',payload,pin=True)
 (root/'rolling/latest.tmp').write_bytes(b'crashed write')
 assert torch.load(root/'rolling/previous.pt',weights_only=True)['step']==20
 saved=torch.load(root/'rolling/latest.pt',weights_only=True);assert type(saved['binding']['torch']) is str;assert saved['step']==40 and torch.equal(saved['torch_rng'],payload['torch_rng'])
 try:s.save('forbidden.pt',dict(expert={'base':torch.zeros(1000001)}));raise AssertionError('dense accepted')
 except AssertionError as e:assert 'Dense' in str(e)
 t=TuckerC4();assert sum(p.numel() for p in t.parameters())==1600
 h=torch.randn(3,14336);target=torch.randn(3,4096);opt=torch.optim.Adam(t.parameters(),lr=.001)
 assert t.residual(h).abs().max()==0
 norms=[]
 for i in range(3):
  opt.zero_grad();(t.residual(h)-target).square().mean().backward();norms.append({n:float(p.grad.norm()) for n,p in t.named_parameters()});opt.step()
 assert norms[0]['L']>0 and all(v>0 for v in norms[-1].values())
 b,a=t.factors();assert torch.allclose(t.residual(h),((h/(h.square().mean(-1,keepdim=True).sqrt()+1e-6))@a.T)@b.T)
 print(json.dumps(dict(status='PASS',concurrent_reservation=True,previous_survives_partial_write=True,protected_deletion=True,no_frozen_base=True,Tucker_parameters=1600,Tucker_gradient_paths=norms)))
if __name__=='__main__':main()
