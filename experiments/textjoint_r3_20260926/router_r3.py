"""R0 nearest-entry routing with only the predeclared R1 rejection gate."""
from dataclasses import asdict,replace
from m3bench_repro.editors.routing import MemoryRouter,distances
import torch
class RejectRouter(MemoryRouter):
 def __init__(self,kappa=1.,mu=0.):super().__init__('euclidean');self.kappa=kappa;self.mu=mu
 def diagnostic(self,q):
  original=super().route(q)
  if not self.keys:return original,dict(d1=None,d2=None,radius=None,ratio=None,margin=None)
  ds=distances(self._key_matrix(q.device),q,'euclidean');ordered=torch.sort(ds).values
  d1=float(ordered[0]);d2=float(ordered[1]) if len(ordered)>1 else None
  margin=1. if d2 is None else (0. if d2<=1e-12 else (d2-d1)/(d2+1e-12))
  radius=original.radius
  active=original.activated and d1<=self.kappa*radius and margin>=self.mu
  decision=replace(original,logical_edit_id=original.nearest_logical_edit_id if active else None,activated=active)
  return decision,dict(d1=d1,d2=d2,radius=radius,ratio=d1/radius if radius else (0. if not d1 else None),margin=margin,R0_activated=original.activated)
 def route(self,q):return self.diagnostic(q)[0]
def calibrate(entries,positives,negatives):
 prefixes=[1,4,8,12,24];grid=[]
 def measure(kappa,mu):
  out=[]
  for n in prefixes:
   router=RejectRouter(kappa,mu)
   for e in entries[:n]:router.add(e['edit'],e['key'],e['radius'])
   ps=[p for p in positives if p['order']<=n]
   row=dict(prefix=n)
   for role in ['native','new_paraphrase']:
    rows=[p for p in ps if p['role']==role]
    row[role+'_hit']=sum(router.route(p['key']).logical_edit_id==p['associated_edit'] for p in rows)/len(rows)
   row['negative_activation']=sum(router.route(p['key']).activated for p in negatives)/len(negatives)
   out.append(row)
  return out
 baseline=measure(1.,0.)
 for k in [1.,.85,.70,.55,.40,.25,.10]:
  for mu in [0.,.05,.10]:
   rows=measure(k,mu)
   legal=all(r['native_hit']>=b['native_hit'] and r['new_paraphrase_hit']>=b['new_paraphrase_hit']-.01-1e-12 for r,b in zip(rows,baseline))
   grid.append(dict(kappa=k,mu=mu,eligible=legal,prefixes=rows,negative_mean=sum(r['negative_activation'] for r in rows)/5,positive_mean=sum(r['native_hit']+r['new_paraphrase_hit'] for r in rows)/10))
 winners=sorted([g for g in grid if g['eligible']],key=lambda g:(g['negative_mean'],-g['positive_mean'],g['mu'],-g['kappa']))
 best=winners[0];base=grid[0]
 if best['negative_mean']>=base['negative_mean']:best=base;status='NO_ROUTING_GAIN'
 else:status='LOCKED'
 return dict(status=status,kappa=best['kappa'],mu=best['mu'],prefixes=prefixes,selection='minimum equal-prefix negative activation; tie positive hit, smaller mu, larger kappa',grid=grid)
def selfcheck():
 r=RejectRouter(.5,.05);r.add('a',torch.tensor([0.,0.]),1);r.add('b',torch.tensor([2.,0.]),1)
 assert r.route(torch.tensor([.1,0.])).logical_edit_id=='a'
 assert not r.route(torch.tensor([1.,0.])).activated
 r=RejectRouter(1,.05);r.add('a',torch.zeros(2),0);r.add('b',torch.zeros(2),0)
 assert r.diagnostic(torch.zeros(2))[1]['margin']==0 and not r.route(torch.zeros(2)).activated
 r=RejectRouter(1,0);r.add('a',torch.zeros(2),0)
 assert r.route(torch.zeros(2)).activated and not r.route(torch.tensor([1e-8,0.])).activated
if __name__=='__main__':selfcheck();print('PASS: nearest, tie, ambiguity, zero radius, single margin')
