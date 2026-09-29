"""Deterministic mining on frozen Base features only; shared AU/AH candidate universe."""
import torch

def select(rows,features,key,radius,limit=16,per_source=2):
 if not rows or len(features)!=len(rows):raise ValueError('Need the full frozen pool features; do not mine a convenient cached subset')
 radius=float(radius)
 if radius<0 or not torch.isfinite(key).all():raise ValueError('Invalid frozen router')
 scored=[]
 for r,z in zip(rows,features):
  if r['scope']!='negative' or r['purpose']!='U_bg':raise ValueError('Only certified fit background allowed')
  if not torch.isfinite(z).all():raise ValueError('Invalid Base feature')
  scored.append((float(torch.linalg.vector_norm(z-key)),r['input_hash'],r))
 selected=[];counts={}
 for d,_,r in sorted(scored,key=lambda x:(x[0],x[1])):
  source=r['source_group']
  if counts.get(source,0)>=per_source:continue
  selected.append(dict(input_hash=r['input_hash'],source_group=source,distance=d,R0_would_activate=d<=radius));counts[source]=counts.get(source,0)+1
  if len(selected)==limit:break
 return selected

if __name__=='__main__':
 rows=[dict(input_hash=str(i),source_group=str(i//3),scope='negative',purpose='U_bg') for i in range(6)]
 result=select(rows,[torch.tensor([float(i)]) for i in range(6)],torch.zeros(1),torch.tensor(1.))
 import json;json.dumps(result)
 assert [x['input_hash'] for x in result]==['0','1','3','4']
 print('PASS: frozen distance ranking, hash tie rule, source cap and limited support')
