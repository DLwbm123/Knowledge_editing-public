"""Frozen full six-arm DEV24 extension, no score-dependent selection."""
def jobs():
 result=[]
 for o in range(1,25):
  for kind,methods in [('LR',['M6','M7']),('CP',['M1','M1_STRUCT']),('TK',['M4','M4_STRUCT'])]:
   result.append(dict(id=f'DEV-{kind}-{o}',block='E1_E2_DEV24',mode='train',kind=kind,methods=methods,seed=20260927,order=o,dev_canary=o==1,requires=[] if o==1 else ['DEV_CANARY_PASS.json'],status='PENDING'))
 for m in ['M6','M7','M1','M1_STRUCT','M4','M4_STRUCT']:
  result.append(dict(id=f'DEV24-{m}-bank',block='E1_E2_DEV24',mode='sequential',seed=20260927,method=m,orders=list(range(1,25)),prefixes=[12,24],requires=['DEV_CANARY_PASS.json']+[f'adapters/s20260927/{m}/e{o:03d}.pt' for o in range(1,25)],status='PENDING'))
 return result

if __name__=='__main__':
 q=jobs();assert len(q)==78 and sum(len(j.get('methods',[])) for j in q)==144
 assert sum(j.get('dev_canary',False) for j in q)==3
 assert all('DEV_CANARY_PASS.json' in j['requires'] for j in q if not j.get('dev_canary'))
 assert len({j['id'] for j in q})==78
 print('PASS: 78 unique jobs, 144 continuations, three canaries and all six paired banks')
