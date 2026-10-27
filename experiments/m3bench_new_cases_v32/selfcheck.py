"""Consumer, shard and data-contract checks with no real model."""
import json,tempfile
from pathlib import Path
import consumers as c

def main():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)
  def w(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
  tasks=[]
  for i in range(6):
   key=f'X{i}';rows=[];folder=root/str(i%2)/key
   for group,n in [('SOURCE',1),('HELDOUT',2)]:
    for pair in range(n):
     for role in ('UPDATE','PRESERVE'):
      q=dict(query_id=f'{key}_{group}_{pair}_{role}',question=f'{key} {group} {pair} {role}',reference=role,relation_role=role,eval_group=group,pair_id=str(pair));rows.append(q)
      for arm in ('BASE','JOINT','JOINT_EXPR'):
       answer='wrong' if arm=='BASE' and role=='UPDATE' and i<3 else role
       w(folder/'outputs'/arm/(q['query_id']+'.json'),dict(binding=q,raw_answer=answer,raw_token_ids=[1,2],EOS=True,at_cap=False))
   tasks.append(dict(key=key,eval_rows=rows,train_rows=rows[:2]))
   w(folder/'LIFECYCLE.json',dict(status='RETIRED'))
   for arm in ('JOINT','JOINT_EXPR'):w(folder/arm/'TRAIN_COMPLETE.json',dict(steps=160,parameters=7168,save_load_exact=True))
  w(root/'DATA_ADMISSION.json',dict(tasks=tasks))
  for part in ('0','1'):
   w(root/part/'EXIT.json',dict(code=0));w(root/part/'MODEL_COMPLETE.json',dict(status='PASS'))
  assert c.validate(root)['consumers']==108;c.blind(root)
  vv={x['code']:dict(label='INCORRECT' if x['answer']=='wrong' else 'CORRECT',reason='synthetic fixture') for x in c.read(root/'BLIND_PACKET.json')};w(root/'BLIND_VERDICTS.json',vv)
  d=c.summarize(root);assert d['eligible_pairs']==3
  for arm in ('JOINT','JOINT_EXPR'):
   assert d['arms'][arm]['strata']['SOURCE_all']['both_correct']==6
   assert d['arms'][arm]['strata']['HELDOUT_source_qualified']['both_correct']==6
  vv.pop(next(iter(vv)));w(root/'BLIND_VERDICTS.json',vv)
  try:c.summarize(root)
  except AssertionError:pass
  else:raise AssertionError('Missing verdict accepted')
  w(root/'1/EXIT.json',dict(code=1))
  try:c.validate(root)
  except AssertionError:pass
  else:raise AssertionError('Failed shard accepted')
 return dict(status='PASS',synthetic_consumers=108,partitions=2,source_only_qualification=True,missing_verdict_rejected=True,failed_shard_rejected=True)
if __name__=='__main__':print(json.dumps(main()))
