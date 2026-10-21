"""Synthetic test of three-group consumers and missing-grade rejection."""
import json,tempfile
from pathlib import Path
import consumers as c

def main():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)
  def w(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
  tasks=[]
  for i in range(6):
   key=f'X{i}';rows=[]
   for group,n in [('SOURCE',1),('OLD_PARAPHRASE',2),('V29_REPLAY',2)]:
    for pair in range(n):
     for role in ('UPDATE','PRESERVE'):
      q=dict(query_id=f'{key}_{group}_{pair}_{role}',question=f'{key} {group} {pair} {role}',reference=role,relation_role=role,eval_group=group,pair_id=str(pair));rows.append(q)
      for arm in ('BASE','JOINT','JOINT_EXPR'):
       w(root/'1'/key/'outputs'/arm/(q['query_id']+'.json'),dict(binding=q,raw_answer=role,raw_token_ids=[1,2],EOS=True,at_cap=False))
   tasks.append(dict(key=key,eval_rows=rows,train_rows=rows[:2]))
   w(root/'1'/key/'LIFECYCLE.json',dict(status='RETIRED'));w(root/'1'/key/'REPLAY_ADMISSION.json',dict(consumers=12))
   for arm in ('JOINT','JOINT_EXPR'):w(root/'1'/key/arm/'TRAIN_COMPLETE.json',dict(steps=160,parameters=7168,save_load_exact=True,reconstruction160='PASS'))
  w(root/'DATA_ADMISSION.json',dict(tasks=tasks));w(root/'BASE_OPPORTUNITY.json',dict(pairs=[dict(case=f'X{i}',opportunity=i<3) for i in range(6)]));w(root/'1/EXIT.json',dict(code=0));w(root/'1/MODEL_COMPLETE.json',dict(status='PASS'))
  assert c.validate(root)['consumers']==180;assert c.blind(root)['unique']==60
  packet=c.read(root/'BLIND_PACKET.json');vv={x['code']:dict(label='CORRECT',reason='synthetic fixture') for x in packet};w(root/'BLIND_VERDICTS.json',vv)
  d=c.summarize(root)
  for arm in d['arms'].values():
   for g,n in [('SOURCE',6),('OLD_PARAPHRASE',12),('V29_REPLAY',12)]:
    assert arm['strata'][g+'_all']['both_correct']==n
    assert arm['strata'][g+'_source_qualified']['pair_n']==n//2
  vv.pop(next(iter(vv)));w(root/'BLIND_VERDICTS.json',vv)
  try:c.summarize(root)
  except AssertionError:pass
  else:raise AssertionError('Missing verdict accepted')
 return dict(status='PASS',synthetic_consumers=180,group_denominators=True,qualified_strata=True,missing_verdict_rejected=True)
if __name__=='__main__':print(json.dumps(main()))
