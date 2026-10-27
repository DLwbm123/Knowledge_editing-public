"""Validate fixed execution, reuse exact frozen labels, then grade only new blinded answers."""
import json,random
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def records(root):
 cfg=read(root/'DATA_ADMISSION.json');out=[]
 for row in cfg['eval_rows']:out.append(dict(arm='BASE',query_id=row['query_id'],row=row,output=cfg['Base_outputs'][row['query_id']]))
 for part,arm in enumerate(['SOURCE_ONLY','JOINT_CE']):
  end=read(root/str(part)/'EXIT.json');done=read(root/str(part)/'MODEL_COMPLETE.json');c=end['counts']
  assert end['code']==0 and done['status']=='PASS' and done['save_load_exact']
  assert (c['updates'],c['backwards'],c['admission_backwards'],c['generations'])==(160,326,6,18)
  for row in cfg['eval_rows']:
   x=read(root/str(part)/'outputs'/(row['query_id']+'.json'));assert x['binding']==cfg['Base_outputs'][row['query_id']]['binding']
   out.append(dict(arm=arm,query_id=row['query_id'],row=row,output=x))
 assert len(out)==51
 return cfg,out

def packet(root):
 assert not (root/'BLIND_PACKET.json').exists()
 cfg,items=records(root);new={};reuse=[]
 for x in items:
  q=x['query_id'];o=x['output'];b=cfg['Base_outputs'][q]
  if o['raw_answer']==b['raw_answer'] and o['raw_token_ids']==b['raw_token_ids']:
   reuse.append(dict(arm=x['arm'],query_id=q,**{k:v for k,v in cfg['Base_labels'][q].items() if k!='query_id'}));continue
  sig=json.dumps([x['row']['question'],x['row']['reference'],o['raw_answer'],o['raw_token_ids']]);new.setdefault(sig,[]).append(x)
 keys=list(new);random.Random(590022).shuffle(keys);packet=[];mapping={}
 for i,k in enumerate(keys):
  x=new[k][0];identifier=f'B{i+1:03d}';packet.append(dict(id=identifier,question=x['row']['question'],reference=x['row']['reference'],answer=x['output']['raw_answer']));mapping[identifier]=[dict(arm=y['arm'],query_id=y['query_id']) for y in new[k]]
 write(root/'EXACT_LABEL_REUSE.json',reuse);write(root/'BLIND_PACKET.json',packet);write(root/'BLIND_MAPPING.json',mapping)
 return dict(bindings=51,reused=len(reuse),new_unique=len(packet))

def summary(root):
 cfg,items=records(root);assert (root/'VERDICTS_FROZEN.json').exists();verdicts=read(root/'BLIND_VERDICTS.json');mapping=read(root/'BLIND_MAPPING.json');assert set(verdicts)==set(mapping)
 labels={(x['arm'],x['query_id']):x['label'] for x in read(root/'EXACT_LABEL_REUSE.json')}
 for key,targets in mapping.items():
  assert verdicts[key]['label'] in ['CORRECT','INCORRECT','UNCERTAIN','ABSTAIN']
  for x in targets:labels[x['arm'],x['query_id']]=verdicts[key]['label']
 assert len(labels)==51
 correct=lambda a,q:labels[a,q]=='CORRECT'
 groups=dict(train=['S00_C','S00_A'],heldout=['S00_B','S00_D','S00_E'],clinical=[x['query_id'] for x in cfg['eval_rows'] if x['role']=='CLINICAL_RETENTION'],annotated=[x['query_id'] for x in cfg['eval_rows'] if x['role']=='ANNOTATED_ANATOMY'])
 table={a:{g:dict(correct=sum(correct(a,q) for q in ids),total=len(ids)) for g,ids in groups.items()} for a in ['BASE','SOURCE_ONLY','JOINT_CE']}
 new_losses=[q for q in groups['clinical']+groups['annotated'] if correct('SOURCE_ONLY',q) and not correct('JOINT_CE',q)]
 protected=all(correct('JOINT_CE',q) for q in cfg['clinical_protection']);held_improvements=[q for q in groups['heldout'] if correct('JOINT_CE',q) and not correct('SOURCE_ONLY',q)]
 passed=table['JOINT_CE']['train']['correct']==2 and table['JOINT_CE']['heldout']['correct']==3 and bool(held_improvements) and protected and not new_losses
 return dict(status='DEVELOPMENT_GATE_PASS' if passed else 'DEVELOPMENT_GATE_FAILED',table=table,clinical_BaseC_retained=protected,clinical_BaseC_denominator=len(cfg['clinical_protection']),retention_control_losses=len(new_losses),heldout_improvements=len(held_improvements),EOS=sum(x['output']['EOS'] for x in items),at_cap=sum(x['output']['at_cap'] for x in items),bindings=51,independent_confirmation=False,annotation_preserved=True,patient_independence='UNKNOWN')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['packet','summary']);p.add_argument('root',type=Path);a=p.parse_args();result=packet(a.root) if a.action=='packet' else summary(a.root);print(json.dumps(result,indent=2))
