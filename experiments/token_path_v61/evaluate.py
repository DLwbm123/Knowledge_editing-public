"""Validate complete receipts, freeze blind packet, then summarize frozen labels."""
import json,random,collections,math,re,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def collect(root):
 cfg=read(root/'DATA_ADMISSION.json');arms=read(root/'PROTOCOL_LOCK.json')['arms'];rows=[]
 qmap=read(root/'QUERY_ROWS.json');assert len(qmap)==183
 for i,arm in enumerate(arms):
  ex=read(root/str(i)/'EXIT.json');m=read(root/str(i)/'MODEL_COMPLETE.json')
  assert ex['code']==0 and m['status']=='PASS'
  assert ex['counts']==m['counts'] and m['counts']['updates']==1600
  assert m['counts']['backwards']==1600+(4 if i==0 else 3)
  assert m['counts']['generations']==(402 if i==0 else 220)
  for t in cfg['tasks']:
   assert read(root/str(i)/t['key']/'COMPLETE.json')['save_load_exact']
   for r in t['eval_rows']:
    path=root/str(i)/t['key']/'outputs'/(r['query_id']+'.json');o=read(path);b=read(root/'base'/(r['query_id']+'.json'))
    if o['binding']!=b['binding']:
     alias=read(root/'PATH_ALIAS_ADMISSION.json');assert alias['status']=='PASS'
     assert any(v['query_id']==r['query_id'] and set(v['paths'])=={o['binding']['image_path'],b['binding']['image_path']} for v in alias['pairs'])
     assert {k:v for k,v in o['binding'].items() if k!='image_path'}=={k:v for k,v in b['binding'].items() if k!='image_path'}
    assert o['binding']['question']==r['question'] and o['binding']['reference']==r['reference'] and o['binding']['image_sha256']==r['image_sha256']
    rows.append(dict(arm=arm,owner=t['order'],query=r['query_id'],roles=r['roles'],source=t['source']['reference'],output=o))
 for q,r in qmap.items():
  o=read(root/'base'/(q+'.json'));assert o['binding']['question']==r['question'] and o['binding']['reference']==r['reference']
  rows.append(dict(arm='BASE',owner=None,query=q,roles=[],output=o))
 assert len(rows)==840
 return cfg,rows

def signature(row):
 o=row['output'];return (o['binding']['question'],o['binding']['reference'],o['raw_answer'])

def packet(root):
 assert not (root/'BLIND_PACKET.json').exists()
 cfg,rows=collect(root);unique=sorted({signature(r) for r in rows});random.Random(610022).shuffle(unique)
 keys={x:f'J{i:04}' for i,x in enumerate(unique)}
 full_packet=[dict(id=keys[x],question=x[0],reference=x[1],answer=x[2]) for x in unique]
 write(root/'BLIND_PACKET.json',full_packet)
 prior=read(root/'PRIOR_SEMANTIC_VERDICTS.json') if (root/'PRIOR_SEMANTIC_VERDICTS.json').exists() else []
 accepted={(x['question'],x['reference'],x['answer']):x['verdict'] for x in prior}
 reused={keys[x]:accepted[x] for x in unique if x in accepted}
 write(root/'EXACT_LABEL_REUSE.json',reused)
 write(root/'BLIND_NEW_PACKET.json',[x for x in full_packet if x['id'] not in reused])
 write(root/'BLIND_MAPPING.json',[dict(arm=r['arm'],owner=r['owner'],query=r['query'],blind_id=keys[signature(r)]) for r in rows])
 audit=dict(status='PASS',bound_outputs=len(rows),unique_blind=len(unique),exact_semantic_reuse=len(reused),new_blind=len(unique)-len(reused),counts=[read(root/str(i)/'EXIT.json') for i in range(3)],EOS=sum(bool(r['output']['EOS']) for r in rows),at_cap=sum(bool(r['output']['at_cap']) for r in rows))
 write(root/'PACKET_ADMISSION.json',audit);return {k:v for k,v in audit.items() if k!='counts'}

def norm(s):return re.sub(r'[^\w\u4e00-\u9fff]+','',s.casefold())
def summary(root):
 assert (root/'VERDICTS_FROZEN.json').exists()
 cfg,rows=collect(root);mapping=read(root/'BLIND_MAPPING.json');verdict=read(root/'BLIND_VERDICTS.json');packet=read(root/'BLIND_PACKET.json')
 assert set(verdict)=={v['id'] for v in packet}
 allowed={'CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'}
 assert all(v['label'] in allowed and v['reason'] for v in verdict.values())
 lookup={(v['arm'],v['owner'],v['query']):verdict[v['blind_id']]['label'] for v in mapping}
 output={(r['arm'],r['owner'],r['query']):r['output'] for r in rows}
 arms=read(root/'PROTOCOL_LOCK.json')['arms'];app={(r['owner'],r['query_id']):r['applicability'] for r in read(root/'APPLICABILITY_FROZEN.json')['rows']}
 table={};losses={};source_sets={};echo={};transitions={}
 def label(a,owner,q):return lookup[(a,None if a=='BASE' else owner,q)]
 for a in ['BASE']+arms:
  values=collections.defaultdict(collections.Counter);loss=[];base_n=collections.Counter();tr=collections.defaultdict(collections.Counter);src=set();echo_rows=[]
  for t in cfg['tasks']:
   for r in t['eval_rows']:
    q=r['query_id'];l=label(a,t['order'],q);b=label('BASE',t['order'],q)
    for role in r['roles']:
     values[role][l]+=1;tr[role][b+'->'+l]+=1
     if role in ['T1L','T2L'] and b=='CORRECT':
      base_n[role]+=1
      if l!='CORRECT':loss.append((t['order'],q,role))
    if 'T0' in r['roles'] and l=='CORRECT':src.add(t['order'])
    if app.get((t['order'],q))=='INAPPLICABLE':
     o=output[(a,None if a=='BASE' else t['order'],q)]
     is_echo=norm(o['raw_answer'])==norm(t['source']['reference']) and l=='INCORRECT'
     echo_rows.append(dict(owner=t['order'],query=q,incorrect_exact_source_echo=is_echo,label=l))
  table[a]={role:dict(total=sum(v.values()),correct=v['CORRECT'],labels=dict(v)) for role,v in values.items()}
  table[a]['Base_correct_locality']={role:dict(denominator=base_n[role],lost=sum(x[2]==role for x in loss),fraction=sum(x[2]==role for x in loss)/base_n[role] if base_n[role] else None) for role in ['T1L','T2L']}
  losses[a]=set(loss);source_sets[a]=src;echo[a]=dict(denominator=len(echo_rows),unique_inputs=len({r['query'] for r in echo_rows}),incorrect_exact_source_echo=sum(r['incorrect_exact_source_echo'] for r in echo_rows));transitions[a]={k:dict(v) for k,v in tr.items()}
 A=arms[0];decisions={}
 for a in arms[1:]:
  delta=sum(table[a][r]['correct']-table[A][r]['correct'] for r in ['T1G','T2G'])
  lost=sorted(source_sets[A]-source_sets[a]);extra=sorted(losses[a]-losses[A]);improved=delta>0 or echo[a]['incorrect_exact_source_echo']<echo[A]['incorrect_exact_source_echo']
  decisions[a]=dict(source_success_losses_vs_A=lost,generalization_correct_delta=delta,additional_Base_correct_locality_losses=len(extra),strict_improvement=improved,development_signal=not lost and delta>=0 and not extra and improved)
 norms={}
 for a in arms:
  vals=[];decode=[]
  for r in rows:
   if r['arm']!=a:continue
   trace=r['output']['generation_trace'];first=trace[0];n=first['active_predictor_count'];vals.append(first['active_residual_norm']/math.sqrt(max(1,n)*4096))
   decode.extend(t['active_residual_norm'] for t in trace[1:])
  norms[a]=dict(mean_prefill_residual_RMS=sum(vals)/len(vals),max_decode_residual_norm=max(decode,default=0.))
 exits=[read(root/str(i)/'EXIT.json') for i in range(3)]
 phase_cost={}
 for i,a in enumerate(arms):
  start=read(root/'BASE_COMPLETE.json') if i==0 else read(root/str(i)/'STAGE0_READY.json')
  finish=read(root/str(i)/'MODEL_COMPLETE.json')
  phase_cost[a]=dict(method_phase_seconds=finish['time']-start['time'],counts={k:exits[i]['counts'][k]-start['counts'][k] for k in exits[i]['counts']},note='Method phase excludes shared Base and admission; small coordination/preparation overhead included')
 result=dict(status='DEVELOPMENT_SIGNAL_REQUIRES_CONFIRMATION' if any(v['development_signal'] for v in decisions.values()) else 'NO_DEVELOPMENT_SIGNAL',table=table,transitions=transitions,source_echo=echo,decisions=decisions,residuals=norms,costs=dict(method_phases=phase_cost,shared_Base_charged_to_worker0=True,GPU_process_hours=sum(x['seconds'] for x in exits)/3600,counts={k:sum(x['counts'][k] for x in exits) for k in exits[0]['counts']}),packet=read(root/'PACKET_ADMISSION.json'))
 write(root/'SUMMARY.json',result);return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['packet','summary']);p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(packet(a.root) if a.action=='packet' else summary(a.root),indent=2,ensure_ascii=False))
