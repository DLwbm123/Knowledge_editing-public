"""Immutable V62-v1 exact-triple reuse and blinded V64 development evaluation."""
import json,random,collections
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def collect(root,parent):
 assert read(root/'EXIT.json')['code']==0 and read(root/'MAIN_COMPLETE.json')['counts']==read(root/'EXIT.json')['counts']
 assert read(root/'A_REPLAY_BARRIER.json')['status']=='PASS'
 tasks=read(root/'DATA_ADMISSION.json')['tasks'];rows=[]
 for arm in ['A_NATIVE','B_VIEW_CE']:
  for t in tasks:
   d=root/'0'/t['key']/arm;done=read(d/'COMPLETE.json');assert done['status']=='PASS' and done['save_load_exact'] and done['outputs']==len(t['eval_rows'])
   for q in t['eval_rows']:
    o=read(d/'outputs'/(q['query_id']+'.json'));old=read(parent/'0'/t['key']/'FIXED160/outputs'/(q['query_id']+'.json'))
    assert o['binding']==old['binding']
    if arm=='A_NATIVE':assert o['raw_token_ids']==old['raw_token_ids']
    rows.append(dict(arm=arm,owner=t['order'],query=q['query_id'],output=o))
 for q in read(parent/'QUERY_ROWS.json'):
  rows.append(dict(arm='BASE',owner=None,query=q,output=read(parent/'base'/(q+'.json'))))
 assert len(rows)==621;return tasks,rows

def packet(root,parent):
 assert not (root/'BLIND_PACKET.json').exists()
 tasks,rows=collect(root,parent)
 sig=lambda o:(o['binding']['question'],o['binding']['reference'],o['raw_answer'])
 oldv=read(parent/'BLIND_VERDICTS.json');old={ (x['question'],x['reference'],x['answer']):oldv[x['id']] for x in read(parent/'BLIND_PACKET.json')}
 triples=sorted({sig(r['output']) for r in rows});random.Random(640023).shuffle(triples);ids={x:f'V{i:04}' for i,x in enumerate(triples)}
 full=[dict(id=ids[x],question=x[0],reference=x[1],answer=x[2]) for x in triples];verdicts={};review=[];reuse=exact=0
 for x in full:
  k=(x['question'],x['reference'],x['answer'])
  if k in old:verdicts[x['id']]=dict(old[k],provenance='V62_V1_EXACT_TRIPLE');reuse+=1
  elif not x['answer'].strip():verdicts[x['id']]=dict(label='ABSTAIN',reason='Empty full answer',provenance='FROZEN_RULE');exact+=1
  elif x['answer'].strip().casefold()==x['reference'].strip().casefold():verdicts[x['id']]=dict(label='CORRECT',reason='Full answer equals reference after outer whitespace/case normalization',provenance='FROZEN_RULE');exact+=1
  else:review.append(x)
 write(root/'BLIND_PACKET.json',full);write(root/'BLIND_REVIEW_PACKET.json',review);write(root/'REUSED_VERDICTS.json',verdicts)
 write(root/'BLIND_MAPPING.json',[dict(arm=r['arm'],owner=r['owner'],query=r['query'],blind_id=ids[sig(r['output'])]) for r in rows])
 admission=dict(status='PASS',consumers=621,unique_triples=len(full),old_exact_triples_reused=reuse,new_exact_or_empty=exact,new_manual_blind=len(review),seed=640023)
 write(root/'PACKET_ADMISSION.json',admission);return admission


def summary(root,parent):
 assert read(root/'VERDICTS_FROZEN.json')['status']=='FROZEN' and read(root/'SCOPE_FROZEN.json')['status']=='FROZEN'
 tasks,rows=collect(root,parent);v=read(root/'BLIND_VERDICTS.json');mapping=read(root/'BLIND_MAPPING.json')
 lookup={(x['arm'],x['owner'],x['query']):v[x['blind_id']]['label'] for x in mapping}
 assert len(v)==len(read(root/'BLIND_PACKET.json'))
 def label(a,o,q):return lookup[(a,None if a=='BASE' else o,q)]
 tables={};sets={};edits={};damage={}
 for a in ['BASE','A_NATIVE','B_VIEW_CE']:
  totals=collections.defaultdict(collections.Counter);correct=collections.defaultdict(set);per={};lost=set()
  for t in tasks:
   counts=collections.defaultdict(collections.Counter)
   for q in t['eval_rows']:
    l=label(a,t['order'],q['query_id']);b=label('BASE',t['order'],q['query_id']);ident=(t['order'],q['query_id'])
    for role in q['roles']:
     totals[role][l]+=1;counts[role][l]+=1
     if l=='CORRECT':correct[role].add(ident)
     if role in ['T1L','T2L'] and b=='CORRECT' and l!='CORRECT':lost.add((*ident,role))
   per[t['order']]={role:dict(total=sum(c.values()),correct=c['CORRECT'],labels=dict(c)) for role,c in counts.items()}
  tables[a]={role:dict(total=sum(c.values()),correct=c['CORRECT'],labels=dict(c)) for role,c in totals.items()};sets[a]=correct;edits[a]=per;damage[a]=lost
 a,b='A_NATIVE','B_VIEW_CE';paired={role:dict(lost=len(sets[a][role]-sets[b][role]),gained=len(sets[b][role]-sets[a][role])) for role in tables[a]}
 caps={arm:{(x['owner'],x['query']) for x in rows if x['arm']==arm and x['output']['at_cap']} for arm in [a,b]}
 sv=read(root/'SCOPE_VERDICTS.json');sm=read(root/'SCOPE_MAPPING.json');unsupported={arm:{(x['owner'],x['query']) for x in sm if x['arm']==arm and sv[x['id']]['unsupported_source_assertion']} for arm in [a,b]}
 gate=dict(T2G_net=len(sets[b]['T2G'])-len(sets[a]['T2G']),positive_edits=sum(edits[b][o]['T2G']['correct']>edits[a][o]['T2G']['correct'] for o in edits[a]),preserved_correct={r:not(sets[a][r]-sets[b][r]) for r in ['T0','T1G','REPHRASE_DIAGNOSTIC']},locality_total_not_lower={r:len(sets[b][r])>=len(sets[a][r]) for r in ['T1L','T2L']},additional_Base_correct_losses=len(damage[b]-damage[a]),additional_unsupported_source_consumers=len(unsupported[b]-unsupported[a]),additional_capped_consumers=len(caps[b]-caps[a]))
 gate['pass']=gate['T2G_net']>=2 and gate['positive_edits']>=2 and all(gate['preserved_correct'].values()) and all(gate['locality_total_not_lower'].values()) and all(gate[k]==0 for k in ['additional_Base_correct_losses','additional_unsupported_source_consumers','additional_capped_consumers'])
 result=dict(status='DEV_SIGNAL' if gate['pass'] else 'SOURCE_VIEW_COVERAGE_NOT_PROMOTED',table=tables,per_edit=edits,paired=paired,gate=gate,unsupported_source_consumers={k:len(v) for k,v in unsupported.items()},capped_consumers={k:len(v) for k,v in caps.items()},packet=read(root/'PACKET_ADMISSION.json'),counts=read(root/'EXIT.json')['counts'],GPU_process_seconds=read(root/'EXIT.json')['seconds']+read(root/'ATTEMPT_HISTORY.json')['prior_exit']['seconds'],independent_confirmation=False)
 write(root/'SUMMARY.json',result);return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['packet','summary']);p.add_argument('root',type=Path);p.add_argument('parent',type=Path);a=p.parse_args()
 print(json.dumps(packet(a.root,a.parent) if a.action=='packet' else summary(a.root,a.parent),indent=2))
