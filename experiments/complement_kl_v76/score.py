"""V76 full-consumer evaluation with immutable V62/V64 reference labels."""
import json,random,collections
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rows(root,parent,base):
 assert read(root/'EXIT.json')['code']==0 and read(root/'REAL_ADMISSION.json')['status']=='PASS'
 tasks=read(root/'DATA_ADMISSION.json')['tasks'];out=[]
 for arm in ['A_NATIVE','B_VIEW_CE','N_COMPLEMENT_KL']:
  origin=root if arm=='N_COMPLEMENT_KL' else parent;folder='B_VIEW_CE' if arm=='N_COMPLEMENT_KL' else arm
  for t in tasks:
   d=origin/'0'/t['key']/folder;done=read(d/'COMPLETE.json');assert done['status']=='PASS' and done['save_load_exact'] and done['outputs']==len(t['eval_rows'])
   for q in t['eval_rows']:
    o=read(d/'outputs'/(q['query_id']+'.json'));ref=read(parent/'0'/t['key']/'A_NATIVE/outputs'/(q['query_id']+'.json'));assert o['binding']==ref['binding']
    out.append(dict(arm=arm,owner=t['order'],query=q['query_id'],output=o))
 for q in read(base/'QUERY_ROWS.json'):out.append(dict(arm='BASE',owner=None,query=q,output=read(base/'base'/(q+'.json'))))
 assert len(out)==840;return tasks,out

def packet(root,parent,base):
 assert not (root/'BLIND_PACKET.json').exists()
 tasks,rr=rows(root,parent,base);old={}
 for p in [base,parent,root.parent.parent/'answer_writer_v65_20261023/completed',root.parent.parent/'source_init_v66_20261023/completed',root.parent.parent/'gradient_guard_v67_20261023/completed',root.parent.parent/'swap_anchor_v68_20261023/completed',root.parent.parent/'output_init_v69_20261023/completed',root.parent.parent/'full_gradient_init_v70_20261023/completed',root.parent.parent/'frozen_tail_v71_20261023/completed',root.parent.parent/'core_metric_v72_20261023/completed',root.parent.parent/'source_surprisal_v73_20261023/completed',root.parent.parent/'source_span_v74_20261023/completed',root.parent.parent/'worst_view_v75_20261023/completed']:
  vv=read(p/'BLIND_VERDICTS.json')
  for x in read(p/'BLIND_PACKET.json'):
   k=(x['question'],x['reference'],x['answer'])
   if k in old:assert old[k]['label']==vv[x['id']]['label']
   old[k]=vv[x['id']]
 sig=lambda o:(o['binding']['question'],o['binding']['reference'],o['raw_answer'])
 triples=sorted({sig(x['output']) for x in rr});random.Random(760023).shuffle(triples);ids={k:f'Z{i:04}' for i,k in enumerate(triples)}
 packet=[dict(id=ids[k],question=k[0],reference=k[1],answer=k[2]) for k in triples];verdicts={};review=[];reuse=exact=0
 for x in packet:
  k=(x['question'],x['reference'],x['answer'])
  if k in old:verdicts[x['id']]=dict(old[k],provenance='FROZEN_V62_V64_V65_V66_V67_V68_V69_V70_V71_V72_V73_V74_V75_EXACT_TRIPLE');reuse+=1
  elif not x['answer'].strip():verdicts[x['id']]=dict(label='ABSTAIN',reason='Empty full answer');exact+=1
  elif x['answer'].strip().casefold()==x['reference'].strip().casefold():verdicts[x['id']]=dict(label='CORRECT',reason='Exact full reference after case/outer-whitespace normalization');exact+=1
  else:review.append(x)
 write(root/'BLIND_PACKET.json',packet);write(root/'REUSED_VERDICTS.json',verdicts);write(root/'BLIND_REVIEW_PACKET.json',review)
 write(root/'BLIND_MAPPING.json',[dict(arm=x['arm'],owner=x['owner'],query=x['query'],blind_id=ids[sig(x['output'])]) for x in rr])
 stats=dict(status='PASS',consumers=840,unique=len(packet),reused=reuse,new_exact=exact,new_manual=len(review),seed=760023);write(root/'PACKET_ADMISSION.json',stats);return stats

def summary(root,parent,base):
 assert read(root/'VERDICTS_FROZEN.json')['status']=='FROZEN' and read(root/'SCOPE_FROZEN.json')['status']=='FROZEN'
 tasks,rr=rows(root,parent,base);v=read(root/'BLIND_VERDICTS.json');lookup={(x['arm'],x['owner'],x['query']):v[x['blind_id']]['label'] for x in read(root/'BLIND_MAPPING.json')}
 def label(a,o,q):return lookup[(a,None if a=='BASE' else o,q)]
 tables={};sets={};edits={};damage={}
 for a in ['BASE','A_NATIVE','B_VIEW_CE','N_COMPLEMENT_KL']:
  total=collections.defaultdict(collections.Counter);correct=collections.defaultdict(set);per={};loss=set()
  for t in tasks:
   tc=collections.defaultdict(collections.Counter)
   for q in t['eval_rows']:
    l=label(a,t['order'],q['query_id']);b=label('BASE',t['order'],q['query_id']);ident=(t['order'],q['query_id'])
    for role in q['roles']:
     total[role][l]+=1;tc[role][l]+=1
     if l=='CORRECT':correct[role].add(ident)
     if role in ['T1L','T2L'] and b=='CORRECT' and l!='CORRECT':loss.add((*ident,role))
   per[t['order']]={role:dict(total=sum(z.values()),correct=z['CORRECT'],labels=dict(z)) for role,z in tc.items()}
  tables[a]={role:dict(total=sum(z.values()),correct=z['CORRECT'],labels=dict(z)) for role,z in total.items()};sets[a]=correct;edits[a]=per;damage[a]=loss
 caps={a:{(x['owner'],x['query']) for x in rr if x['arm']==a and x['output']['at_cap']} for a in ['A_NATIVE','B_VIEW_CE','N_COMPLEMENT_KL']}
 sv=read(root/'SCOPE_VERDICTS.json');sm=read(root/'SCOPE_MAPPING.json');u={a:{(x['owner'],x['query']) for x in sm if x['arm']==a and sv[x['id']]['unsupported_source_assertion']} for a in caps}
 assert all(sum(x['arm']==a for x in sm)==89 for a in caps)
 a,b,c='A_NATIVE','B_VIEW_CE','N_COMPLEMENT_KL';pairs={ref:{role:dict(lost=len(sets[ref][role]-sets[c][role]),gained=len(sets[c][role]-sets[ref][role])) for role in tables[ref]} for ref in [a,b]}
 g=dict(T2G_net_vs_A=len(sets[c]['T2G'])-len(sets[a]['T2G']),positive_edits_vs_A=sum(edits[c][o]['T2G']['correct']>edits[a][o]['T2G']['correct'] for o in edits[a]),B_T2G_correct_losses=len(sets[b]['T2G']-sets[c]['T2G']),preserved={ref:{role:not(sets[ref][role]-sets[c][role]) for role in ['T0','T1G','REPHRASE_DIAGNOSTIC']} for ref in [a,b]},locality_not_lower={ref:{role:len(sets[c][role])>=len(sets[ref][role]) for role in ['T1L','T2L']} for ref in [a,b]},extra_Base_correct_losses={ref:len(damage[c]-damage[ref]) for ref in [a,b]},extra_unsupported_vs_A=len(u[c]-u[a]),extra_caps={ref:len(caps[c]-caps[ref]) for ref in [a,b]})
 g['pass']=g['T2G_net_vs_A']>=2 and g['positive_edits_vs_A']>=2 and g['B_T2G_correct_losses']==0 and all(all(z.values()) for z in g['preserved'].values()) and all(all(z.values()) for z in g['locality_not_lower'].values()) and not any(g['extra_Base_correct_losses'].values()) and not g['extra_unsupported_vs_A'] and not any(g['extra_caps'].values())
 out=dict(status='DEV_SIGNAL' if g['pass'] else 'COMPLEMENT_KL_NOT_PROMOTED',table=tables,per_edit=edits,pairwise=pairs,gate=g,unsupported={k:len(x) for k,x in u.items()},capped={k:len(x) for k,x in caps.items()},packet=read(root/'PACKET_ADMISSION.json'),counts=read(root/'EXIT.json')['counts'],GPU_process_seconds=read(root/'EXIT.json')['seconds'],independent_confirmation=False)
 write(root/'SUMMARY.json',out);return out
