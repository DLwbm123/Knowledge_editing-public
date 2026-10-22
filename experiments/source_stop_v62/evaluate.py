"""Bound-output validation, fresh unified blind packet, and frozen paired decisions."""
import collections,json,random,math
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def collect(root):
 cfg=read(root/'DATA_ADMISSION.json');lock=read(root/'PROTOCOL_LOCK.json');rows=[];selections=[]
 for i,structure in enumerate(lock['structures']):
  ex=read(root/str(i)/'EXIT.json');done=read(root/str(i)/'MODEL_COMPLETE.json')
  assert ex['code']==0 and done['status']=='PASS' and ex['counts']==done['counts'] and done['counts']['updates']==1600
  assert read(root/str(i)/'BASE_REUSE_ADMISSION.json')['status']=='PASS'
  for t in cfg['tasks']:
   d=root/str(i)/t['key'];receipt=read(d/'COMPLETE.json');selection=read(d/'SELECTION_FROZEN.json')
   assert receipt['save_load_exact'] and receipt['outputs_per_state']==len(t['eval_rows'])
   passed=[c['step'] for c in selection['checks'] if c['source_pass']]
   assert selection['selected_step']==(min(passed) if passed else 160)
   assert all(c['step'] in lock['checks'] and c['trajectory_state_unchanged'] for c in selection['checks'])
   if i==0:assert receipt['curve_replay_exact'] and receipt['TT160_tokens_checked']==len(t['eval_rows'])
   selections.append(dict(structure=structure,owner=t['order'],selected_step=selection['selected_step'],fallback=selection['fallback'],train_seconds=selection['train_seconds'],check_seconds=selection['check_seconds'],selected_training_seconds=selection['selected_training_seconds'],selected_check_seconds=selection['selected_check_seconds'],check_generations=sum(c['counts']['generations'] for c in selection['checks']),same_state_outputs_reused=receipt['same_state_outputs_reused']))
   for state in lock['states']:
    for q in t['eval_rows']:
     o=read(d/state/'outputs'/(q['query_id']+'.json'));base=read(root/'base'/(q['query_id']+'.json'))
     if o['binding']!=base['binding']:
      alias=read(root/'PATH_ALIAS_ADMISSION.json');assert alias['status']=='PASS'
      assert any(p['query_id']==q['query_id'] and set(p['paths'])=={o['binding']['image_path'],base['binding']['image_path']} for p in alias['pairs'])
      assert {k:v for k,v in o['binding'].items() if k!='image_path'}=={k:v for k,v in base['binding'].items() if k!='image_path'}
     assert o['binding']['question']==q['question'] and o['binding']['reference']==q['reference'] and o['binding']['image_sha256']==q['image_sha256']
     rows.append(dict(arm=structure+'_'+state,owner=t['order'],query=q['query_id'],output=o))
 for q in read(root/'QUERY_ROWS.json'):
  rows.append(dict(arm='BASE',owner=None,query=q,output=read(root/'base'/(q+'.json'))))
 assert len(rows)==1059
 return cfg,rows,selections

def packet(root):
 assert not (root/'BLIND_PACKET.json').exists()
 cfg,rows,selections=collect(root)
 def sig(r):
  o=r['output'];return o['binding']['question'],o['binding']['reference'],o['raw_answer']
 triples=sorted({sig(r) for r in rows});random.Random(620022).shuffle(triples);keys={x:f'K{i:04}' for i,x in enumerate(triples)}
 full=[dict(id=keys[x],question=x[0],reference=x[1],answer=x[2]) for x in triples]
 write(root/'BLIND_PACKET.json',full)
 # Fresh reference equality / empty-answer rules only, never historical labels.
 exact={}
 for x in full:
  if not x['answer'].strip():exact[x['id']]=dict(label='ABSTAIN',reason='Empty answer under unified v1.')
  elif x['reference'].strip() and x['answer'].strip().casefold()==x['reference'].strip().casefold():exact[x['id']]=dict(label='CORRECT',reason='Complete answer equals the frozen reference after outer whitespace and case normalization; no additional claims.')
 write(root/'FRESH_EXACT_VERDICTS.json',exact)
 write(root/'BLIND_REVIEW_PACKET.json',[x for x in full if x['id'] not in exact])
 write(root/'BLIND_MAPPING.json',[dict(arm=r['arm'],owner=r['owner'],query=r['query'],blind_id=keys[sig(r)]) for r in rows])
 admission=dict(status='PASS',bound_outputs=len(rows),unique_blind=len(triples),prior_semantic_labels_reused=0,fresh_exact_or_empty=len(exact),manual_blind=len(full)-len(exact),Base_generations_reused=183,formal_output_consumers=876,EOS=sum(r['output']['EOS'] for r in rows),at_cap=sum(r['output']['at_cap'] for r in rows))
 write(root/'PACKET_ADMISSION.json',admission);write(root/'SELECTION_SUMMARY.json',selections);return admission

def summary(root):
 assert (root/'VERDICTS_FROZEN.json').exists()
 cfg,rows,selections=collect(root);mapping=read(root/'BLIND_MAPPING.json');v=read(root/'BLIND_VERDICTS.json')
 assert set(v)=={x['id'] for x in read(root/'BLIND_PACKET.json')}
 assert all(x['label'] in ['CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'] and x['reason'] for x in v.values())
 lookup={(x['arm'],x['owner'],x['query']):v[x['blind_id']]['label'] for x in mapping}
 arms=['BASE']+[s+'_'+a for s in read(root/'PROTOCOL_LOCK.json')['structures'] for a in ['FIXED160','SOURCE_STOP']]
 def label(a,owner,q):return lookup[(a,None if a=='BASE' else owner,q)]
 tables={};correct_sets={};damage_sets={};transitions={};per_edit={}
 for a in arms:
  total=collections.defaultdict(collections.Counter);correct=collections.defaultdict(set);damage=set();base_denom=collections.Counter();tr=collections.defaultdict(collections.Counter);edit={}
  for t in cfg['tasks']:
   e=collections.defaultdict(collections.Counter)
   for q in t['eval_rows']:
    l=label(a,t['order'],q['query_id']);b=label('BASE',t['order'],q['query_id'])
    for role in q['roles']:
     total[role][l]+=1;e[role][l]+=1;tr[role][b+'->'+l]+=1
     if l=='CORRECT':correct[role].add((t['order'],q['query_id']))
     if role in ['T1L','T2L'] and b=='CORRECT':
      base_denom[role]+=1
      if l!='CORRECT':damage.add((t['order'],q['query_id'],role))
   edit[t['order']]={k:dict(total=sum(x.values()),correct=x['CORRECT'],labels=dict(x)) for k,x in e.items()}
  tables[a]={k:dict(total=sum(x.values()),correct=x['CORRECT'],labels=dict(x)) for k,x in total.items()}
  tables[a]['Base_correct_locality']={k:dict(denominator=base_denom[k],lost=sum(x[2]==k for x in damage),fraction=sum(x[2]==k for x in damage)/base_denom[k] if base_denom[k] else None) for k in ['T1L','T2L']}
  correct_sets[a]=correct;damage_sets[a]=damage;transitions[a]={k:dict(x) for k,x in tr.items()};per_edit[a]=edit
 decisions={};pairwise={}
 for s in read(root/'PROTOCOL_LOCK.json')['structures']:
  a=s+'_FIXED160';b=s+'_SOURCE_STOP';loss0=correct_sets[a]['T0']-correct_sets[b]['T0'];loss1=correct_sets[a]['T1G']-correct_sets[b]['T1G'];delta=len(correct_sets[b]['T2G'])-len(correct_sets[a]['T2G']);positive=sum(per_edit[b][o]['T2G']['correct']>per_edit[a][o]['T2G']['correct'] for o in per_edit[a]);extra=damage_sets[b]-damage_sets[a]
  decisions[s]=dict(T0_correct_losses=len(loss0),T1G_correct_losses=len(loss1),T2G_net_gain=delta,net_positive_edits=positive,additional_Base_correct_locality_losses=len(extra),development_signal=not loss0 and not loss1 and delta>=2 and positive>=2 and not extra)
 for a,b in [('TT_FIXED160','TT_SOURCE_STOP'),('NORMALIZED_LORA_R1_FIXED160','NORMALIZED_LORA_R1_SOURCE_STOP'),('TT_FIXED160','NORMALIZED_LORA_R1_FIXED160'),('TT_SOURCE_STOP','NORMALIZED_LORA_R1_SOURCE_STOP')]:
  pairwise[a+'->'+b]={role:dict(lost=len(correct_sets[a][role]-correct_sets[b][role]),gained=len(correct_sets[b][role]-correct_sets[a][role])) for role in ['T0','T1G','T2G','T1L','T2L','REPHRASE_DIAGNOSTIC']}
 norms={}
 for a in arms[1:]:
  rs=[r['output']['generation_trace'] for r in rows if r['arm']==a]
  norms[a]=dict(mean_prefill_residual_RMS=sum(x[0]['active_residual_norm']/math.sqrt(max(1,x[0]['active_predictor_count'])*4096) for x in rs)/len(rs),max_decode_residual_norm=max((y['active_residual_norm'] for x in rs for y in x[1:]),default=0))
 exits=[read(root/str(i)/'EXIT.json') for i in range(2)]
 result=dict(status='TT_DEVELOPMENT_SIGNAL_REQUIRES_CONFIRMATION' if decisions['TT']['development_signal'] else 'TT_SOURCE_STOP_NOT_PROMOTED',unified_scoring_version='V62 v1; fresh blind labels; V61 immutable',table=tables,per_edit=per_edit,Base_transitions=transitions,pairwise=pairwise,decisions=decisions,selections=selections,residuals=norms,packet=read(root/'PACKET_ADMISSION.json'),costs=dict(GPU_process_hours=sum(e['seconds'] for e in exits)/3600,counts={k:sum(e['counts'][k] for e in exits) for k in exits[0]['counts']},actual_training_updates=3200,hypothetical_source_stop_updates=sum(x['selected_step'] for x in selections),source_check_generations=sum(x['check_generations'] for x in selections),same_state_output_reuses=sum(x['same_state_outputs_reused'] for x in selections),note='Actual experiment trains all to160; hypothetical stopping must include selection checks; no claimed observed experiment training savings'))
 write(root/'SUMMARY.json',result);return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['packet','summary']);p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(packet(a.root) if a.action=='packet' else summary(a.root),indent=2))
