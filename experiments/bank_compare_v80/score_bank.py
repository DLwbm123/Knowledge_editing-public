"""Full-answer, exact-binding bank scoring; private raw material stays private."""
import collections,json,random
from pathlib import Path
from unify import read,write
P=Path('.private'); D=P/'bank_compare_v80_20261023'; R=D/'completed'; S=D/'bank_scoring'

def collect():
 tasks=read(R/'DATA_ADMISSION.json')['tasks'];rows=[]
 for n in (1,5,10):
  for arm in ('TT_V69','BALANCEDIT'):
   root=R/'bank'/arm/str(n);assert read(root/'COMPLETE.json')['status']=='PASS'
   cs=read(root/'CONSUMERS.json'); des={(x['owner'],x['query']):x for x in cs if x['condition']=='DESIGNATED'};nat={x['query']:x for x in cs if x['condition']=='NATURAL'}
   assert len(des)=={1:50,5:114,10:219}[n] and len(nat)=={1:50,5:114,10:183}[n]
   for t in tasks[:n]:
    for q in t['eval_rows']:
     query=q['query_id'];base=read(P/'source_stop_v62_20261022/completed/base'/(query+'.json'))
     choices=[('DESIGNATED',arm,read(R/des[t['key'],query]['output']),t['key']),('NATURAL',arm,read(R/nat[query]['output']),nat[query]['selected'])]
     if arm=='TT_V69':choices += [('NATURAL','RETRIEVAL',read(root/'retrieval'/(query+'.json')),nat[query]['selected']),('NATURAL','BASE',base,None)]
     else:
      assert read(root/'retrieval'/(query+'.json'))==read(R/'bank/TT_V69'/str(n)/'retrieval'/(query+'.json'))
     for condition,a,o,selected in choices:
      assert all(o['binding'][k]==q[k] for k in ['question','reference','image_sha256'])
      rows.append(dict(prefix=n,condition=condition,arm=a,owner=t['order'],query=query,roles=q['roles'],selected=selected,owner_key=t['key'],source_reference=t['source']['reference'],question=q['question'],reference=q['reference'],answer=o['raw_answer']))
 return rows

def prepare():
 assert not (S/'BLIND_PACKET.json').exists()
 rows=collect();old={}
 allowed=['source_stop_v62','source_view_v64','answer_writer_v65','source_init_v66','gradient_guard_v67','swap_anchor_v68','output_init_v69','full_gradient_init_v70','frozen_tail_v71','core_metric_v72','source_surprisal_v73','source_span_v74','worst_view_v75','complement_kl_v76','fixed_input_v77','label_smooth_v78','method_decision_v79']
 paths=[p for p in P.glob('*/completed/BLIND_PACKET.json') if any(p.parent.parent.name.startswith(x+'_') for x in allowed)]+[D/'unified/BLIND_PACKET.json']
 for p in paths:
  v=read(p.parent/'BLIND_VERDICTS.json')
  for x in read(p):
   k=tuple(x[a] for a in ('question','reference','answer'));z=v[x['id']]
   if k in old:assert old[k]['label']==z['label']
   old[k]=z
 sig=lambda x:tuple(x[a] for a in ('question','reference','answer'))
 keys=sorted({sig(x) for x in rows});random.Random(800024).shuffle(keys);ids={k:f'B{i:04}' for i,k in enumerate(keys)};packet=[dict(id=ids[k],question=k[0],reference=k[1],answer=k[2]) for k in keys];reuse={}
 for k in keys:
  if k in old:reuse[ids[k]]=dict(old[k],provenance='EXACT_FROZEN_V62_V80_TRIPLE')
  elif k[2].strip().casefold()==k[1].strip().casefold():reuse[ids[k]]=dict(label='CORRECT',reason='Complete reference exact after case and outer whitespace normalization.')
  elif not k[2].strip():reuse[ids[k]]=dict(label='ABSTAIN',reason='Empty full answer.')
 write(S/'BLIND_PACKET.json',packet);write(S/'REUSED_VERDICTS.json',reuse);write(S/'BLIND_REVIEW_PACKET.json',[x for x in packet if x['id'] not in reuse]);write(S/'MAPPING.json',[dict(x,id=ids[sig(x)]) for x in rows]);write(S/'COVERAGE.json',dict(consumers=len(rows),unique_triples=len(keys),reused_or_exact=len(reuse),new_manual=len(keys)-len(reuse),seed=800024))
 return read(S/'COVERAGE.json')
if __name__=='__main__':print(json.dumps(prepare(),indent=2))

def prepare_scope():
 from unify import collect as unified_collect
 assert read(S/'VERDICTS_FROZEN.json')['status']=='FROZEN'
 rr,_=unified_collect(P);lookup={(x['arm'],x['owner'],x['query']):x for x in rr};old={};u=D/'unified';v=read(u/'SCOPE_VERDICTS.json')
 for m in read(u/'SCOPE_MAPPING.json'):
  x=lookup[m['arm'],m['owner'],m['query']];o=x['output'];k=(x['source_reference'],o['binding']['question'],o['binding']['reference'],o['raw_answer']);old[k]=v[m['id']]
 rows=[x for x in collect() if set(x['roles'])&{'T1L','T2L'} and x['arm']!='BASE'];sig=lambda x:tuple(x[k] for k in ('source_reference','question','reference','answer'));keys=sorted({sig(x) for x in rows});random.Random(800025).shuffle(keys);ids={k:f'S{i:03}' for i,k in enumerate(keys)}
 write(S/'SCOPE_PACKET.json',[dict(id=ids[k],source_reference=k[0],question=k[1],reference=k[2],answer=k[3]) for k in keys]);write(S/'SCOPE_REUSED.json',{ids[k]:old[k] for k in keys if k in old});write(S/'SCOPE_REVIEW.json',[dict(id=ids[k],source_reference=k[0],question=k[1],reference=k[2],answer=k[3]) for k in keys if k not in old]);write(S/'SCOPE_MAPPING.json',[dict(prefix=x['prefix'],condition=x['condition'],arm=x['arm'],owner=x['owner'],query=x['query'],id=ids[sig(x)]) for x in rows]);return dict(consumers=len(rows),unique=len(keys),new=sum(k not in old for k in keys))

def summarize():
 assert read(S/'VERDICTS_FROZEN.json')['status']==read(S/'SCOPE_FROZEN.json')['status']=='FROZEN'
 rows=read(S/'MAPPING.json');v=read(S/'BLIND_VERDICTS.json');assert set(v)=={x['id'] for x in read(S/'BLIND_PACKET.json')};sv=read(S/'SCOPE_VERDICTS.json');sc={(x['prefix'],x['condition'],x['arm'],x['owner'],x['query']):sv[x['id']]['unsupported_source_assertion'] for x in read(S/'SCOPE_MAPPING.json')}
 panels=[];sets={};labs={};rlookup={}
 for n in (1,5,10):
  for condition,arm in [('DESIGNATED','TT_V69'),('DESIGNATED','BALANCEDIT')]+[('NATURAL',a) for a in ('TT_V69','BALANCEDIT','RETRIEVAL','BASE')]:
   rr=[x for x in rows if (x['prefix'],x['condition'],x['arm'])==(n,condition,arm)];counts=collections.defaultdict(collections.Counter);per=collections.defaultdict(lambda:collections.defaultdict(collections.Counter));cs=collections.defaultdict(set);labels={};base={(x['owner'],x['query']):v[x['id']]['label'] for x in rows if x['prefix']==n and x['arm']=='BASE'};loss=[];scope=0;routes=collections.Counter();unique={}
   for x in rr:
    l=v[x['id']]['label'];k=x['owner'],x['query'];labels[k]=l;rlookup[n,condition,arm,k]=x;unique[x['query']]=l
    if condition=='NATURAL':routes['Base_fallback' if x['selected'] is None else ('same_owner' if x['selected']==x['owner_key'] else 'different_owner')]+=1
    if set(x['roles'])&{'T1L','T2L'}:
     if base[k]=='CORRECT' and l!='CORRECT':loss.append(k)
     if arm!='BASE':scope+=sc[n,condition,arm,*k]
    for role in x['roles']:
     counts[role][l]+=1;per[x['owner']][role][l]+=1
     if l=='CORRECT':cs[role].add(k)
   tables=lambda d:{r:dict(total=sum(c.values()),correct=c['CORRECT'],labels=dict(c)) for r,c in d.items()}
   panels.append(dict(prefix=n,condition=condition,arm=arm,table=tables(counts),per_edit={o:tables(d) for o,d in per.items()},Base_correct_locality_losses=len(loss),source_unsupported_assertions=scope,scope_denominator=sum(bool(set(x['roles'])&{'T1L','T2L'}) for x in rr),route_identity_consumers=dict(routes),unique_query_labels=dict(collections.Counter(unique.values())),unique_queries=len(unique)))
   sets[n,condition,arm]=cs;labs[n,condition,arm]=labels
 retention=[]
 for a in ('TT_V69','BALANCEDIT','RETRIEVAL'):
  for before,after in ((1,5),(1,10),(5,10)):
   for condition in ('NATURAL',):
    for role,old in sets[before,condition,a].items():
     new=sets[after,condition,a][role];lost=old-new;details=[]
     for k in sorted(lost):
      x=rlookup[before,condition,a,k];y=rlookup[after,condition,a,k];details.append(dict(owner=k[0],query=k[1],before_selected=x['selected'],after_selected=y['selected'],route_changed=x['selected']!=y['selected'],after_label=labs[after,condition,a][k]))
     retention.append(dict(arm=a,before=before,after=after,role=role,old_correct=len(old),retained=len(old&new),lost=len(lost),lost_details=details))
 pairs=[]
 for n in (1,5,10):
  for condition in ('DESIGNATED','NATURAL'):
   for role in sets[n,condition,'TT_V69']:
    tt=sets[n,condition,'TT_V69'][role];be=sets[n,condition,'BALANCEDIT'][role];pairs.append(dict(prefix=n,condition=condition,role=role,TT_only=len(tt-be),BE_only=len(be-tt),both=len(tt&be)))
 cost=read(R/'BANK_COST.json');out=dict(status='ALL_BANK_CONSUMERS_SCORED',panels=panels,retention=retention,pairwise=pairs,coverage=read(S/'COVERAGE.json'),scope_coverage=dict(consumers=len(sc),unique=len(sv),new_full_text=len(read(S/'SCOPE_REVIEW.json'))),cost=cost,execution=read(R/'EXIT.json'),independent_confirmation=False,LoRA_new=False,old_V69_joint_gate='FAIL_UNCHANGED',source_scope_is_not_clinical_false_positive=True)
 write(S/'RESULTS.json',out);return out
