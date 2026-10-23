"""Re-score retained V22 BalancEdit under frozen V62 full-answer contract; no GPU."""
import json,random,collections
from pathlib import Path

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def collect(private):
 old=private/'m3bench_algorithm_v22_20261020';v79=private/'method_decision_v79_20261023/completed';root=private/'bank_compare_v80_20261023';tasks=read(v79/'DATA_ADMISSION.json')['tasks'];oldtasks=read(old/'COMPACT_DATA_ADMISSION.json')['tasks'];out=[];missing=[]
 for t,ot in zip(tasks,oldtasks):
  assert t['key']==ot['key'] and t['order']==ot['order']
  assert all(t['source'][k]==ot['source'][k] for k in ['question','reference','image_sha256'])
  oq={q['query_id']:q for q in ot['eval_rows']}
  for q in t['eval_rows']:
   if q['query_id'] not in oq:missing.append(dict(owner=t['order'],query=q['query_id'],roles=q['roles']));continue
   assert all(q[k]==oq[q['query_id']][k] for k in ['question','reference','image_sha256','roles'])
   be=read(root/'historical'/str((t['order']-1)%5)/t['key']/'outputs/BALANCEDIT'/(q['query_id']+'.json'))
   tt=read(v79/'R_V69'/t['key']/'outputs'/(q['query_id']+'.json'))
   base=read(private/'source_stop_v62_20261022/completed/base'/(q['query_id']+'.json'))
   for a,o in [('TT_V69',tt),('BALANCEDIT',be),('BASE',base)]:
    assert all(o['binding'][k]==q[k] for k in ['question','reference','image_sha256'])
    assert o['binding']['prompt_ids']==tt['binding']['prompt_ids'] and o['binding']['generation']==tt['binding']['generation']
    out.append(dict(arm=a,owner=t['order'],query=q['query_id'],roles=q['roles'],source_reference=t['source']['reference'],output=o))
 assert len(out)==648 and len(missing)==3 and all(x['roles']==['T1G'] for x in missing)
 return out,missing

def prepare(private):
 root=private/'bank_compare_v80_20261023/unified';assert not (root/'BLIND_PACKET.json').exists()
 rr,missing=collect(private);old={};oldscope={}
 prior=private/'method_decision_v79_20261023/completed';vv=read(prior/'BLIND_VERDICTS.json')
 for x in read(prior/'BLIND_PACKET.json'):old[x['question'],x['reference'],x['answer']]=vv[x['id']]
 # All V62-V78 unique judgments may be reused only for exact complete triples.
 for p in private.glob('*/completed/BLIND_PACKET.json'):
  if not (p.parent/'BLIND_VERDICTS.json').exists():continue
  packet=read(p)
  if not isinstance(packet,list):continue
  v=read(p.parent/'BLIND_VERDICTS.json')
  if not all(all(k in x for k in ['id','question','reference','answer']) for x in packet):continue
  # Restrict to declared V62-V79 scoring lineage, not older incompatible contracts.
  allowed=['source_stop_v62','source_view_v64','answer_writer_v65','source_init_v66','gradient_guard_v67','swap_anchor_v68','output_init_v69','full_gradient_init_v70','frozen_tail_v71','core_metric_v72','source_surprisal_v73','source_span_v74','worst_view_v75','complement_kl_v76','fixed_input_v77','label_smooth_v78','method_decision_v79']
  if not any(p.parent.parent.name.startswith(a+'_') for a in allowed):continue
  for x in packet:
   k=x['question'],x['reference'],x['answer'];z=v[x['id']]
   if k in old:assert old[k]['label']==z['label']
   old[k]=z
 sig=lambda x:(x['output']['binding']['question'],x['output']['binding']['reference'],x['output']['raw_answer'])
 keys=sorted({sig(x) for x in rr});random.Random(800023).shuffle(keys);ids={k:f'U{i:04}' for i,k in enumerate(keys)}
 packet=[dict(id=ids[k],question=k[0],reference=k[1],answer=k[2]) for k in keys];reuse={ids[k]:dict(old[k],provenance='EXACT_FROZEN_V62_V79_TRIPLE') for k in keys if k in old};review=[]
 for x in packet:
  if x['id'] in reuse:continue
  if x['answer'].strip().casefold()==x['reference'].strip().casefold():reuse[x['id']]=dict(label='CORRECT',reason='Complete reference exact after case/outer whitespace normalization.')
  elif not x['answer'].strip():reuse[x['id']]=dict(label='ABSTAIN',reason='Empty full answer.')
  else:review.append(x)
 write(root/'BLIND_PACKET.json',packet);write(root/'REUSED_VERDICTS.json',reuse);write(root/'BLIND_REVIEW_PACKET.json',review)
 write(root/'BLIND_MAPPING.json',[dict(arm=x['arm'],owner=x['owner'],query=x['query'],id=ids[sig(x)]) for x in rr]);write(root/'COVERAGE.json',dict(consumers=len(rr),common_per_arm=216,current_TT_only=missing,unique_triples=len(keys),reused_or_exact=len(reuse),new_manual=len(review),seed=800023,prior_V22_labels_not_used=True))
 return read(root/'COVERAGE.json')
if __name__=='__main__':print(json.dumps(prepare(Path('.private')),indent=2))

def scope_packet(private):
 root=private/'bank_compare_v80_20261023/unified';prior=private/'method_decision_v79_20261023/completed';tasks={t['order']:t for t in read(prior/'DATA_ADMISSION.json')['tasks']};v=read(prior/'SCOPE_VERDICTS.json');old={}
 for m in read(prior/'SCOPE_MAPPING.json'):
  t=tasks[m['owner']];a=m['arm'];p=(prior/a/t['key']) if a in ['R_V69','L_VIEW_LORA'] else (private/'source_view_v64_20261022/completed/0'/t['key']/a)
  o=read(p/'outputs'/(m['query']+'.json'));k=(t['source']['reference'],o['binding']['question'],o['binding']['reference'],o['raw_answer']);val=v[m['id']]
  if k in old:assert old[k]['unsupported_source_assertion']==val['unsupported_source_assertion']
  old[k]=val
 rr,_=collect(private);rr=[x for x in rr if x['arm']!='BASE' and set(x['roles']) & {'T1L','T2L'}]
 sig=lambda x:(x['source_reference'],x['output']['binding']['question'],x['output']['binding']['reference'],x['output']['raw_answer'])
 keys=sorted({sig(x) for x in rr});random.Random(800023).shuffle(keys);ids={k:f'S{i:03}' for i,k in enumerate(keys)}
 write(root/'SCOPE_REUSED.json',{ids[k]:old[k] for k in keys if k in old});write(root/'SCOPE_REVIEW.json',[dict(id=ids[k],source_reference=k[0],question=k[1],reference=k[2],answer=k[3]) for k in keys if k not in old]);write(root/'SCOPE_MAPPING.json',[dict(arm=x['arm'],owner=x['owner'],query=x['query'],id=ids[sig(x)]) for x in rr]);return dict(consumers=len(rr),unique=len(keys),new=sum(k not in old for k in keys))

def summarize(private):
 root=private/'bank_compare_v80_20261023/unified';assert read(root/'VERDICTS_FROZEN.json')['status']==read(root/'SCOPE_FROZEN.json')['status']=='FROZEN'
 rr,missing=collect(private);v=read(root/'BLIND_VERDICTS.json');lookup={(m['arm'],m['owner'],m['query']):v[m['id']]['label'] for m in read(root/'BLIND_MAPPING.json')};tables={};correct={};damage={};caps={};per={};scope={}
 for a in ['BASE','TT_V69','BALANCEDIT']:
  total=collections.defaultdict(collections.Counter);sets=collections.defaultdict(set);lost=set();capped=set();pe=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
  for x in rr:
   if x['arm']!=a:continue
   o,q=x['owner'],x['query'];l=lookup[a,o,q];b=lookup['BASE',o,q]
   if x['output']['at_cap']:capped.add((o,q))
   for role in x['roles']:
    total[role][l]+=1;pe[o][role][l]+=1
    if l=='CORRECT':sets[role].add((o,q))
    if role in ['T1L','T2L'] and b=='CORRECT' and l!='CORRECT':lost.add((o,q,role))
  tables[a]={role:dict(total=sum(c.values()),correct=c['CORRECT'],labels=dict(c)) for role,c in total.items()};correct[a]=sets;damage[a]=lost;caps[a]=len(capped);per[a]={o:{r:dict(total=sum(c.values()),correct=c['CORRECT'],labels=dict(c)) for r,c in d.items()} for o,d in pe.items()}
 sv=read(root/'SCOPE_VERDICTS.json')
 for a in ['TT_V69','BALANCEDIT']:scope[a]=sum(sv[m['id']]['unsupported_source_assertion'] for m in read(root/'SCOPE_MAPPING.json') if m['arm']==a)
 pairs={role:dict(TT_lost_vs_BE=len(correct['BALANCEDIT'][role]-correct['TT_V69'][role]),TT_gained_vs_BE=len(correct['TT_V69'][role]-correct['BALANCEDIT'][role])) for role in tables['TT_V69']}
 out=dict(status='UNIFIED_FULL_TEXT_SCORED',table=tables,per_edit=per,pairwise=pairs,Base_correct_locality_losses={a:len(v) for a,v in damage.items()},unsupported_source_assertions=scope,capped=caps,coverage=read(root/'COVERAGE.json'),model_calls=0,old_V22_labels_unchanged=True,independent_confirmation=False)
 write(root/'RESULTS.json',out);return out
