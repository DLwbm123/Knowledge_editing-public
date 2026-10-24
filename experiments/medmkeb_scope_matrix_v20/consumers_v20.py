"""Complete capability, blinded labels, natural output reuse and finite lifecycle."""
import json,random,collections
from pathlib import Path
from route_v20 import validate_matrix,calibrate,scores,choose
LABELS={'CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def collect(root):
 root=Path(root);c=read(root/'MATRIX_CONFIG.json');records=[]
 for i,t in enumerate(c['tasks']):
  p=root/'source'/str(i);assert read(p/'COMPLETE.json')['status']=='SOURCE_AND_ALL_MATRIX_MODEL_CONSUMERS_COMPLETE' and read(p/'EXIT.json')['code']==0
  assert read(p/'TRAIN_COMPLETE.json')['steps']==160
  for row in c['rows']:
   out=p/'outputs'/(row['id']+'.json');d=read(out);assert d['expert']==t['key'] and d['query']==row['id'];records.append(dict(expert=t['key'],query=row['id'],path=str(out)))
 validate_matrix(records,[t['key'] for t in c['tasks']],[q['id'] for q in c['rows']]);return c,records

def blind(root):
 root=Path(root);c,recs=collect(root);basever=read(root/'BASE_VERDICTS.json');unique={};aliases={}
 for row in c['rows']:
  q=row['id'];out=read(c['Base_outputs'][q]);k=(q,out['raw_answer']);unique[k]=dict(query=q,question=row['question'],reference=row['reference'],answer=out['raw_answer'],label=basever[q]);aliases['BASE|'+q]=k
 for r in recs:
  out=read(r['path']);q=r['query'];row=next(x for x in c['rows'] if x['id']==q);k=(q,out['raw_answer']);unique.setdefault(k,dict(query=q,question=row['question'],reference=row['reference'],answer=out['raw_answer']));aliases[r['expert']+'|'+q]=k
 order=list(unique);random.Random(20261020).shuffle(order);codes={k:'B%04d'%i for i,k in enumerate(order,1)};materials={codes[k]:{f:v for f,v in unique[k].items() if f not in ('query','label')} for k in order};ver={codes[k]:unique[k]['label'] for k in order if 'label' in unique[k]}
 write(root/'BLIND_MATERIALS.json',materials);write(root/'BLIND_ALIASES.json',{a:codes[k] for a,k in aliases.items()});write(root/'BLIND_REUSED_VERDICTS.json',ver);return dict(unique=len(order),reused=len(ver),new=len(order)-len(ver))

def route(root):
 import torch
 root=Path(root);c,recs=collect(root);keys={};anchors={}
 for i in range(2):
  p=root/'features'/str(i);assert read(p/'EXIT.json')['code']==0 and read(p/'COMPLETE.json')['status']=='KEY_CONSUMERS_COMPLETE';d=torch.load(p/'KEYS_FINAL.pt',map_location='cpu',weights_only=True);keys.update(d['keys']);anchors.update(d['anchors'])
 assert set(keys)=={q['id'] for q in c['rows']} and set(anchors)=={t['key'] for t in c['tasks']}
 relations={(x['query_id'],x['scope']):x['label'] for x in c['relations']};r0=read(root/'R0_DECISIONS.json')['decisions'];out={'R0':{q:r['logical_edit_id'] for q,r in r0.items()}};cal={};score_receipts={}
 for kind in ('JOINT','CONTEXT_MIN'):
  result=calibrate(keys,c['tasks'],c['rows'],relations,kind);cal[kind]=result;assert result['status']=='PASS';supports=[]
  for t in c['tasks']:
   ids=[t['source']['id']]+[x['id'] for x in c['rows'] if x['role']=='ROUTER_TRAIN' and relations[x['id'],t['key']]=='YES'];supports.append([keys[q] for q in ids])
  decisions={};test_scores={};fold_scores=[]
  for q in c['rows']:
   vals=[scores(keys[q['id']],s,kind) for s in supports];test_scores[q['id']]=vals;index=choose(vals,result['tau']);decisions[q['id']]=None if index is None else c['tasks'][index]['key']
  train=[q for q in c['rows'] if q['role']=='ROUTER_TRAIN']
  for q in train:
   fold_supports=[]
   for t in c['tasks']:
    ids=[t['source']['id']]+[x['id'] for x in train if x['article']!=q['article'] and relations[x['id'],t['key']]=='YES'];fold_supports.append([keys[z] for z in ids])
   fold_scores.append(dict(query=q['id'],values=[scores(keys[q['id']],s,kind) for s in fold_supports],labels=[relations[q['id'],t['key']] for t in c['tasks']]))
  score_receipts[kind]=dict(test_scores=test_scores,leave_group_out_scores=fold_scores,expert_order=[t['key'] for t in c['tasks']])
  out[kind]=decisions
 outputs={(x['expert'],x['query']):x['path'] for x in recs};natural=[dict(router=kind,query=q,selected=e,path=c['Base_outputs'][q] if e is None else outputs[e,q]) for kind,dec in out.items() for q,e in dec.items()]
 assert len(natural)==3*len(c['rows']);write(root/'ROUTES.json',dict(calibration=cal,decisions=out,natural=natural,new_generations=0));write(root/'ROUTE_SCORE_RECEIPTS.json',score_receipts);return len(natural)

def summarize(root):
 root=Path(root);c,recs=collect(root);a=read(root/'BLIND_ALIASES.json');m=read(root/'BLIND_MATERIALS.json');v=read(root/'BLIND_VERDICTS.json');assert set(v)==set(m) and all(x.get('label') in LABELS and x.get('reason') for x in v.values()),'Missing/invalid/extra blinded verdicts'
 rel={(x['query_id'],x['scope']):x['label'] for x in c['relations']};label=lambda e,q:v[a[(e or 'BASE')+'|'+q]]['label'];held=[x for x in c['rows'] if x['role']=='HELDOUT'];basecorrect=[x for x in held if label(None,x['id'])=='CORRECT'];routes=read(root/'ROUTES.json');result=dict(status='COMPLETE',experts=len(c['tasks']),matrix_rows=len(c['rows']),capability_consumers=len(recs),heldout=len(held),heldout_Base_correct=len(basecorrect),forced={},natural={},scope_limitations='Anatomical-target development scopes; no independent clinical/patient generalization.')
 for t in c['tasks']:
  key=t['key'];source=t['source']['id'];yes=[x for x in held if rel[x['id'],key]=='YES'];result['forced'][key]=dict(source_correct=label(key,source)=='CORRECT',applicable_heldout_correct=sum(label(key,x['id'])=='CORRECT' for x in yes),applicable_heldout_total=len(yes),all_heldout_labels=dict(collections.Counter(label(key,x['id']) for x in held)),Base_correct_damage=sum(label(key,x['id'])=='INCORRECT' for x in basecorrect),Base_correct_abstain=sum(label(key,x['id'])=='ABSTAIN' for x in basecorrect))
 for kind,decisions in routes['decisions'].items():
  rows=[]
  for x in held:
   q=x['id'];e=decisions[q];legal={t['key'] for t in c['tasks'] if rel[q,t['key']]=='YES'};reachable=any(label(t['key'],q)=='CORRECT' for t in c['tasks'] if t['key'] in legal);y=label(e,q);b=label(None,q)
   rows.append(dict(correct=y=='CORRECT',damage=b=='CORRECT' and y=='INCORRECT',abstain=b=='CORRECT' and y=='ABSTAIN',legal=e in legal if legal else e is None,selection_loss=(reachable or b=='CORRECT') and y!='CORRECT',writing_failure=bool(legal) and b!='CORRECT' and not reachable,activation_side_effect=e is not None and b=='CORRECT' and y=='INCORRECT',OFF=e is None,has_legal=bool(legal),unknown=any(rel[q,t['key']]=='UNKNOWN' for t in c['tasks'])))
  result['natural'][kind]={k:sum(x[k] for x in rows) for k in rows[0]};result['natural'][kind]['total']=len(rows)
  result['natural'][kind]['source_correct']=sum(label(decisions[t['source']['id']],t['source']['id'])=='CORRECT' for t in c['tasks'])
  result['natural'][kind]['source_total']=len(c['tasks'])
 counts=[read(root/str(i)/'EXIT.json') for i in range(5)]+[read(root/'source'/str(i)/'EXIT.json') for i in range(len(c['tasks']))]+[read(root/'features'/str(i)/'EXIT.json') for i in range(2)]
 result['counts']={k:sum(x['counts'].get(k,0) for x in counts) for k in ('forwards','backwards','updates','generations','vision_forwards','query_keys','anchor_keys')};result['GPU_process_seconds']=sum(x['seconds'] for x in counts)
 write(root/'RESULTS_PRIVATE.json',result);write(root/'CONSUMERS_COMPLETE.json',dict(status='PASS',capability=len(recs),natural=len(routes['natural']),all_labels=len(v),all_model_EXIT0=all(x['code']==0 for x in counts)));return result

def retire(root):
 import os,time
 root=Path(root);assert read(root/'CONSUMERS_COMPLETE.json')['status']=='PASS';c,_=collect(root);paths=[]
 for i in range(len(c['tasks'])):paths += [root/'source'/str(i)/n for n in ('active.pt','final.pt')]
 for i in range(2):paths += [root/'features'/str(i)/n for n in ('KEYS_ACTIVE.pt','KEYS_FINAL.pt')]
 for p in paths:
  assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
  launch=read(p.parent/'LAUNCH.json');proc=Path('/proc')/str(launch['pid'])/'stat';assert not proc.exists() or int(proc.read_text().split()[21])!=launch['start_ticks'],'Original owner still running'
 sizes=[p.stat().st_size for p in paths]
 for p in paths:p.unlink()
 receipt=dict(files=[str(p) for p in paths],count=len(paths),bytes=sum(sizes),reason='All frozen model, complete matrix/routing/binding/save-load/scoring consumers complete',retained_copy=False,requires_retraining_to_rebuild=True,time=time.time());write(root/'RETIREMENT.json',receipt);return receipt
