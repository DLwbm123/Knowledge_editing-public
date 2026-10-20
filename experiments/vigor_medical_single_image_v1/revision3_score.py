"""Terminal-only blinded P/R qualification; exact caches are criterion-specific."""
import json,os,time,sys,uuid,random
from pathlib import Path
from .factF3_score import remote
from .prepare import write

def main():
 c=json.loads(Path(os.environ['CONFIG_FILE']).read_text());root=Path(c['local_root']);root.mkdir(exist_ok=False);write(root/'STARTED.json',dict(pid=os.getpid(),epoch=time.time()))
 code="""import json,time
from pathlib import Path
r=Path('/data/bmw/Knowledge_editing/outputs/vigor-revision3-v1-20261020');assert json.loads((r/'ENDED.json').read_text())['status']=='COMPLETE';clock=json.loads((r/'EXPERIMENT_CLOCK.json').read_text());assert time.time()<clock['deadline_epoch']-1200
assert not (r/'private/QUALIFICATION_RESERVED.json').exists();(r/'private/QUALIFICATION_RESERVED.json').write_text(json.dumps(dict(epoch=time.time(),attempts_per_criterion=1)))
sources=json.loads((r/'private/SOURCES.json').read_text())['sources'];result=json.loads((r/'public/RESULT.json').read_text());rows=[]
for event in result['outputs']:
 s=next(s for s in sources if s['alias']==event['sample']);v=json.loads((r/'private'/(event['alias']+'_GENERATION.json')).read_text());assert v['status']=='COMPLETE'
 rows.append(dict(alias=event['alias'],question=s['question'],gold_answer=s['reference'],raw_base_answer=v['raw_answer']))
assert len(rows)==6
# Reuse already independently graded source answers only; no evaluation QA joins.
cache=[];prior=Path('/data/bmw/Knowledge_editing/outputs/vigor-editperf8-v1-20261019/attempt2');m=json.loads((prior/'private/MANIFEST.json').read_text());grades=json.loads((prior/'private/scoring/SUMMARY.json').read_text())['scores']
for t in m['tasks'][:3]:
 qid=next(e['query'] for e in t['panel'] if e['task']=='T0');q=m['queries'][qid]
 for alias,f in [('B_'+qid,prior/'private/outputs/base'/(qid+'.json')),(t['alias']+'_'+qid,prior/'private/outputs'/t['alias']/(qid+'.json'))]:
  if alias in grades and f.exists():
   v=json.loads(f.read_text())
   if v['status']=='COMPLETE':cache.append(dict(question=q['question'],gold_answer=q['reference'],raw_base_answer=v['raw_answer'],score=grades[alias]))
for parent in ['vigor-controllerBC3-v1-20261020/run','vigor-controllerN3-v1-20261020/run','vigor-factF3-v1-20261020']:
 parent=Path('/data/bmw/Knowledge_editing/outputs')/parent;d=parent/('private/post_scoring' if parent.name=='vigor-factF3-v1-20261020' else 'private/scoring')
 if not (d/'PRIVATE_INPUT.json').exists():continue
 inp=json.loads((d/'PRIVATE_INPUT.json').read_text());scores=json.loads((d/'SUMMARY.json').read_text())['scores']
 for v in inp['records']:
  if not any(v['question']==s['question'] and v['gold_answer']==s['reference'] for s in sources):continue
  vals=[scores[a] for a in inp['mapping'][v['opaque_query_id']] if a in scores]
  if vals:
   assert len(set(vals))==1;cache.append(dict(v,score=vals[0]))
print(json.dumps(dict(rows=rows,cache=cache,deadline_epoch=clock['deadline_epoch'])))
"""
 data=remote(code);key=lambda x:(x['question'],x['gold_answer'],x['raw_base_answer']);cache={}
 for x in data['cache']:
  k=key(x)
  if k in cache and cache[k]!=x['score']:raise ValueError('Cache conflict, no rejudge')
  cache[k]=x['score']
 criteria={};evidence=[]
 for criterion in ['agreement','faithfulness']:
  if time.time()>=data['deadline_epoch']-600:raise TimeoutError('Shared experiment clock')
  records=[];mapping={};bykey={};scores={};provenance={}
  for x in data['rows']:
   k=key(x);alias=x['alias']
   if criterion=='agreement' and k in cache:scores[alias]=cache[k];provenance[alias]='prior_exact_source_agreement';continue
   if k in bykey:mapping[bykey[k]].append(alias);continue
   opaque=uuid.uuid4().hex;bykey[k]=opaque;mapping[opaque]=[alias];records.append(dict(opaque_query_id=opaque,question=x['question'],gold_answer=x['gold_answer'],raw_base_answer=x['raw_base_answer']))
  random.SystemRandom().shuffle(records);packet=root/(criterion+'_INPUT.json');write(packet,dict(records=records,mapping=mapping,cache_scores=scores));details=None
  if records:
   cfg=dict(scorer_source=c['scorer_source'],private_root=str(root/criterion),packets=str(packet),payload_max=6,attempts=1,cli=c['cli'],criterion=criterion);cp=root/(criterion+'_config.json');write(cp,cfg);os.environ['CONFIG_FILE']=str(cp)
   from .revision3_judge import main as judge
   judge();details=json.loads((Path(cfg['private_root'])/'SUMMARY.json').read_text())
   for d in details['decisions']:
    for alias in mapping[d['opaque_query_id']]:scores[alias]=d['is_correct'];provenance[alias]='new_blind_'+criterion
   evidence.append(dict(criterion=criterion,**{k:v for k,v in details.items() if k!='decisions'}))
  criteria[criterion]=dict(scores=scores,provenance=provenance,missing=[x['alias'] for x in data['rows'] if x['alias'] not in scores]);write(root/'PARTIAL.json',dict(criteria=criteria,evidence=evidence))
  if details is not None and details['status']!='COMPLETE':break
 result=dict(status='COMPLETE' if len(criteria)==2 and all(not v['missing'] for v in criteria.values()) else 'INCOMPLETE_NO_RETRY',criteria=criteria,judge_evidence=evidence,targets=[])
 for x in data['rows']:
  vals=[criteria.get(k,{}).get('scores',{}).get(x['alias']) for k in ['agreement','faithfulness']];qualified=all(v is True for v in vals) if all(v is not None for v in vals) else None
  result['targets'].append(dict(alias=x['alias'],agreement=vals[0],faithfulness=vals[1],qualified=qualified))
 result['same_P_R']={s:next(x['raw_base_answer'] for x in data['rows'] if x['alias']==s+'_P')==next(x['raw_base_answer'] for x in data['rows'] if x['alias']==s+'_R') for s in ['E01','E02','E03']}
 write(root/'SUMMARY.json',result)
 public={k:v for k,v in result.items() if k!='criteria'};public['criteria']=criteria
 sync="""import json,sys
from pathlib import Path
p=json.loads(sys.stdin.readline());r=Path('/data/bmw/Knowledge_editing/outputs/vigor-revision3-v1-20261020');d=r/'private/scoring';d.mkdir(exist_ok=False)
for name,text in p['files'].items():
 f=d/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text)
(r/'public/QUALIFICATION.json').write_text(json.dumps(p['public'],indent=2)+'\\n');print(json.dumps(dict(saved=True)))
"""
 remote(sync,dict(files={str(p.relative_to(root)):p.read_text() for p in root.rglob('*') if p.is_file()},public=public));print(json.dumps(public),flush=True)
if __name__=='__main__':main()
