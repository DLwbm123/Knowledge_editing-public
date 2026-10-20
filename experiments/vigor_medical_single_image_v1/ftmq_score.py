"""Terminal-only blinded teacher qualification; exact caches are criterion-specific."""
import json,os,time,sys,uuid,random
from pathlib import Path
from .factF3_score import remote
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+"\n")

def main():
 c=json.loads(Path(os.environ['CONFIG_FILE']).read_text());root=Path(c['local_root']);root.mkdir(exist_ok=False);write(root/'STARTED.json',dict(pid=os.getpid(),epoch=time.time()))
 code="import json,time\nfrom pathlib import Path\nr=Path('/data/bmw/Knowledge_editing/outputs/vigor-ftmq3-v1-20261020');end=json.loads((r/'ENDED.json').read_text());result=json.loads((r/'public/RESULT.json').read_text());assert end['status'] in ['COMPLETE','FAILED_NO_RETRY'] and all(result['restoration'].values());clock=json.loads((r/'EXPERIMENT_CLOCK.json').read_text());assert time.time()<clock['deadline_epoch']-1200\nassert not (r/'private/QUALIFICATION_RESERVED.json').exists()\nwith (r/'private/QUALIFICATION_RESERVED.json').open('x') as f:json.dump(dict(epoch=time.time(),attempts_per_criterion=1),f)\nparent=Path('/data/bmw/Knowledge_editing/outputs');old=parent/'vigor-revision3-v1-20261020';sources=json.loads((old/'private/SOURCES.json').read_text())['sources'];rows=[]\nfor sample in ['E01','E02','E03']:\n f=r/'private'/(sample+'_GENERATION.json');v=json.loads(f.read_text()) if f.exists() else {};status=v.get('status','MISSING')\n event=next((e for e in result['teachers'] if e['sample']==sample),{})\n if event.get('status')!='COMPLETE':status='MISSING'\n ss=next(s for s in sources if s['alias']==sample);rows.append(dict(alias=sample+'_TQ',status=status,question=ss['question'],gold_answer=ss['reference'],raw_base_answer=v.get('raw_answer','')))\ncache=[];prior=parent/'vigor-editperf8-v1-20261019/attempt2';m=json.loads((prior/'private/MANIFEST.json').read_text());grades=json.loads((prior/'private/scoring/SUMMARY.json').read_text())['scores']\nfor t in m['tasks'][:3]:\n qid=next(e['query'] for e in t['panel'] if e['task']=='T0');q=m['queries'][qid]\n for alias,f in [('B_'+qid,prior/'private/outputs/base'/(qid+'.json')),(t['alias']+'_'+qid,prior/'private/outputs'/t['alias']/(qid+'.json'))]:\n  if alias in grades and f.exists():\n   v=json.loads(f.read_text())\n   if v['status']=='COMPLETE':cache.append(dict(criterion='agreement',question=q['question'],gold_answer=q['reference'],raw_base_answer=v['raw_answer'],score=grades[alias]))\nfor pname,dname in [('vigor-controllerBC3-v1-20261020/run','private/scoring'),('vigor-controllerN3-v1-20261020/run','private/scoring'),('vigor-factF3-v1-20261020','private/post_scoring'),('vigor-hpse2-v1-20261020','private/scoring')]:\n d=parent/pname/dname\n if not (d/'PRIVATE_INPUT.json').exists():continue\n inp=json.loads((d/'PRIVATE_INPUT.json').read_text());scores=json.loads((d/'SUMMARY.json').read_text())['scores']\n for v in inp['records']:\n  if not any(v['question']==ss['question'] and v['gold_answer']==ss['reference'] for ss in sources):continue\n  vals=[scores[a] for a in inp['mapping'][v['opaque_query_id']] if a in scores]\n  if vals:\n   assert len(set(vals))==1;cache.append(dict(v,score=vals[0],criterion='agreement'))\nqual=json.loads((old/'public/QUALIFICATION.json').read_text())\nfor ev in json.loads((old/'public/RESULT.json').read_text())['outputs']:\n ss=next(s for s in sources if s['alias']==ev['sample']);v=json.loads((old/'private'/(ev['alias']+'_GENERATION.json')).read_text())\n for criterion in ['agreement','faithfulness']:\n  score=qual['criteria'][criterion]['scores'].get(ev['alias'])\n  if score is not None:cache.append(dict(question=ss['question'],gold_answer=ss['reference'],raw_base_answer=v['raw_answer'],score=score,criterion=criterion))\nprint(json.dumps(dict(rows=rows,cache=cache,deadline_epoch=clock['deadline_epoch'])))\n"
 data=remote(code);key=lambda x:(x['question'],x['gold_answer'],x['raw_base_answer']);cache={}
 for x in data['cache']:
  k=(x['criterion'],key(x))
  if k in cache and cache[k]!=x['score']:raise ValueError('Cache conflict, no rejudge')
  cache[k]=x['score']
 criteria={};evidence=[]
 for criterion in ['agreement','faithfulness']:
  if time.time()>=data['deadline_epoch']-600:raise TimeoutError('Shared experiment clock')
  records=[];mapping={};bykey={};scores={};provenance={}
  for x in data['rows']:
   k=key(x);alias=x['alias']
   if x['status']!='COMPLETE':continue
   if (criterion,k) in cache:scores[alias]=cache[(criterion,k)];provenance[alias]='prior_exact_'+criterion;continue
   if k in bykey:mapping[bykey[k]].append(alias);continue
   opaque=uuid.uuid4().hex;bykey[k]=opaque;mapping[opaque]=[alias];records.append(dict(opaque_query_id=opaque,question=x['question'],gold_answer=x['gold_answer'],raw_base_answer=x['raw_base_answer']))
  random.SystemRandom().shuffle(records);packet=root/(criterion+'_INPUT.json');write(packet,dict(records=records,mapping=mapping,cache_scores=scores));details=None
  if records:
   cfg=dict(scorer_source=c['scorer_source'],private_root=str(root/criterion),packets=str(packet),payload_max=3,attempts=1,cli=c['cli'],criterion=criterion);cp=root/(criterion+'_config.json');write(cp,cfg);os.environ['CONFIG_FILE']=str(cp)
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
 write(root/'SUMMARY.json',result)
 public={k:v for k,v in result.items() if k!='criteria'};public['criteria']=criteria
 sync="""import json,sys
from pathlib import Path
p=json.loads(sys.stdin.readline());r=Path('/data/bmw/Knowledge_editing/outputs/vigor-ftmq3-v1-20261020');d=r/'private/scoring';d.mkdir(exist_ok=False)
for name,text in p['files'].items():
 f=d/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text)
(r/'public/QUALIFICATION.json').write_text(json.dumps(p['public'],indent=2)+'\\n');print(json.dumps(dict(saved=True)))
"""
 remote(sync,dict(files={str(p.relative_to(root)):p.read_text() for p in root.rglob('*') if p.is_file()},public=public));print(json.dumps(public),flush=True)
if __name__=='__main__':main()
