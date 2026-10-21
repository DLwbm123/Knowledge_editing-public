"""Terminal-only protected on-policy final semantic and six-query locality scoring."""
import os,json,sys,subprocess,uuid,random,time
from pathlib import Path

def remote(code,data=None):
 p=subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n'+(json.dumps(data)+'\n' if data is not None else ''),capture_output=True,text=True,timeout=60);p.check_returncode();return json.loads(p.stdout)

def main():
 config=json.loads(Path(os.environ['CONFIG_FILE']).read_text());phase=config['phase'];assert phase=='post';root=Path(config['local_root']);root.mkdir(exist_ok=False);(root/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),epoch=time.time(),phase=phase)))
 code="""import json
from pathlib import Path
m=Path('/data/bmw/Knowledge_editing/outputs/vigor-factF3-v1-20261020');p=Path('/data/bmw/Knowledge_editing/outputs/vigor-editperf8-v1-20261019/attempt2');phase=json.loads(__import__('sys').stdin.readline())['phase'];cache=[];old=json.loads((p/'private/MANIFEST.json').read_text());scores=json.loads((p/'private/scoring/SUMMARY.json').read_text())['scores']
for qid,q in old['queries'].items():
 f=p/'private/outputs/base'/(qid+'.json');sid='B_'+qid
 if f.exists() and sid in scores:
  v=json.loads(f.read_text())
  if v['status']=='COMPLETE':cache.append(dict(question=q['question'],gold_answer=q['reference'],raw_base_answer=v['raw_answer'],score=scores[sid],provenance='formal_Base_exact_input'))
# Add all previously scored formal post outputs and recent development outputs.
for task in old['tasks']:
 for event in task['panel']:
  qid=event['query'];q=old['queries'][qid];f=p/'private/outputs'/task['alias']/(qid+'.json');sid=task['alias']+'_'+qid
  if f.exists() and sid in scores:
   v=json.loads(f.read_text())
   if v['status']=='COMPLETE':cache.append(dict(question=q['question'],gold_answer=q['reference'],raw_base_answer=v['raw_answer'],score=scores[sid],provenance='formal_post_exact_input'))
for parent_name in ['vigor-controllerBC3-v1-20261020','vigor-controllerN3-v1-20261020']:
 cp=Path('/data/bmw/Knowledge_editing/outputs')/parent_name/'run/private/scoring'
 if not (cp/'PRIVATE_INPUT.json').exists():continue
 ci=json.loads((cp/'PRIVATE_INPUT.json').read_text());cs=json.loads((cp/'SUMMARY.json').read_text())['scores']
 for rec in ci['records']:
  vals=[cs[a] for a in ci['mapping'][rec['opaque_query_id']] if a in cs]
  if vals:
   assert all(v==vals[0] for v in vals)
   cache.append(dict(question=rec['question'],gold_answer=rec['gold_answer'],raw_base_answer=rec['raw_base_answer'],score=vals[0],provenance='previous_development_exact_input'))
cp=m/'private/post_scoring'
ci=json.loads((cp/'PRIVATE_INPUT.json').read_text());cs=json.loads((cp/'SUMMARY.json').read_text())['scores']
for rec in ci['records']:
 vals=[cs[a] for a in ci['mapping'][rec['opaque_query_id']] if a in cs]
 if vals:
  assert all(v==vals[0] for v in vals);cache.append(dict(rec,score=vals[0],provenance='previous_F_exact_input'))
for pname in ['vigor-hpse2-v1-20261020','vigor-alignA3-v1-20261020','vigor-layer6N3-v1-20261020','vigor-layer6G3-v1-20261020']:
 cp=Path('/data/bmw/Knowledge_editing/outputs')/pname/'private/scoring'
 ci=json.loads((cp/'PRIVATE_INPUT.json').read_text());cs=json.loads((cp/'SUMMARY.json').read_text())['scores']
 for rec in ci['records']:
  vals=[cs[a] for a in ci['mapping'][rec['opaque_query_id']] if a in cs]
  if vals:
   assert all(x==vals[0] for x in vals);cache.append(dict(rec,score=vals[0],provenance='previous_H_exact_input'))
ss=json.loads((Path('/data/bmw/Knowledge_editing/outputs/vigor-revision3-v1-20261020')/'private/SOURCES.json').read_text())['sources']
for pname,events in [('vigor-revision3-v1-20261020',[(s['alias']+'_'+a,s['alias']) for s in ss for a in ['P','R']]),('vigor-ftmq3-v1-20261020',[(s['alias'],s['alias']) for s in ss])]:
 cp=Path('/data/bmw/Knowledge_editing/outputs')/pname;cs=json.loads((cp/'public/QUALIFICATION.json').read_text())['criteria']['agreement']['scores']
 for alias,sample in events:
  score_alias=sample+'_TQ' if pname.startswith('vigor-ftmq') else alias
  if score_alias not in cs:continue
  rec=next(s for s in ss if s['alias']==sample);v=json.loads((cp/'private'/(alias+'_GENERATION.json')).read_text())
  if v['status']=='COMPLETE':cache.append(dict(question=rec['question'],gold_answer=rec['reference'],raw_base_answer=v['raw_answer'],score=cs[score_alias],provenance='qualified_teacher_source_exact_input'))
rows=[]
if phase=='base':
 r=m/'basequalify';assert json.loads((r/'ENDED.json').read_text())['status']=='COMPLETE';pool=json.loads((m/'private/LOCALITY_CANDIDATES.json').read_text())['rows']
 for x in pool:
  v=json.loads((r/'private'/(x['alias']+'_GENERATION.json')).read_text());rows.append(dict(alias=x['alias'],status=v['status'],question=x['query']['question'],gold_answer=x['query']['reference'],raw_base_answer=v.get('raw_answer','')))
else:
 r=Path('/data/bmw/Knowledge_editing/outputs/vigor-teacherLocality3-v1-20261020');assert json.loads((r/'ENDED.json').read_text())['status']=='COMPLETE'
 import time
 assert time.time()<json.loads((r/'EXPERIMENT_CLOCK.json').read_text())['deadline_epoch']-600
 assert not (r/'private/SCORING_RESERVED.json').exists();(r/'private/SCORING_RESERVED.json').write_text(json.dumps(dict(epoch=time.time(),attempts=1)));result=json.loads((r/'public/RESULT.json').read_text());panel=json.loads((m/'private/LOCALITY_PANEL.json').read_text())['rows'];sources=json.loads((m/'private/SOURCE_FACT_GROUPS.json').read_text())['facts'];base_scores=json.loads((m/'private/BASE_SCORES.json').read_text())
 for x in panel:cache.append(dict(question=x['query']['question'],gold_answer=x['query']['reference'],raw_base_answer=x['base_answer'],score=True,provenance='qualified_current_Base_exact_input'))
 for event in result.get('evaluation',[]):
  x=next(x for x in panel if x['alias']==event['panel_id']);v=json.loads((r/'private'/(event['alias']+'_GENERATION.json')).read_text()) if event['status']=='COMPLETE' else {};rows.append(dict(alias=event['alias'],status=event['status'],question=x['query']['question'],gold_answer=x['query']['reference'],raw_base_answer=v.get('raw_answer','')))
print(json.dumps(dict(rows=rows,cache=cache)))
"""
 data=remote(code,dict(phase=phase));key=lambda x:(x['question'],x['gold_answer'],x['raw_base_answer']);cache={}
 for x in data['cache']:
  k=key(x)
  if k in cache and cache[k]['score']!=x['score']:raise ValueError('Conflicting existing exact-input grades, no rejudge')
  cache[k]=x
 records=[];mapping={};bykey={};scores={};provenance={};missing=[]
 for x in data['rows']:
  alias=x['alias']
  if x['status']!='COMPLETE':missing.append(alias);continue
  k=key(x)
  if k in cache:scores[alias]=cache[k]['score'];provenance[alias]=cache[k]['provenance'];continue
  if k in bykey:mapping[bykey[k]].append(alias);continue
  opaque=uuid.uuid4().hex;bykey[k]=opaque;mapping[opaque]=[alias];records.append(dict(opaque_query_id=opaque,question=x['question'],gold_answer=x['gold_answer'],raw_base_answer=x['raw_base_answer']))
 assert len(records)<=18 and len(data['rows'])==18;random.SystemRandom().shuffle(records);(root/'PRIVATE_INPUT.json').write_text(json.dumps(dict(records=records,mapping=mapping,cache_scores=scores,cache_provenance=provenance)))
 evidence=[];pending=[]
 for i in range(0,len(records),25):
  batch=records[i:i+25];packet=root/('PACKETS_'+str(i//25)+'.json');packet.write_text(json.dumps(dict(records=batch)));cfg=dict(scorer_source=config['scorer_source'],private_root=str(root/('batch_'+str(i//25))),packets=str(packet),payload_max=18,attempts=1,cli=config['cli']);cp=root/('config_'+str(i//25)+'.json');cp.write_text(json.dumps(cfg));os.environ['CONFIG_FILE']=str(cp)
  from .factF3_judge import main as judge
  judge();v=json.loads((Path(cfg['private_root'])/'SUMMARY.json').read_text());evidence.append({k:v for k,v in v.items() if k!='decisions'})
  for d in v['decisions']:
   for alias in mapping[d['opaque_query_id']]:scores[alias]=d['is_correct'];provenance[alias]='new_blind_score'
  if v['status']!='COMPLETE':pending.extend(r['opaque_query_id'] for r in records[i+25:]);break
 result=dict(status='COMPLETE' if all(x['alias'] in scores for x in data['rows']) else 'SCORING_INCOMPLETE_NO_RETRY',phase=phase,scores=scores,provenance=provenance,new_payloads=sum(x['payloads'] for x in evidence),batches=len(evidence),missing=[x['alias'] for x in data['rows'] if x['alias'] not in scores],attempts_per_payload=1,retries=0,judge_evidence=evidence)
 (root/'SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 sync="""import json,sys
from pathlib import Path
p=json.loads(sys.stdin.readline());r=Path('/data/bmw/Knowledge_editing/outputs/vigor-teacherLocality3-v1-20261020');d=r/'private/scoring';d.mkdir(exist_ok=False)
for name,text in p['files'].items():
 q=d/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(text)
(r/'public/SCORES.json').write_text(json.dumps(p['result'],indent=2)+'\\n');print(json.dumps(dict(saved=True)))
"""
 remote(sync,dict(files={str(p.relative_to(root)):p.read_text() for p in root.rglob('*') if p.is_file()},result=result))
if __name__=='__main__':main()
