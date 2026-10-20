"""Terminal-only preparation, exact-input score reuse, at most one blinded batch."""
import os,sys,json,time,subprocess,uuid,tempfile,random
from pathlib import Path

def remote(code):
    p=subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n',capture_output=True,text=True,timeout=60);p.check_returncode();return json.loads(p.stdout)

def main():
    c=json.loads(Path(os.environ['CONFIG_FILE']).read_text());root=Path(c['local_root']);root.mkdir(exist_ok=False)
    (root/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),epoch=time.time())))
    deadline=time.time()+1950
    code='''from pathlib import Path
import json,uuid
r=Path('/data/bmw/Knowledge_editing/outputs/vigor-controllerN3-v1-20261020/run')
if not (r/'ENDED.json').exists():print(json.dumps(dict(ready=False)))
else:
 result=json.loads((r/'public/RESULT.json').read_text());ended=json.loads((r/'ENDED.json').read_text())
 if ended['status']!='COMPLETE':print(json.dumps(dict(ready=True,failed=True,status=ended['status'])))
 else:
  prior=Path('/data/bmw/Knowledge_editing/outputs/vigor-editperf8-v1-20261019/attempt2');m=json.loads((prior/'private/MANIFEST.json').read_text());new=[];cache=[]
  old=Path('/data/bmw/Knowledge_editing/outputs/vigor-controllerBC-semantic-v1-20261020/run/private');op=json.loads((old/'PACKETS.json').read_text());oscores={d['opaque_query_id']:d['is_correct'] for d in json.loads((old/'SUMMARY').read_text())['decisions']}
  for p in op['records']:
   if p['opaque_query_id'] in oscores:cache.append(dict(question=p['question'],gold_answer=p['gold_answer'],raw_base_answer=p['raw_base_answer'],score=oscores[p['opaque_query_id']],provenance='prior_E01_semantic'))
  fs=json.loads((prior/'private/scoring/SUMMARY.json').read_text())['scores']
  for t in m['tasks'][:3]:
   qid=next(e['query'] for e in t['panel'] if e['task']=='T0');qr=m['queries'][qid]
   for phase,sid in [('base','B_'+qid),(t['alias'],t['alias']+'_'+qid)]:
    f=prior/'private/outputs'/phase/(qid+'.json')
    if f.exists() and sid in fs:
     out=json.loads(f.read_text())
     if out['status']=='COMPLETE':cache.append(dict(question=qr['question'],gold_answer=qr['reference'],raw_base_answer=out['raw_answer'],score=fs[sid],provenance='prior_formal_exact_input'))
  cp=Path('/data/bmw/Knowledge_editing/outputs/vigor-controllerBC3-v1-20261020/run/private/scoring')
  ci=json.loads((cp/'PRIVATE_INPUT.json').read_text());cs=json.loads((cp/'SUMMARY.json').read_text())['scores']
  for rec in ci['records']:
   aliases=ci['mapping'][rec['opaque_query_id']];vals=[cs[a] for a in aliases if a in cs]
   if vals:
    assert all(v==vals[0] for v in vals)
    cache.append(dict(question=rec['question'],gold_answer=rec['gold_answer'],raw_base_answer=rec['raw_base_answer'],score=vals[0],provenance='prior_BC3_exact_input'))
  for case in result['cases']:
   t=next(t for t in m['tasks'] if t['alias']==case['sample']);row=m['queries'][next(e['query'] for e in t['panel'] if e['task']=='T0')]
   for arm in case['arms']:
    for tag,g in arm['generations'].items():
     alias=case['sample']+'_'+arm['name']+'_'+tag
     if g['status']!='COMPLETE':new.append(dict(alias=alias,status='GENERATION_INCOMPLETE'));continue
     source_tag='step16' if tag=='final' and g.get('reused_same_native_endpoint') else tag
     p=r/'private'/(case['sample']+'_'+arm['name']+'_'+source_tag+'_GENERATION.json');v=json.loads(p.read_text());assert v['status']=='COMPLETE'
     new.append(dict(alias=alias,status='COMPLETE',question=row['question'],gold_answer=row['reference'],raw_base_answer=v['raw_answer']))
  print(json.dumps(dict(ready=True,failed=False,outputs=new,prior_cache=cache)))
'''
    while True:
        data=remote(code)
        if data['ready']:break
        if time.time()>deadline:raise TimeoutError('Terminal wait ceiling, no judge')
        time.sleep(30)
    if data['failed']:
        (root/'SUMMARY.json').write_text(json.dumps(dict(status='NOT_STARTED_GPU_ENGINEERING_FAILURE',GPU_status=data['status'],new_judge_requests=0)));return
    def key(x):return (x['question'],x['gold_answer'],x['raw_base_answer'])
    cache={key(x):dict(score=x['score'],provenance=x['provenance']) for x in data['prior_cache']};seen={};records=[];mapping={};scores={};provenance={};missing=[]
    for x in data['outputs']:
        alias=x['alias']
        if x['status']!='COMPLETE':missing.append(alias);continue
        k=key(x)
        if k in cache:scores[alias]=cache[k]['score'];provenance[alias]=cache[k]['provenance'];continue
        if k in seen:mapping[seen[k]].append(alias);continue
        opaque=uuid.uuid4().hex;seen[k]=opaque;mapping[opaque]=[alias];records.append(dict(opaque_query_id=opaque,question=x['question'],gold_answer=x['gold_answer'],raw_base_answer=x['raw_base_answer']))
    assert len(records)<=3;random.SystemRandom().shuffle(records)
    (root/'PRIVATE_INPUT.json').write_text(json.dumps(dict(records=records,mapping=mapping,cache_scores=scores,cache_provenance=provenance,missing_generation=missing)))
    if records:
        packet=root/'PACKETS.json';packet.write_text(json.dumps(dict(records=records)))
        config=dict(scorer_source=c['scorer_source'],private_root=str(root/'judge_first_call'),packets=str(packet),payload_max=3,attempts=1,cli=c['cli']);cp=root/'judge_config.json';cp.write_text(json.dumps(config))
        os.environ['CONFIG_FILE']=str(cp)
        from .controllerN3_judge import main as judge_main
        judge_main()
        q=json.loads((root/'judge_first_call/SUMMARY.json').read_text())
        for d in q['decisions']:
            for alias in mapping[d['opaque_query_id']]:scores[alias]=d['is_correct'];provenance[alias]='new_blind_score'
        usage=q['usage'];state=q['status']
    else:usage=None;state='COMPLETE_ALL_EXACT_INPUT_REUSED'
    summary=dict(status=state,scores=scores,provenance=provenance,new_payloads=len(records),unique_new_batches=int(bool(records)),generation_missing=missing,unscored=[x['alias'] for x in data['outputs'] if x['alias'] not in scores],usage=usage,attempts_per_payload=1,retries=0,new_generations=0)
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
