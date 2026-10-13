"""Read-only result audit; publish only aggregate counts and paired intervals."""
import os,json,random,collections
from pathlib import Path

def summarize(rows,scores):
 vals=[scores.get(x['judge_key']) for x in rows];n=len(vals);c=sum(x is True for x in vals);missing=sum(x is None for x in vals)
 by=collections.defaultdict(list)
 for r,v in zip(rows,vals):by[r['edit']].append(v)
 return dict(n=n,correct=c,missing=missing,micro=None if missing or not n else c/n,bounds=None if not n else [c/n,(c+missing)/n],edit_macro_bounds=None if not by else [sum(sum(x is True for x in v)/len(v) for v in by.values())/len(by),sum(sum(x is not False for x in v)/len(v) for v in by.values())/len(by)],edits=len(by),sources=len({x['source_group'] for x in rows}))

def paired(a,b,scores):
 key=lambda r:(r['edit'],r['task'],r['query_id'])
 aa={key(x):x for x in a};bb={key(x):x for x in b};assert set(aa)==set(bb)
 clusters=collections.defaultdict(list);missing=0
 for k,x in aa.items():
  u=scores.get(x['judge_key']);v=scores.get(bb[k]['judge_key'])
  if u is None or v is None:missing+=1
  else:clusters[x['source_group']].append(int(v)-int(u))
 values=[x for v in clusters.values() for x in v];n=len(aa);total=sum(values)
 out=dict(pairs=n,missing_pairs=missing,delta_bounds=None if not n else [(total-missing)/n,(total+missing)/n],source_clusters=len(clusters),ci95=None)
 if values:
  rng=random.Random(20260929);vs=list(clusters.values());samples=[]
  for _ in range(2000):
   z=[x for _ in vs for x in rng.choice(vs)];samples.append(sum(z)/len(z))
  samples.sort();out.update(complete_pair_delta=sum(values)/len(values),ci95_complete_pairs=[samples[49],samples[1949]],ci_excludes_missing=bool(missing))
 return out

def main(root):
 read=lambda p:json.loads(p.read_text())
 scores={f.stem:read(f)['is_correct'] for f in (root/'private/judge/scores').glob('*.json')};failed=set(read(root/'private/judge/JUDGE_MISSING_LOCK.json')['keys'])
 masks={(x['edit'],x['task'],x['query_id']):x['base_correct'] for x in read(root/'private/BASE_MASKS.json')['rows']}
 groups=collections.defaultdict(list);needed=set();seen={}
 for f in (root/'jobs').rglob('CONSUMERS.json'):
  for r in read(f):
   cohort='PILOT8' if f.relative_to(root/'jobs').parts[0].startswith('P1-bank') else ('DEV24' if r['order']<=24 else 'REG24')
   ident=(cohort,)+tuple(r[k] for k in ['arm','mode','prefix','edit','task','query_id'])
   if ident in seen:assert seen[ident]==r['judge_key'];continue
   seen[ident]=r['judge_key'];needed.update([r['judge_key'],r['base_judge_key']]);panel=cohort;groups[(panel,r['mode'],r['prefix'])].append(r)
 out={}
 for (panel,mode,prefix),rows in groups.items():
  result={};methods=sorted({x['arm'] for x in rows});tasks=sorted({x['task'] for x in rows})
  for m in methods:
   result[m]={}
   for t in tasks:
    rs=[r for r in rows if r['arm']==m and r['task']==t];result[m][t]=summarize(rs,scores)
    if t in ['T1L','T2L','T2L_PRESSURE']:
     eligible=[r for r in rs if masks.get((r['edit'],t,r['query_id']),t=='T2L_PRESSURE') is True]
     result[m][t]['frozen_base_correct']=summarize(eligible,scores)
  comparisons={}
  for a,b in [('E_orig','A0'),('A0','AH'),('A0','AHS_01'),('E_orig','AHS_01'),('AH','AHS_01')]:
   if a not in methods or b not in methods:continue
   for t in tasks:
    ar=[r for r in rows if r['arm']==a and r['task']==t];br=[r for r in rows if r['arm']==b and r['task']==t]
    if ar and {(r['edit'],r['query_id']) for r in ar}=={(r['edit'],r['query_id']) for r in br}:comparisons[a+'->'+b+'/'+t]=paired(ar,br,scores)
  out[f'{panel}/{mode}/{prefix}']=dict(metrics=result,paired_source_bootstrap=comparisons)
 ledger=read(root/'RESOURCE_LEDGER.json');q=read(root/'QUEUE.json');assert all(j['status']=='COMPLETE' for j in q);assert not needed-set(scores)-failed
 assert all(x['status'] in ['FORMAT_VALID','FAILED_NO_RETRY'] for x in ledger['judge_attempts'])
 audit=dict(compute_jobs=len(q),scoring_required_unique=len(needed),scored=len(needed&set(scores)),permanent_missing=len(needed-set(scores)),pending=0,cumulative_attempts=ledger['judge_submission_attempt_items'],inherited_attempts=3958,scope_attempts=ledger['judge_submission_attempt_items']-3958,GPU_hours=ledger['current_gpu_seconds']/3600,not_confirmatory=True)
 (root/'public/FINAL_METRICS.json').write_text(json.dumps(out,indent=2));(root/'public/FINAL_EXECUTION_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
if __name__=='__main__':main(Path(os.environ['RUN_ROOT']))
