"""Fixed-input paired estimates; seeds are repeated observations, not new cases."""
from collections import defaultdict
from statistics import mean
import random

def summarize_values(values):
 missing=sum(v['delta'] is None for v in values);ed=defaultdict(list);src=defaultdict(list)
 for v in values:ed[v['edit']].append(v['delta']);src[v['source']].append(v)
 means=[mean(v) for v in ed.values()] if values and not missing else [];rng=random.Random(20260927)
 boot=sorted(mean(rng.choices(means,k=len(means))) for _ in range(1000)) if means else [];cluster=[]
 if means:
  for _ in range(1000):
   sampled=defaultdict(list)
   for group in rng.choices(list(src),k=len(src)):
    for v in src[group]:sampled[v['edit']].append(v['delta'])
   cluster.append(mean(mean(v) for v in sampled.values()))
  cluster.sort()
 return dict(missing=missing,denominator=len(values),edits=len(ed),source_groups=len(src),delta_edit_macro=mean(means) if means else None,edit_CI95=[boot[24],boot[974]] if boot else None,source_cluster_CI95=[cluster[24],cluster[974]] if cluster else None,estimable=bool(means))

def interaction(rows,scores,retention):
 arms={a:{(r['edit'],r['task'],r['query_id']):r for r in rows if r['arm']==a} for a in ['M1','M3','M4','M5']};first=arms['M1'];paired=bool(first) and all(set(v)==set(first) for v in arms.values());bytask=defaultdict(list)
 if paired:
  for key,r in first.items():
   if scores.get(r['base_judge_key']) is not retention(r['task']):continue
   vals={a:scores.get(v[key]['judge_key']) for a,v in arms.items()};delta=None if any(v is None for v in vals.values()) else int(vals['M5'])-int(vals['M4'])-int(vals['M3'])+int(vals['M1'])
   bytask[r['task']].append(dict(edit=r['edit'],source=r['source_group'],delta=delta))
 metrics={k:summarize_values(v) for k,v in bytask.items()}
 return dict(formula='(M5-M4)-(M3-M1)',paired_observed=paired,metric_estimable=any(v['estimable'] for v in metrics.values()),metrics=metrics)

def seed_aggregate(allrows,scores,retention,seeds=(20260927,20260928,20260929)):
 groups=defaultdict(dict)
 for r in allrows:
  if r['task'].startswith(('G_','U_')):continue
  key=('DEV24' if r['order']<=24 else 'REG24',r['mode'],r['prefix'],r['arm'],r['task'],r['edit'],r['query_id'])
  groups[key][r['seed']]=r
 output=defaultdict(list)
 for key,rs in groups.items():
  values=[];eligible=[]
  for seed in seeds:
   r=rs.get(seed)
   if r is None:continue
   eligible.append(scores.get(r['base_judge_key']) is retention(r['task']))
   if eligible[-1]:values.append(scores.get(r['judge_key']))
  if not any(eligible):continue
  complete=set(rs)==set(seeds) and all(eligible) and len(values)==len(seeds) and all(v is not None for v in values)
  r=next(iter(rs.values()));output[key[:5]].append(dict(edit=r['edit'],source=r['source_group'],delta=mean(values) if complete else None))
 result=[]
 for k,values in output.items():
  stats=summarize_values(values);stats['edit_macro']=stats.pop('delta_edit_macro');stats['micro']=mean(v['delta'] for v in values) if values and not stats['missing'] else None
  result.append(dict(panel=k[0],mode=k[1],prefix=k[2],arm=k[3],task=k[4],required_seeds=list(seeds),seed_average_before_edit_aggregation=True,**stats))
 return result

def supports(rows,scores,expected_orders):
 groups=defaultdict(list)
 for r in rows:groups[(r['seed'],r['arm'],r['task'])].append(r)
 result=[]
 for (seed,arm,task),rs in groups.items():
  target=rs if task.startswith('G_') else [r for r in rs if scores.get(r['base_judge_key']) is True]
  values=[dict(edit=r['edit'],source=r['source_group'],delta=scores.get(r['judge_key'])) for r in target];stats=summarize_values(values);stats['edit_macro']=stats.pop('delta_edit_macro')
  result.append(dict(seed=seed,arm=arm,task=task,metric='target_PostAcc' if task.startswith('G_') else 'current_Base_correct_retention',numerator=sum(scores.get(r['judge_key']) is True for r in target),micro=sum(scores.get(r['judge_key']) is True for r in target)/len(target) if target and not stats['missing'] else None,**stats,observed_edits=len({r['edit'] for r in rs}),expected_edits=len(expected_orders),planned_coverage_complete={r['order'] for r in rs}==set(expected_orders),unknown_current_Base=sum(scores.get(r['base_judge_key']) is None for r in rs),Base_token_consistency=sum(r['exact_Base_token_consistency'] for r in rs)/len(rs),consistency_denominator=len(rs)))
 return result
