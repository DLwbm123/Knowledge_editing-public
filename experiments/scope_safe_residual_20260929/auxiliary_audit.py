"""Aggregate retained routes, fixed early cohorts and measured runtime costs."""
import json,collections,statistics,os
from pathlib import Path
from final_audit import summarize,paired

def metric(rows,scores,field='judge_key'):
 return summarize([dict(r,judge_key=r[field]) for r in rows],scores)

def routing(rows,scores):
 active=[r for r in rows if r['route']['activated']]
 positive=[r for r in rows if r['task'] in ['T0','T1G','T2G']]
 negative=[r for r in rows if r['task'] in ['T1L','T2L','T2L_PRESSURE','NEW_STRESS_HOLDOUT']]
 negon=[r for r in negative if r['route']['activated']]
 def damage(rs):
  known=[r for r in rs if scores.get(r['base_judge_key']) is not None and scores.get(r['judge_key']) is not None]
  return dict(n=len(rs),complete_pairs=len(known),missing=len(rs)-len(known),Base_correct_to_wrong=sum(scores[r['base_judge_key']] is True and scores[r['judge_key']] is False for r in known),error_rate_bounds=None if not rs else [sum(scores[r['base_judge_key']] is True and scores[r['judge_key']] is False for r in known)/len(rs),(sum(scores[r['base_judge_key']] is True and scores[r['judge_key']] is False for r in known)+len(rs)-len(known))/len(rs)])
 return dict(inputs=len(rows),activation=len(active),positive_inputs=len(positive),associated_expert_hits=sum(r['route']['logical_edit_id']==r['edit'] for r in positive),positive_hit_interpretation='Associated expert is one known legal expert; no complete compatible-scope sets. Not a unique-ID error metric.',negative_inputs=len(negative),negative_activation=len(negon),unconditional_damage=damage(negative),activated_negative_damage=damage(negon),risk_coverage='Observed fixed R0 operating point only; no calibrated alternate gates',correctness=metric(rows,scores))

def main(root):
 read=lambda p:json.loads(p.read_text());scores={p.stem:read(p)['is_correct'] for p in (root/'private/judge/scores').glob('*.json')}
 groups={};execution_ids=set()
 for p in (root/'jobs').rglob('CONSUMERS.json'):
  job=p.relative_to(root/'jobs').parts[0];rows=read(p)
  if not rows:continue
  execution_ids.update(r['execution_id'] for r in rows)
  cohort='PILOT8' if job.startswith('P1-bank') else 'DEV24' if rows[0]['order']<=24 else 'REG24'
  if rows[0]['mode']=='sequential':groups[(cohort,rows[0]['arm'],rows[0]['prefix'])]=rows
 routes={};growth={};invariance=[]
 for (panel,arm,n),rows in groups.items():
  routes[f'{panel}/{arm}/{n}']={task:routing([r for r in rows if r['task']==task],scores) for task in sorted({r['task'] for r in rows})}
  earlygroups={r['edit'] for r in groups[(panel,arm,4)]};early=[r for r in rows if r['edit'] in earlygroups];first=groups[(panel,arm,4)]
  growth[f'{panel}/{arm}/{n}']={}
  for task in sorted({r['task'] for r in first}):
   aa=[r for r in first if r['task']==task];bb=[r for r in early if r['task']==task]
   growth[f'{panel}/{arm}/{n}'][task]=dict(metrics=metric(bb,scores),paired_vs_prefix4=paired(aa,bb,scores),edit_cluster_paired_vs_prefix4=paired([dict(r,source_group=r['edit']) for r in aa],[dict(r,source_group=r['edit']) for r in bb],scores),routing=routing(bb,scores))
  if arm!='A0' and (panel,'A0',n) in groups:
   aa={(r['edit'],r['task'],r['query_id']):r for r in groups[(panel,'A0',n)]};bb={(r['edit'],r['task'],r['query_id']):r for r in rows};assert aa.keys()==bb.keys()
   invariance.append(dict(panel=panel,arm=arm,prefix=n,inputs=len(aa),same_activation=sum(aa[k]['route']['activated']==bb[k]['route']['activated'] for k in aa),same_selected_expert=sum(aa[k]['route']['logical_edit_id']==bb[k]['route']['logical_edit_id'] for k in aa)))
 costs=collections.defaultdict(lambda:dict(continuations=0,steps=0,CE_calls=0,KL_calls=0,residual_calls=0,training_seconds=0))
 for p in (root/'private/curves').rglob('continuation.json'):
  d=read(p);arm=p.parts[-3];c=costs[arm];c['continuations']+=1;c['steps']+=d['steps'];c['CE_calls']+=d['forwards'];c['training_seconds']+=d['seconds']
  c['KL_calls']+=sum(x.get('KL_forward_calls',0) for x in d['curve']);c['residual_calls']+=sum(x.get('extra_residual_forward_calls',0) for x in d['curve'])
 counts=[read(p) for p in (root/'jobs').glob('*/COMPUTE_COUNTS.json')];generations=[read(root/'private/generations'/f'{k}.json')['output'] for k in execution_ids]
 durations=[x['seconds'] for x in generations if 'seconds' in x]
 storage={name:dict(files=len(ps),bytes=sum(p.stat().st_size for p in ps)) for name in ['adapters','checkpoints','initializers','private/keys'] for ps in [[p for p in (root/name).rglob('*') if p.is_file()]]}
 out=dict(routes=routes,fixed_early_four_edits=growth,expert_route_invariance=invariance,costs=dict(training_by_arm=dict(costs),completed_job_model_calls=sum(x['model_calls'] for x in counts),completed_job_resident_seconds=sum(x['seconds'] for x in counts),unique_cached_generations=len(generations),generation_seconds_median=statistics.median(durations) if durations else None,generation_seconds_total=sum(durations),prefill_latency=None,prefill_status='Not separately instrumented; cannot reconstruct from total generation latency',storage=storage),risk_coverage_limitation='R1/Rneg not calibrated: no legal CAL positives; report fixed R0 operating points without post-hoc threshold selection',patient_clusters='Unavailable; source and edit clusters reported separately')
 (root/'public/AUXILIARY_METRICS.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(banks=len(groups),route_invariance=all(x['same_activation']==x['same_selected_expert']==x['inputs'] for x in invariance),training=dict(costs),storage=storage)))
if __name__=='__main__':main(Path(os.environ['RUN_ROOT']))
