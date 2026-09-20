"""CPU evidence analysis; existing labels are read-only and missing labels stay NA."""
import sys,json,csv
from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean,median
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.stage20_closeout import outputs

def table(path,rows):
 if not rows:return
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(values):
 if not values:return dict(N=0,correct=None,accuracy=None,source_macro=None,sources=0)
 by=defaultdict(list)
 for source,v in values:by[source].append(v)
 complete=all(type(v) is bool for _,v in values)
 return dict(N=len(values),correct=sum(v is True for _,v in values) if complete else None,accuracy=mean(v for _,v in values) if complete else None,source_macro=mean(mean(v) for v in by.values()) if complete else None,sources=len(by))
def run(dest):
 dest=Path(dest);c=ROOT/'reports/medtrace_stage24c_20260919/private/run';common=ROOT/'reports/medtrace_stage22_20260919/private/run';scores=read(common/'private/QUALIFIED_SCORE_CACHE.json')['scores'];rows=outputs(c);panel=read(c/'private/PANEL_MEMBERSHIPS.json');stream=read(common/'private/STREAM.json');position={t['canonical_edit_id']:t['order'] for t in stream['tasks']};arms={a:{r['query_id']:r for r in rows if r['arm']==a and r['mode']=='DEV79'} for a in ('E0','E2','R_H')};qids=sorted(arms['E0']);qid={q:f'Q{i+1:03d}' for i,q in enumerate(qids)};sources=sorted({r['source']['source_group'] for r in arms['E0'].values()});sid={s:f'S{i+1:03d}' for i,s in enumerate(sources)};per=[];raw=[]
 score=lambda r,field='output':scores.get(score_key(r['source'],r[field]))
 for q in qids:
  a={k:arms[k][q] for k in arms};base=score(a['E0'],'Base');membership=[k for k,v in panel.items() if q in v];r=dict(query=qid[q],source=sid[a['E0']['source']['source_group']],panels=';'.join(membership),Base=base,**{k:score(v) for k,v in a.items()},**{k+'_winner':position.get(v['route']['logical_edit_id'],0) for k,v in a.items()},RH_E2_tokens_differ=a['R_H']['output']['raw_token_ids']!=a['E2']['output']['raw_token_ids'],RH_E2_correctness_change=score(a['R_H'])!=score(a['E2']))
  per.append(r);raw.append(dict(**r,query_id=q,source_record=a['E0']['source'],outputs={k:v['output'] for k,v in a.items()}))
 table(dest/'PER_QUERY_DEV.csv',per);write(dest/'private/PER_QUERY_PRIVATE.json',raw)
 summaries=[]
 for name,ids in panel.items():
  for arm in arms:
   for stratum in (['all','Base_correct_retention','Base_wrong_fix'] if ('H' in name or 'U' in name) else ['all']):
    rr=[arms[arm][q] for q in ids if stratum=='all' or score(arms[arm][q],'Base')==(stratum=='Base_correct_retention')];summaries.append(dict(panel=name,arm=arm,stratum=stratum,**metrics([(r['source']['source_group'],score(r)) for r in rr])))
 write(dest/'DEV_METRICS.json',dict(rows=summaries,exact_reuse=True,formal_results_unchanged=True));table(dest/'DEV_METRICS.csv',summaries)
 # Use genuine recorded endpoints only; absence remains absence.
 transitions=[]
 old=outputs(common);oldids={q:f'P{i+1:03d}' for i,q in enumerate(sorted({r['query_id'] for r in old}))}
 for arm in ('R_E0','R_E2'):
  prior={}
  for n in (11,19,32,45):
   rr=[r for r in old if r['arm']==arm and r['prefix']==n and r['mode']=='endpoint']
   for r in rr:
    q=r['query_id'];prev=prior.get(q)
    if prev:transitions.append(dict(query=oldids[q],arm=arm,from_N=prev['prefix'],to_N=n,role=r['source']['role'],winner_before=position.get(prev['route']['logical_edit_id'],0),winner_after=position.get(r['route']['logical_edit_id'],0),correct_before=score(prev),correct_after=score(r)))
    prior[q]=r
 table(dest/'RECORDED_PREFIX_TRANSITIONS.csv',transitions)
 profile=dest/'private/run/public/ALL_WRITER_DIAGNOSTICS.json'
 if profile.exists():
  ds=read(profile)['rows'];flat=[]
  for r in ds:
   z=r['regularization'];g=r['total_preclip_gradient'];flat.append(dict(position=r['position'],step=r['step'],reg_gradient=z['applied_regularizer_gradient_norm'],preclip_total=g,reg_to_total=z['applied_regularizer_gradient_norm']/g if g else None,added_increment=z['accumulated_gradient_increment_norm'],changed_elements=z['accumulated_gradient_changed_elements'],native_gradient=r['training_terms']['native']['weighted_gradient_norm'],fit_gradient=r['training_terms']['fit']['weighted_gradient_norm'],H_gradient=r['training_terms']['extra']['weighted_gradient_norm'],U_gradient=r['training_terms']['U']['weighted_gradient_norm'],x_component_gradient=z['x_gradient_norm'],y_component_gradient=z['y_gradient_norm'],scale_relative=z['scale_plateau']['relative_difference'],**{k+'_centered_to_ridge':v['centered_norm']/v['ridge'] for k,v in z['kernels'].items()},**{k+'_offdiag_mean':v['offdiag_mean'] for k,v in z['kernels'].items()}))
  table(dest/'ALL_WRITER_DIAGNOSTICS.csv',flat);write(dest/'ALL_WRITER_DIAGNOSTICS.json',dict(rows=flat,component_norms='Historical isolated component diagnostics; combined applied gradient uses R2 scaled VJP',total_gradient='Includes regularizer; norms cannot be subtracted to recover training-only vector',effective_update_evidence='Existing R2 two cases: exact OFF repeats, nonzero ON delta, no new training',summary={k:dict(min=min(r[k] for r in flat),median=median(r[k] for r in flat),max=max(r[k] for r in flat)) for k in ('reg_gradient','reg_to_total','patch_centered_to_ridge','scale_relative')}))
 ownfile=dest/'private/run/private/OWNERSHIP.json'
 if ownfile.exists():
  own=read(ownfile);slots=own['slots']
  for r in slots:
   role_owners={role:[t['order'] for t in stream['tasks'][:r['N']] if any(query_id(x)==r['query_id'] for x in t[role])] for role in ('H_fit','U_fit')}
   r['winner_H_supervised']=r['winner'] in role_owners['H_fit'];r['winner_U_supervised']=r['winner'] in role_owners['U_fit']
  oq=sorted({r['query_id'] for r in slots});qi={q:f'T{i+1:03d}' for i,q in enumerate(oq)};table(dest/'PROTECTION_RESPONSIBILITY.csv',[{k:(qi[v] if k=='query_id' else v) for k,v in r.items() if k!='source_group'} for r in slots]);table(dest/'PROTECTION_PREFIX_HISTORY.csv',[{k:(qi[v] if k=='query_id' else v) for k,v in r.items()} for r in own['history']]);summ=[]
  for n in (19,45):
   for role in ('H_fit','U_fit'):
    rr=[r for r in slots if r['N']==n and r['role']==role];unique={r['query_id']:r for r in rr};summ.append(dict(N=n,role=role,slots=len(rr),unique_QA=len(unique),slot_owner_wins=sum(r['winner_is_this_owner'] for r in rr),any_owner_wins_unique=sum(r['winner_is_any_owner'] for r in unique.values()),within_own_radius=sum(r['own_radius'] for r in rr),winner_switched_since_owner_insertion=sum(r['switched'] for r in rr),denominator_note='Per-slot and any-owner unique-QA are distinct'))
  
  for r in own['history']:assert all(k<=r['prefix']<=r['N'] for k in r['known_owners'])
  for x in summ:
   uq={r['query_id']:r for r in slots if r['N']==x['N'] and r['role']==x['role']};x['any_owner_source_macro']=metrics([(r['source_group'],r['winner_is_any_owner']) for r in uq.values()])['source_macro']
  write(dest/'OWNERSHIP_SUMMARY.json',dict(rows=summ,no_future_prefix_used=True))
  base=read(dest/'private/run/private/BASE.json')['records'];bd={r['query_id']:r for r in base};teachers=read(dest/'private/run/private/TEACHER_REPLAY.json')['rows'];teacher_exact={r['query_id']:r['teacher_tokens_equal_current_Base'] for r in teachers};cross=[]
  for n in (19,45):
   owners={role:defaultdict(list) for role in ('H_fit','U_fit')}
   for t in stream['tasks'][:n]:
    for role in owners:
     for row in t[role]:owners[role][query_id(row)].append(t['order'])
   for q in sorted(set(owners['H_fit'])&set(owners['U_fit'])):
    b=bd[q];bc=scores.get(score_key(b['source'],b['output']));cross.append(dict(N=n,query=qi[q],H_owners=owners['H_fit'][q],U_owners=owners['U_fit'][q],Base_correct=bc,teacher_exact=teacher_exact.get(q),H_gold_vs_U_teacher_conflict=bc is False and teacher_exact.get(q) is True))
  table(dest/'CROSS_ROLE_RESPONSIBILITY.csv',cross);write(dest/'TEACHER_REPLAY_SUMMARY.json',dict(checked=len(teachers),exact=sum(r['teacher_tokens_equal_current_Base'] for r in teachers),cross_role_by_N={str(n):dict(QA=sum(r['N']==n for r in cross),confirmed_target_conflicts=sum(r['N']==n and r['H_gold_vs_U_teacher_conflict'] for r in cross)) for n in (19,45)}))
  generated=outputs(dest/'private/run');forced=[];fs=[]
  for n in (19,45):
   for arm in (('E0','E2','R_H') if n==19 else ('E0','E2')):
    for role in ('H_fit','U_fit'):
     rr=[r for r in generated if r['N']==n and r['arm']==arm and r['role']==role];nat={r['query_id']:r for r in rr if r['mode']=='natural'};fo=[r for r in rr if r['mode']=='forced'];pairs=[]
     for r in fo:
      b=nat[r['query_id']];o=next(x for x in slots if x['N']==n and x['role']==role and x['query_id']==r['query_id'] and x['owner_position']==r['owner_position']);fc,nc,bc=score(r),score(b),score(r,'Base');v=dict(N=n,arm=arm,role=role,query=qi[r['query_id']],owner=r['owner_position'],natural_winner=b['selected_position'],any_owner_wins=o['winner_is_any_owner'],Base=bc,natural=nc,forced=fc,supervision_present=r['supervision_present']);forced.append(v);pairs.append((r,v))
     for subset in ('all','displaced_any_owner'):
      pp=[(r,v) for r,v in pairs if subset=='all' or not v['any_owner_wins']];by=defaultdict(list)
      for r,v in pp:by[r['query_id']].append((r,v))
      complete=all(type(v[k]) is bool for _,v in pp for k in ('Base','natural','forced'));sources=defaultdict(list)
      for q,qq in by.items():sources[qq[0][0]['source']['source_group']].append(mean(float(v['forced'])-float(v['natural']) for _,v in qq) if complete else None)
      fs.append(dict(N=n,arm=arm,role=role,subset=subset,owner_pairs=len(pp),unique_QA=len(by),source_count=len(sources),complete=complete,rescues=sum(v['forced'] is True and v['natural'] is False for _,v in pp),harms=sum(v['forced'] is False and v['natural'] is True for _,v in pp),source_macro_delta=mean(mean(x) for x in sources.values()) if complete and sources else None,supervision_present=not (arm=='E0' and role=='H_fit')))
  base_compare=[]
  for n in (19,45):
   for arm in (('E0','E2','R_H') if n==19 else ('E0','E2')):
    for role in ('H_fit','U_fit'):
     for stratum in ('all','Base_correct_retention','Base_wrong_fix'):
      pp=[v for v in forced if v['N']==n and v['arm']==arm and v['role']==role and (stratum=='all' or v['Base']==(stratum=='Base_correct_retention'))];groups=defaultdict(list)
      for v in pp:groups[v['query']].append(v)
      for mode in ('Base','natural','forced'):
       sv=defaultdict(list)
       for q,vs in groups.items():
        source=next(r['source_group'] for r in slots if qi[r['query_id']]==q);vv=[v[mode] for v in vs];sv[source].append(mean(vv) if all(type(v) is bool for v in vv) else None)
       valid=all(v is not None for vs in sv.values() for v in vs);base_compare.append(dict(N=n,arm=arm,role=role,stratum=stratum,mode=mode,unique_QA=len(groups),sources=len(sv),QA_macro=mean(v for vs in sv.values() for v in vs) if sv and valid else None,source_macro=mean(mean(vs) for vs in sv.values()) if sv and valid else None))
  table(dest/'BASE_NATURAL_FORCED_METRICS.csv',base_compare)
  table(dest/'FORCED_DIAGNOSTIC_PAIRS.csv',forced);write(dest/'FORCED_DIAGNOSTIC_SUMMARY.json',dict(rows=fs,diagnostic_only=True,trained_support_not_generalization=True,forced_multiple_owners='All owners, no best-owner oracle; QA then source equal weighting'))
  enough=all(r['complete'] for r in fs);relevant=[r for r in fs if r['N']==19 and r['arm']=='R_H' and r['role']=='H_fit' and r['subset']=='displaced_any_owner'];v=relevant[0];supported=enough and v['source_macro_delta'] is not None and v['source_macro_delta']>0 and v['rescues']>v['harms'];write(dest/'CONCLUSION.json',dict(complete=enough,responsibility_mismatch_functionally_supported_on_training_support=supported,alignment_experiment_start_conditions='NOT_MET' if not supported else 'MECHANISTIC_SUPPORT_ONLY_NOT_PRE_REGISTERED_FOR_LAUNCH',Stage24E_started=False,Stage25_started=False,HSIC='Mechanical effect confirmed; no Stage24C DEV semantic improvement',limits=['Trained-support forced diagnosis is not held-out method performance','Historical controls retain original training backend','No evidence that changing routing preserves native or U; no automatic next experiment']))
if __name__=='__main__':
 assert metrics([('a',True),('a',False),('b',True)])['source_macro']==.75
 assert metrics([('a',None)])['accuracy'] is None
 run(sys.argv[1])
