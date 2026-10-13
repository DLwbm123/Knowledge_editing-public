"""Fixed CPU provenance audit of executed TXT; no new model, labels or Judge."""
import os,sys,json,time,collections,unicodedata,traceback
from pathlib import Path
ROOT=Path(os.environ.get('RUN_ROOT','/tmp'));SOURCE=Path(os.environ.get('SOURCE_ROOT','/tmp'))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def canon(q):return ' '.join(unicodedata.normalize('NFKC',q).casefold().split())
def identity(r):return tuple(r[k] for k in ['mode','prefix','edit','task','input_id'])
def origin(q,native,fit):return 'native_and_fit' if q==native and q in fit else 'native_only' if q==native else 'fit_only' if q in fit else 'unmatched'
def summary(rows):
 return dict(occurrences=len(rows),unique_inputs=len({r['input_id'] for r in rows}),unique_input_winner_pairs=len({(r['input_id'],r['winner']) for r in rows}),missing_occurrences=sum(r['missing'] for r in rows),origins=dict(collections.Counter(r['origin'] for r in rows)),same_image=sum(r['same_image'] is True for r in rows),cross_image=sum(r['same_image'] is False for r in rows),same_source_group=sum(r['same_source'] is True for r in rows),tasks=dict(collections.Counter(r['task'] for r in rows)),known_R0_to_TXT_rescue=sum(r['R0_to_TXT']=='wrong_correct' for r in rows),known_R0_to_TXT_damage=sum(r['R0_to_TXT']=='correct_wrong' for r in rows),Base_counterfactual=dict(collections.Counter(r['Base_counterfactual'] for r in rows)),global_question_owner_multiplicity=dict(collections.Counter(str(r['global_owners']) for r in rows)),current_bank_question_owner_multiplicity=dict(collections.Counter(str(r['bank_owners']) for r in rows)))
def selfcheck():
 assert canon('  Ａ?  ')=='a?' and canon('a')!='a?'
 assert origin('a','a',{'a'})=='native_and_fit' and origin('b','a',{'b'})=='fit_only'
 assert origin('c','a',{'b'})=='unmatched'
 a=dict(input_id='x',winner='e',missing=True,origin='native_only',same_image=False,same_source=False,task='T1G',R0_to_TXT='missing',Base_counterfactual='missing',global_owners=2,bank_owners=1)
 s=summary([a,a]);assert s['occurrences']==2 and s['unique_inputs']==1 and s['missing_occurrences']==2

def main():
 started=time.time();selfcheck();config=read(ROOT/'public/ROUTER_CONFIG.json');manifest=read(ROOT/'RUN_MANIFEST.json');assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert read(SOURCE/'RUN_STATUS.json')['status']=='COMPLETE'
 sys.path[:0]=[str(SOURCE),str(SOURCE/'source_patch'),str(SOURCE/'source')]
 # The inherited worker module reads CONFIG_LOCK at import; bind its read-only source.
 previous=os.environ['RUN_ROOT'];os.environ['RUN_ROOT']=str(SOURCE)
 try:
  from worker_v3 import input_id
  from judge_protocol import read_scores
 finally:os.environ['RUN_ROOT']=previous
 prior_CPU=read(ROOT/'INITIALIZATION_RECOVERY.json')['failed_CPU_wall_upper_bound'] if (ROOT/'INITIALIZATION_RECOVERY.json').exists() else 0
 tasks=read(SOURCE/'private/TASKS_R2_LOCKED.json')['tasks'];owners={t['canonical_edit_id']:t for t in tasks};bg={canon(r['question']) for r in read(SOURCE/'private/U_bg.json')['rows']};supports={};fit={};natives={}
 for e,t in owners.items():
  natives[e]=canon(t['native']['question']);fit[e]={canon(q) for q in t['semantic_fit_questions']};supports[e]=({natives[e]}|fit[e])-bg
 assert sum(map(len,supports.values()))==240
 raw={input_id(r):r for t in tasks for r in t['evaluation']};raw.update({input_id(r):r for r in read(SOURCE/'private/LOCALITY_STRESS_HOLDOUT.json')['rows']})
 scores=read_scores(SOURCE);masks={(r['edit'],r['task'],r['query_id']):r['base_correct'] for r in read(SOURCE/'private/BASE_MASKS.json')['rows']};metrics=[]
 write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase='PROVENANCE',epoch=started))
 for phase,expected in config['panels'].items():
  rows=[]
  for j in read(SOURCE/'QUEUE.json'):
   assert j['status']=='COMPLETE'
   if j['phase']==phase:
    for p in (SOURCE/'jobs'/j['id']).rglob('CONSUMERS.json'):rows.extend(read(p))
  ref={identity(r):r for r in rows if r['arm']=='R0'};txt={identity(r):r for r in rows if r['arm']=='TXT'};assert ref.keys()==txt.keys() and len(ref)==expected
  cohort=[t['canonical_edit_id'] for t in tasks if (t['order']<=24)==(phase=='DEV')]
  for key,a in ref.items():
   assert time.time()<manifest['deadline_epoch'] and time.time()-started+prior_CPU<config['CPU_task_seconds_limit'];assert not (ROOT/'STOP').exists()
   b=txt[key];winner=a['route']['logical_edit_id'];r=raw[a['input_id']];q=canon(r['question']);protected=winner is not None and q in supports[winner];d=b['route_diagnostics'];lexical=winner is not None and d['q']<0 and protected
   expected_winner=winner if winner is not None and (d['q']>=0 or protected) else None
   assert b['route']['logical_edit_id']==expected_winner and bool(d.get('text_guard',False))==protected
   assert b['judge_key']==(a['judge_key'] if expected_winner is not None or winner is None else a['base_judge_key'])
   x=scores.get(a['judge_key']);y=scores.get(b['judge_key']);base=scores.get(b['base_judge_key']);missing=x is None or y is None
   transition='missing' if missing else 'wrong_correct' if not x and y else 'correct_wrong' if x and not y else 'both_correct' if y else 'both_wrong'
   cf='not_lexical_only' if not lexical else 'missing' if y is None or base is None else 'benefit' if y and not base else 'harm' if base and not y else 'both_correct' if y else 'both_wrong'
   active_bank=[a['edit']] if a['mode']=='single' else cohort[:a['prefix']]
   metrics.append(dict(phase=phase,mode=a['mode'],prefix=a['prefix'],task=a['task'],input_id=a['input_id'],winner=winner,origin=origin(q,natives[winner],fit[winner]) if winner else 'R0_OFF',lexical_only_retain=lexical,missing=missing,Base_counterfactual=cf,R0_to_TXT=transition,same_image=r['image_sha256']==owners[winner]['native']['image_sha256'] if winner else None,same_source=r['source_group']==owners[winner]['native']['source_group'] if winner else None,global_owners=sum(q in s for s in supports.values()),bank_owners=sum(q in supports[e] for e in active_bank),locality_qualified=masks.get((a['edit'],a['task'],a['query_id'])) is True if a['task'] in ['T1L','T2L'] else None))
 assert len(metrics)==2047
 write(ROOT/'private/POINTWISE_PROVENANCE.json',metrics)
 public=dict(status='COMPLETE_EXPOSED_DIAGNOSTIC',consumers=len(metrics),all=summary(metrics),by_phase={phase:dict(all=summary([r for r in metrics if r['phase']==phase]),lexical_only_retain=summary([r for r in metrics if r['phase']==phase and r['lexical_only_retain']]),by_mode={mode:summary([r for r in metrics if r['phase']==phase and r['mode']==mode]) for mode in ['single','sequential','EXPOSED_REGRESSION']}) for phase in config['panels']},support_question_ambiguity=dict(unique_canonical_questions=len(set().union(*supports.values())),shared_by_multiple_experts=sum(sum(q in s for s in supports.values())>1 for q in set().union(*supports.values())),not_clinical_scope_labels=True),independent_CONFIRM=False,new_verified_CAL=0,new_router_or_threshold_fit=False,cached_Base_counterfactual_not_executed_REG_NEG0=True)
 write(ROOT/'public/PROVENANCE_AGGREGATES.json',public)
 seconds=time.time()-started;assert seconds+prior_CPU<config['CPU_task_seconds_limit'];assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<config['storage_bytes_limit']
 write(ROOT/'RESOURCE_LEDGER.json',dict(inherited_Judge_attempts=read(SOURCE/'RESOURCE_LEDGER.json')['judge_submission_attempt_items'],new_Judge_attempts=0,GPU_hours=0,CPU_seconds=seconds,failed_CPU_wall_upper_bound=prior_CPU,new_generation=0,training_steps=0))
 write(ROOT/'public/FINAL_EXECUTION_AUDIT.json',dict(status='COMPLETE',consumers=2047,bindings_and_route_formula='PASS',CPU_seconds=seconds,failed_CPU_wall_upper_bound=prior_CPU,GPU_hours=0,new_Judge_attempts=0,new_generation=0,training_steps=0,new_verified_CAL=0,independent_CONFIRM=False,public_delivery='PENDING_GITHUB'))
 text='# TXT词面迁移诊断结果\n\n完整2047暴露消费者完成来源与路由公式审计；CPU-only，无训练、生成、Judge或新增CAL。\n\n'
 for phase in config['panels']:
  z=public['by_phase'][phase]['lexical_only_retain'];text+=f'{phase}词面单独保留：{z["occurrences"]}次出现，{z["unique_inputs"]}个不同输入，{z["unique_input_winner_pairs"]}个输入/winner组合；来源{z["origins"]}；同图{z["same_image"]}、跨图{z["cross_image"]}；缓存Base反事实{z["Base_counterfactual"]}。\n\n'
 text+='完整分母、missing和歧义计数见PROVENANCE_AGGREGATES.json。多prefix出现不是独立样本量；Base反事实不是新执行的REG NEG0；source group隔离不是患者独立。native/S_fit严格词面迁移不能代替真实verified医学scope标注，暴露面板PASS不能声称独立确认或临床可部署。下一步需要合法、带来源验证且角色隔离的同图/跨图scope标注；没有这类材料时保持小时监测，不空转GPU或追加无依据搜索。私有QA、逐题诊断及权重不发布。\n'
 (ROOT/'public/FINAL_RESULTS_ZH.md').write_text(text);write(ROOT/'RUN_STATUS.json',dict(status='COMPLETE',phase='CLOSED',next_status='NEEDS_VERIFIED_SCOPE_CALIBRATION',epoch=time.time()))
 print(json.dumps(public['by_phase']))
if __name__=='__main__':
 if '--selfcheck' in sys.argv:selfcheck();print('PASS: exact lexical provenance, unique denominator and missing retention')
 else:
  try:main()
  except Exception as e:
   write(ROOT/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()));write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',phase='PROVENANCE'));raise
