"""Score all frozen conditions, separate shared repairs from MDU-specific gain."""
import importlib.util,json,os,sys,time,traceback
from pathlib import Path

def main(cfg):
 w=Path(cfg['worktree']);control=Path(cfg['control']);private=Path(cfg['private_root']);report=Path(cfg['report'])
 (control/'CLOSEOUT_ONCE').touch(exist_ok=False)
 sys.path.insert(0,str(w/'experiments/med_t5_core_v1'))
 spec=importlib.util.spec_from_file_location('long_ctl',w/'experiments/padchest_longitudinal_v1/controller.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 spec=importlib.util.spec_from_file_location('long_metrics',w/'experiments/padchest_longitudinal_v1/metrics.py');metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
 history=list(cfg['history_roots']);done={};cost={};failure=False
 inputs=ctl.remote(cfg,'from pathlib import Path\nprint((Path('+repr(cfg['corrected_root'])+')/"T5_INPUTS.private.json").read_text())')
 base=ctl.read(Path(cfg['corrected_private_root'])/'BASE_MASK_FROZEN.private.json')['query_core']
 conditions=['ROUTE_ONLY','CONDITIONAL_CE','CONDITIONAL_MDU','CE_OLD_ROUTE','MDU_OLD_ROUTE']
 while len(done)<len(conditions):
  ctl.budget(cfg)
  names=['FAILURE.private.json','GUARD_FAILURE.private.json','DATA_ADMISSION.json','TRAIN_ONLY_ADMISSION.json','TRAIN_COMPLETE.json','GENERATION_COMPLETE.json','GUARD_EXIT.json']+[c+'/GENERATION_COMPLETE.json' for c in conditions]
  st=ctl.state(cfg,names);ctl.write(private/'LIVE_STATE.private.json',st)
  for condition in conditions:
   if condition in done or not st[condition+'/GENERATION_COMPLETE.json']:continue
   values=ctl.fetch(cfg,condition+'/outputs');assert len(values)==80
   ctl.write(private/(condition+'_OUTPUTS.private.json'),values)
   decisions,cost[condition]=ctl.score(cfg,condition,values,history)
   panels=metrics.summarize(values,decisions,inputs,base);rows=[v for v in values if v['mode']=='bank' and v['role']=='SAME_STATE']
   consumer={v['consumer_id']:v for v in values};packet=ctl.read(control/condition/'PACKET_PRIVATE.json');d=ctl.read(control/condition/'DECISIONS.json')['decisions'];core={v['consumer_id']:d.get(v['key'],{}).get('core_correct') for v in packet['consumers']}
   qmap={(x['edit_id'],x['query']['query_id']):x['query'] for x in inputs['plan']};tmap={t['edit_id']:t for t in inputs['tasks']}
   routed_correct=sum(v['route']['activated'] and core[v['consumer_id']] is True for v in rows)
   cross=sum(v['route']['activated'] and tmap[v['effective_expert']]['label']!=qmap[(v['edit_id'],v['query_id'])]['label'] for v in values if v['mode']=='bank' and v['role'] in ['CONTRAST','SAME_STATE'])
   done[condition]=dict(panels=panels,cost=cost[condition],bank_same_state_probe_ON_and_correct=routed_correct,bank_cross_finding_routes=cross)
   ctl.write(report/(condition+'_RESULTS.json'),done[condition])
  if st['FAILURE.private.json'] or st['GUARD_FAILURE.private.json']:
   failure=True;ctl.write(private/'FAILURE_CAPTURE.private.json',st['FAILURE.private.json'] or st['GUARD_FAILURE.private.json']);break
  if len(done)<len(conditions):time.sleep(20)
 if not failure:
  while True:
   st=ctl.state(cfg,['GUARD_EXIT.json','GENERATION_COMPLETE.json','TRAIN_COMPLETE.json','DATA_ADMISSION.json','TRAIN_ONLY_ADMISSION.json'])
   if st['GUARD_EXIT.json']:assert st['GUARD_EXIT.json']['code']==0;break
   ctl.budget(cfg);time.sleep(5)
  code='from pathlib import Path\nimport sys\nsys.stdout.buffer.write((Path('+repr(cfg['remote_root'])+')/"CONDITIONAL_CANONICAL.private.pt").read_bytes())'
  import subprocess
  copy=subprocess.run(['ssh','-o','BatchMode=yes','pro5000',cfg['remote_python'],'-'],input=code.encode(),capture_output=True,timeout=300)
  assert copy.returncode==0 and 0<len(copy.stdout)<256*1024**2
  with (private/'CONDITIONAL_CANONICAL_BACKUP.pt').open('xb') as f:f.write(copy.stdout)
 criteria={}
 if 'CONDITIONAL_MDU' in done and 'CONDITIONAL_CE' in done:
  m,c=[done[k]['panels']['bank'] for k in ['CONDITIONAL_MDU','CONDITIONAL_CE']]
  criteria=dict(source_all_confirmed_correct=m['roles']['T0']['correct']==4,unseen_paraphrases_all_confirmed_correct=m['roles']['PARAPHRASE']['correct']==8,
   at_least_one_same_state_probe_ON_and_correct=done['CONDITIONAL_MDU']['bank_same_state_probe_ON_and_correct']>0,
   macro_lower_exceeds_matched_CE_upper=m['probe_balanced_accuracy_pct']['lower_pct']>c['probe_balanced_accuracy_pct']['upper_pct'],
   no_cross_finding_route=done['CONDITIONAL_MDU']['bank_cross_finding_routes']==0)
 criteria['all_predeclared_pilot_advantage_gates_pass']=bool(criteria) and all(criteria.values())
 result=dict(status='INCOMPLETE_ENGINEERING_FAILURE' if failure else 'COMPLETE_WITH_UNKNOWN' if any(v['unknown_payloads'] for v in cost.values()) else 'COMPLETE',
  conditions=done,MDU_specific_pilot_gates=criteria,original_MDU_reference=ctl.read(Path(cfg['corrected_report'])/'MDU_RESULTS.json')['panels'],independent_confirmation=False,
  interpretation='Shared routing/projection gains are not attributed to the MDU objective. Four source and six probe patients are exposed DEV; this pilot cannot establish general superiority.')
 ctl.write(report/'RESULTS.json',result)
 ctl.write(report/'EXECUTION_AUDIT.json',dict(clock=cfg['clock'],completed_conditions=list(done),failure=failure,DATA_ADMISSION=st.get('DATA_ADMISSION.json'),TRAIN_ONLY_ADMISSION=st.get('TRAIN_ONLY_ADMISSION.json'),TRAIN_COMPLETE=st.get('TRAIN_COMPLETE.json'),old_states_unchanged=True,new_Judge_policy='Exact accepted reuse, inherited UNKNOWN unchanged, no automatic retry'))
 sys.path.insert(0,str(w/'experiments/med_t5_data_v1'));from job import publish
 publish(cfg);ctl.write(control/'COMPLETE.json',dict(time=time.time(),failure=failure))
if __name__=='__main__':
 cfg=json.loads(Path(os.environ['LONG_LOCAL_CONFIG']).read_text())
 try:main(cfg)
 except BaseException:(Path(cfg['private_root'])/'CLOSEOUT_FAILURE.private.json').write_text(json.dumps(dict(time=time.time(),error=traceback.format_exc())));raise
