"""Collect only the previously unstarted LoRA arm; retain original unknowns."""
import importlib.util,json,os,sys,time,traceback
from pathlib import Path

def main(cfg):
 w=Path(cfg['worktree']);control=Path(cfg['control']);private=Path(cfg['private_root']);report=Path(cfg['report'])
 (control/'LORA_REPAIR1_CLOSEOUT_ONCE').touch(exist_ok=False)
 sys.path.insert(0,str(w/'experiments/med_t5_core_v1'))
 spec=importlib.util.spec_from_file_location('long_controller',w/'experiments/padchest_longitudinal_v1/controller.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 original=ctl.read(report/'RESULTS.json');assert original['methods_missing']==['LORA']
 ctl.write(private/'PRE_REPAIR_RESULTS.private.json',original)
 while True:
  ctl.budget(cfg);state=ctl.state(cfg,['LORA/T5/GENERATION_COMPLETE.json','LORA_REPAIR1_EXIT.json','LORA_REPAIR1_FAILURE.private.json','LORA_REPAIR1_HEALTH.json'])
  ctl.write(private/'REPAIR_LIVE_STATE.private.json',state)
  if state['LORA_REPAIR1_FAILURE.private.json']:raise RuntimeError('LoRA dependency repair failed; original evidence retained')
  if state['LORA/T5/GENERATION_COMPLETE.json'] and state['LORA_REPAIR1_EXIT.json']:
   assert state['LORA_REPAIR1_EXIT.json']['code']==0;break
  time.sleep(20)
 values=ctl.fetch(cfg,'LORA/T5/outputs');assert len(values)==80
 ctl.write(private/'LORA_OUTPUTS.private.json',values)
 inputs=ctl.remote(cfg,'from pathlib import Path\nprint((Path('+repr(cfg['remote_root'])+')/"T5_INPUTS.private.json").read_text())')
 history=cfg['history_roots']+[str(control/n) for n in ['base','MDU','BALANCEDIT','GRACE']]
 decisions,cost=ctl.score(cfg,'LORA',values,history)
 spec=importlib.util.spec_from_file_location('long_metrics',w/'experiments/padchest_longitudinal_v1/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 base=ctl.read(private/'BASE_MASK_FROZEN.private.json')['query_core'];panels=m.summarize(values,decisions,inputs,base)
 ctl.write(report/'LORA_RESULTS.json',dict(panels=panels,cost=cost,independent_confirmation=False,dependency_repair='Existing PEFT0.15.2; no scientific configuration change'))
 original['methods']['LORA']=panels;original['cost']['LORA']=cost;original['methods_missing']=[]
 original['status']='COMPLETE_WITH_UNKNOWN' if any(v['unknown_payloads'] for v in original['cost'].values()) else 'COMPLETE'
 original['historical_engineering_failure']='LoRA missing PEFT before any training or generation; repaired with existing read-only PEFT0.15.2 in original clock'
 ctl.write(report/'RESULTS.json',original)
 audit=ctl.read(report/'EXECUTION_AUDIT.json');audit.update(complete_methods=list(original['methods']),failure=False,historical_failure_retained=True,dependency_repair=True,repair_completed_at=time.time())
 ctl.write(report/'EXECUTION_AUDIT.json',audit)
 sys.path.insert(0,str(w/'experiments/med_t5_data_v1'));from job import publish
 publish(cfg);ctl.write(control/'LORA_REPAIR1_CLOSEOUT_COMPLETE.json',dict(time=time.time(),method='LORA'))
if __name__=='__main__':
 cfg=json.loads(Path(os.environ['LONG_LOCAL_CONFIG']).read_text())
 try:main(cfg)
 except BaseException:
  (Path(cfg['private_root'])/'REPAIR_CLOSEOUT_FAILURE.private.json').write_text(json.dumps(dict(time=time.time(),error=traceback.format_exc())));raise
