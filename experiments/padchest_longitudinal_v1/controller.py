"""Finite Base CORE admission, four-method scoring and aggregate publication."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def remote(cfg,code,timeout=120):
 v=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',cfg['remote_python'],'-'],input=code,text=True,capture_output=True,timeout=timeout)
 assert v.returncode==0,v.stderr[-4000:]
 return json.loads(v.stdout)
def state(cfg,names):
 return remote(cfg,'from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')\nprint(json.dumps({n:json.loads((r/n).read_text()) if (r/n).exists() else None for n in '+repr(names)+'}))')
def fetch(cfg,sub):
 return remote(cfg,'from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')/'+repr(sub)+'\nprint(json.dumps([json.loads(p.read_text()) for p in sorted(r.glob("*.json"))]))',240)
def budget(cfg):
 assert time.time()<cfg['clock']['hard_end']-660
 used=0
 for base in [Path(cfg['control']),Path(cfg['private_root'])]:
  for directory,dirs,files in os.walk(base,followlinks=False):
   for name in files:
    p=Path(directory)/name
    try:
     if not p.is_symlink():used+=p.stat().st_blocks*512
    except FileNotFoundError:pass
 assert used+256*1024**2<2_000_000_000
 write(Path(cfg['control'])/'HEALTH.json',dict(time=time.time(),bytes=used))
def score(cfg,name,values,history):
 from controller import packet
 control=Path(cfg['control']);folder=control/name;folder.mkdir()
 durable=Path(cfg['private_root'])/name;durable.mkdir()
 payload=packet(cfg,values,history);write(folder/'PACKET_PRIVATE.json',payload)
 settings=dict(cfg,root=str(folder),private_root=str(durable),clock=str(control/'clock.json'),neutral_entry=str(control/'s.py'),stage='long')
 write(folder/'config.json',settings)
 env=dict(os.environ,T4_CONFIG=str(folder/'config.json'),T4_MODE='guard',CURRENT_SOURCE=str(control/'score.py'))
 with (folder/'guard.log').open('x') as log:
  result=subprocess.run([sys.executable,str(control/'s.py')],env=env,cwd=folder,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,timeout=max(1,int(cfg['clock']['hard_end']-time.time()-120)))
 assert result.returncode==0 and read(folder/'GUARD_EXIT.json')['code']==0
 dec=read(folder/'DECISIONS.json');core={v['consumer_id']:dec['decisions'].get(v['key'],{}).get('core_correct') for v in payload['consumers']}
 for n in ['PACKET_PRIVATE.json','DECISIONS.json','COMPLETE.json','GUARD_EXIT.json','COST.json']:
  if (folder/n).exists():write(durable/n,read(folder/n))
 history.append(str(folder));return core,dict(new_attempt_items=read(folder/'COMPLETE.json')['new_attempt_items'],unknown_payloads=len(dec['unknown']))
def main(cfg):
 control=Path(cfg['control']);private=Path(cfg['private_root']);report=Path(cfg['report']);w=Path(cfg['worktree'])
 (control/'CONTROLLER_ONCE').touch(exist_ok=False);signal.alarm(max(1,int(cfg['clock']['hard_end']-time.time()-60)))
 sys.path.insert(0,str(w/'experiments/med_t5_core_v1'))
 history=list(cfg['history_roots']);done={};cost={};base_core={};source=read(private/'SOURCE_INPUTS.private.json');qualified=False
 while True:
  budget(cfg)
  st=state(cfg,['BASE_COMPLETE.json','BASE_FAILURE.json','GUARD_FAILURE.private.json','COHORT_FROZEN.json','CPU_ADMISSION.json','MDU/GENERATION_COMPLETE.json',
    'BALANCEDIT/T5/GENERATION_COMPLETE.json','GRACE/T5/GENERATION_COMPLETE.json','LORA/T5/GENERATION_COMPLETE.json','GENERATION_COMPLETE.json','COMPLETE.json'])
  write(private/'LIVE_STATE.private.json',st)
  if st['BASE_COMPLETE.json'] and not qualified:
   values=fetch(cfg,'base_outputs');assert len(values)==48
   write(private/'BASE_OUTPUTS.private.json',values);dec,cost['BASE']=score(cfg,'base',values,history)
   base_core={v['query_id']:dec[v['consumer_id']] for v in values}
   native={t['edit_id']:base_core[t['native']['query_id']] for t in source['tasks']}
   mask=dict(native_core=native,query_core=base_core,no_replacement=True)
   write(private/'BASE_MASK_FROZEN.private.json',mask)
   remote(cfg,'from pathlib import Path\nimport json\np=Path('+repr(cfg['remote_root'])+')/"BASE_MASK_FROZEN.private.json"\nwith p.open("x") as f:json.dump('+repr(mask)+',f)\nprint("{}")')
   write(report/'BASE_QUALIFICATION.json',dict(candidates=6,confirmed_wrong=sum(v is False for v in native.values()),correct=sum(v is True for v in native.values()),unknown=sum(v is None for v in native.values()),
      probe_correct=sum(base_core[q['query_id']] is True for q in source['probes']),probe_n=6,source_class_support={s:sum(v is False and t['progression']==s for t in source['tasks'] for k,v in native.items() if k==t['edit_id']) for s in ['improving','stable','worsening']},**cost['BASE']))
   qualified=True
  for method in ['MDU','BALANCEDIT','GRACE','LORA']:
   key='MDU/GENERATION_COMPLETE.json' if method=='MDU' else method+'/T5/GENERATION_COMPLETE.json'
   if st[key] and method not in done:
    values=fetch(cfg,method+'/outputs' if method=='MDU' else method+'/T5/outputs');assert values
    task_inputs=remote(cfg,'from pathlib import Path\nimport json\nprint((Path('+repr(cfg['remote_root'])+')/"T5_INPUTS.private.json").read_text())')
    assert len(values)==2*len(task_inputs['plan'])
    write(private/(method+'_OUTPUTS.private.json'),values)
    decisions,cost[method]=score(cfg,method,values,history)
    # Put this task's pure metric module before the generic legacy modules.
    import importlib.util
    spec=importlib.util.spec_from_file_location('long_metrics',w/'experiments/padchest_longitudinal_v1/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    done[method]=m.summarize(values,decisions,task_inputs,base_core)
    write(report/(method+'_RESULTS.json'),dict(panels=done[method],cost=cost[method],independent_confirmation=False))
    if method=='MDU':
     code='from pathlib import Path\nimport sys\nsys.stdout.buffer.write((Path('+repr(cfg['remote_root'])+')/"MDU/T5_CANONICAL.pt").read_bytes())'
     copy=subprocess.run(['ssh','-o','BatchMode=yes','pro5000',cfg['remote_python'],'-'],input=code.encode(),capture_output=True,timeout=180)
     assert copy.returncode==0 and 0<len(copy.stdout)<20_000_000
     with (private/'MDU_CANONICAL_BACKUP.pt').open('xb') as f:f.write(copy.stdout)
  if st['GUARD_FAILURE.private.json'] or st['BASE_FAILURE.json'] or st['COMPLETE.json'] or len(done)==4:break
  time.sleep(20)
 failure=st['GUARD_FAILURE.private.json'] or st['BASE_FAILURE.json']
 write(report/'RESULTS.json',dict(status='INCOMPLETE_ENGINEERING_FAILURE' if failure else 'NO_CONFIRMED_BASE_ERRORS' if not done else 'COMPLETE_WITH_UNKNOWN' if any(v['unknown_payloads'] for v in cost.values()) else 'COMPLETE',
    methods=done,methods_missing=[m for m in ['MDU','BALANCEDIT','GRACE','LORA'] if m not in done],MEND='MATCHED_META_EDITOR_UNAVAILABLE',cost=cost,
    controls=dict(constant_class_probe_balanced_accuracy_pct=100/3,source_answer_copy_probe_balanced_accuracy_pct=100/3,joint_success_pct=0),independent_confirmation=False))
 if failure:write(private/'FAILURE_CAPTURE.private.json',failure)
 write(report/'EXECUTION_AUDIT.json',dict(clock=cfg['clock'],CPU_admission=st['CPU_ADMISSION.json'],complete_methods=list(done),failure=bool(failure),
    source_qualification_frozen_before_methods=True,old_T5_unchanged=True,source_exposure='DEV',time=time.time()))
 # The generic publisher stages only this report; source was committed before launch.
 sys.path.insert(0,str(w/'experiments/med_t5_data_v1'));from job import publish
 publish(cfg);write(control/'COMPLETE.json',dict(methods=list(done),time=time.time()))
if __name__=='__main__':
 cfg=read(os.environ['LONG_LOCAL_CONFIG'])
 try:main(cfg)
 except BaseException:write(Path(cfg['private_root'])/'CONTROLLER_FAILURE.private.json',dict(error=traceback.format_exc(),time=time.time()));raise
