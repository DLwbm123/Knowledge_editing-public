"""Finite collection/publication dependency; no new experiment scheduling."""
import json,os,sys,time,traceback
from pathlib import Path

def main(cfg):
 control=Path(cfg['control']);private=Path(cfg['private_root']);report=Path(cfg['report']);w=Path(cfg['worktree'])
 (control/'CLOSEOUT_ONCE').touch(exist_ok=False)
 import importlib.util
 spec=importlib.util.spec_from_file_location('long_ctl',w/'experiments/padchest_longitudinal_v1/controller.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 while True:
  ctl.budget(cfg);st=ctl.state(cfg,['SUMMARY.json','GUARD_EXIT.json','GUARD_FAILURE.private.json','FAILURE.private.json','PROGRESS.json'])
  ctl.write(private/'LIVE_STATE.private.json',st)
  if st['GUARD_FAILURE.private.json'] or st['FAILURE.private.json']:
   ctl.write(report/'SUMMARY.json',dict(status='INCOMPLETE_ENGINEERING_FAILURE',progress=st['PROGRESS.json']));break
  if st['SUMMARY.json'] and st['GUARD_EXIT.json']:
   assert st['GUARD_EXIT.json']['code']==0;ctl.write(report/'SUMMARY.json',st['SUMMARY.json']);break
  time.sleep(15)
 sys.path.insert(0,str(w/'experiments/med_t5_data_v1'));from job import publish
 publish(cfg);ctl.write(control/'COMPLETE.json',dict(time=time.time()))
if __name__=='__main__':
 cfg=json.loads(Path(os.environ['LONG_LOCAL_CONFIG']).read_text())
 try:main(cfg)
 except BaseException:(Path(cfg['private_root'])/'CLOSEOUT_FAILURE.private.json').write_text(json.dumps(dict(time=time.time(),error=traceback.format_exc())));raise
