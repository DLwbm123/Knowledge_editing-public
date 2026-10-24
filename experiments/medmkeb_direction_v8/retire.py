"""Retire eight run-owned active/final states after the final trajectory consumer."""
import os,json,time
from pathlib import Path

def retire(root):
    root=Path(root).resolve();assert root.name=='admission1' and 'medmkeb-direction-v8-' in str(root)
    read=lambda p:json.loads(p.read_text());paths=[]
    for i in (0,1):
        p=root/str(i);assert read(p/'EXIT.json')['code']==0;assert read(p/'TRAJECTORY_COMPLETE.json')['status'] in ('COMPLETE','NO_FEASIBLE_LEARNING_STEP')
        assert read(p/'PROJECT_COMPLETE.json')['save_load_exact'] and read(p/'NORM_COMPLETE.json')['save_load_exact']
        launch=read(p/'LAUNCH.json');proc=Path('/proc')/str(launch['pid']);assert not proc.exists() or proc.joinpath('stat').read_text().split()[21]!=str(launch['start_ticks'])
        paths.extend([p/('active_'+arm+'.pt') for arm in ('PROJECT','NORM_SOURCE')]+[p/('final_'+arm+'.pt') for arm in ('PROJECT','NORM_SOURCE')])
    assert len(paths)==8 and all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) for p in paths)
    receipt=dict(time=time.time(),files=[dict(path=str(p),bytes=p.stat().st_size) for p in paths],reason='Both EXIT0, PROJECT and matched prefix metrics/save-load and all registered consumers complete; no generation dependency',historical_assets_touched=False,reconstruction_requires_training=True);receipt['total_bytes']=sum(x['bytes'] for x in receipt['files'])
    (root/'RETIREMENT_PLAN.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for p in paths:p.unlink()
    (root/'RETIREMENT_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n');return dict(files=len(paths),bytes=receipt['total_bytes'])
if __name__=='__main__':print(json.dumps(retire(os.environ['OUTPUT_ROOT'])))
