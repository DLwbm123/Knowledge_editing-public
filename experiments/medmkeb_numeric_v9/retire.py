"""Four explicit forensic states, only after complete metric consumers."""
import json,os,time
from pathlib import Path

def retire(root):
    root=Path(root).resolve();assert root.name=='admission1' and 'medmkeb-numeric-v9-' in str(root)
    read=lambda p:json.loads(p.read_text());assert read(root/'CONSUMERS_COMPLETE.json')['status']=='COMPLETE_FORENSIC_CONSUMERS';paths=[]
    for i in (0,1):
        p=root/str(i);assert read(p/'EXIT.json')['code']==0 and read(p/'AUDIT_COMPLETE.json')['save_load_exact']
        a=read(p/'LAUNCH.json');proc=Path('/proc')/str(a['pid'])/'stat';assert not proc.exists() or proc.read_text().split()[21]!=str(a['start_ticks'])
        paths.extend(p/x for x in ('active_RECON.pt','AUDIT_INPUTS.pt'))
    assert len(paths)==4 and all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) for p in paths)
    v=dict(time=time.time(),files=[dict(path=str(p),bytes=p.stat().st_size) for p in paths],historical_assets_touched=False,reconstruction_requires_new_training_authorization=True);v['total_bytes']=sum(x['bytes'] for x in v['files']);(root/'RETIREMENT_PLAN.json').write_text(json.dumps(v,indent=2)+'\n')
    for p in paths:p.unlink()
    (root/'RETIREMENT_COMPLETE.json').write_text(json.dumps(v,indent=2)+'\n');return dict(files=4,bytes=v['total_bytes'])
if __name__=='__main__':print(retire(os.environ['OUTPUT_ROOT']))
