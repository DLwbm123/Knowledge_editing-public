"""V11 dependency-ordered four-plus-two retirement for fixed V12 consumer."""
import os,json,time
from pathlib import Path
from answer_check import check

def retire(source,audit,phase):
    source=Path(source).resolve();audit=Path(audit).resolve();assert source.name==audit.name=='admission1' and 'medmkeb-reserve-v11-' in str(source) and 'medmkeb-answer-v12-' in str(audit) and phase in ('PARTIAL','FINAL')
    read=lambda p:json.loads(p.read_text());assert read(source/'CONSUMERS_COMPLETE.json')['status']=='COMPLETE_RESERVE_CONSUMERS';paths=[]
    for i in (0,1):
        p=source/str(i);assert read(p/'EXIT.json')['code']==0 and read(p/'STUDY_COMPLETE.json')['save_load_exact'];a=read(p/'LAUNCH.json');proc=Path('/proc')/str(a['pid'])/'stat';assert not proc.exists() or proc.read_text().split()[21]!=str(a['start_ticks'])
        if phase=='FINAL':
            q=audit/str(i);assert read(q/'EXIT.json')['code']==0 and read(q/'STUDY_COMPLETE.json')['source_final_state_exact'];a=read(q/'LAUNCH.json');proc=Path('/proc')/str(a['pid'])/'stat';assert not proc.exists() or proc.read_text().split()[21]!=str(a['start_ticks'])
            task=read(audit/'DATA_ADMISSION.json')['tasks'][i];ids=[x['query_id'] for x in read(audit/'EVAL_ROWS.json')]+[task['rephrase']['query_id']];assert check(q/'outputs',ids)==32
        paths.extend(p/name for name in (('active.pt','LAST_LINEAR.pt') if phase=='PARTIAL' else ('final.pt',)))
    assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(source) for p in paths)
    v=dict(time=time.time(),phase=phase,files=[dict(path=str(p),bytes=p.stat().st_size) for p in paths],historical_assets_touched=False,reconstruction_requires_new_authorization=True,retained_finals_for_V12=phase=='PARTIAL');v['total_bytes']=sum(x['bytes'] for x in v['files']);(source/('RETIREMENT_'+phase+'_PLAN.json')).write_text(json.dumps(v,indent=2)+'\n')
    if phase=='PARTIAL':
        import torch
        v['final_scalar_optimizer_states']={str(i):torch.load(source/str(i)/'active.pt',map_location='cpu',weights_only=True)['scalar'] for i in (0,1)}
        (source/('RETIREMENT_'+phase+'_PLAN.json')).write_text(json.dumps(v,indent=2)+'\n')
    for p in paths:p.unlink()
    (source/('RETIREMENT_'+phase+'_COMPLETE.json')).write_text(json.dumps(v,indent=2)+'\n');return dict(files=len(paths),bytes=v['total_bytes'],phase=phase)
if __name__=='__main__':print(retire(os.environ['V11_ROOT'],os.environ['V12_ROOT'],os.environ['RETIRE_PHASE']))
