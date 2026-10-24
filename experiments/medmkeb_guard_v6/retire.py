"""Retire only the twenty new active/final states after complete consumers."""
import os,json,time
from pathlib import Path

def retire(root):
    root=Path(root).resolve();assert root.name=='admission1' and 'medmkeb-guard-v6-' in str(root)
    read=lambda p:json.loads(p.read_text())
    paths=[]
    for i in (0,1):
        part=root/str(i);assert read(part/'EXIT.json')['code']==0 and read(part/'STUDY_COMPLETE.json')['status']=='COMPLETE'
        launch=read(part/'LAUNCH.json');proc=Path('/proc')/str(launch['pid']);assert not proc.exists() or proc.joinpath('stat').read_text().split()[21]!=str(launch['start_ticks'])
        frozen=read(root/f'FROZEN_{i}.json');assert len(frozen['arms'])==(6 if i==0 else 4)
        for item in frozen['arms']:
            task=next(t for t in read(root/'DATA_ADMISSION.json')['tasks'] if t['key']==item['task']);folder=part/task['key'];arm=item['arm'];assert read(folder/(arm+'_TRAIN.json'))['save_load_exact']
            expected={x['query_id'] for x in read(root/'EVAL_ROWS.json')}|{task['rephrase']['query_id']};outputs=list((folder/'outputs'/arm).glob('*.json'));assert {f.stem for f in outputs}==expected and all(read(f)['raw_token_ids'] for f in outputs)
            paths.extend([folder/('active_'+arm+'.pt'),folder/'finals'/(arm+'.pt')])
    assert len(paths)==20 and all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) for p in paths)
    receipt=dict(time=time.time(),files=[dict(path=str(p),bytes=p.stat().st_size) for p in paths],historical_assets_touched=False,weights_reconstruction_requires_training=True,reason='Both EXIT0 and all ten save/load plus 320 generation consumers complete')
    receipt['total_bytes']=sum(x['bytes'] for x in receipt['files']);(root/'RETIREMENT_PLAN.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for p in paths:p.unlink()
    (root/'RETIREMENT_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n');return dict(files=len(paths),bytes=receipt['total_bytes'])
if __name__=='__main__':print(json.dumps(retire(os.environ['OUTPUT_ROOT'])))
