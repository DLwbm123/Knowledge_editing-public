"""Retire only this run's generated states after all declared weight consumers finish."""
import json,time,os
from pathlib import Path

def retire(root):
    root=Path(root).resolve();assert root.name=='admission1' and 'medmkeb-proxy-audit-v1-' in str(root)
    read=lambda p:json.loads(p.read_text())
    assert read(root/'DATA_ADMISSION.json')['status']=='PASS'
    for i in (0,1):
        part=root/str(i);assert read(part/'STUDY_COMPLETE.json')['status']=='COMPLETE' and read(part/'EXIT.json')['code']==0
        launch=read(part/'LAUNCH.json');proc=Path('/proc')/str(launch['pid'])
        assert not proc.exists() or proc.joinpath('stat').read_text().split()[21]!=launch['start_ticks']
        frozen=read(root/f'FROZEN_{i}.json');assert len(frozen['arms'])==8
        for task in read(root/'DATA_ADMISSION.json')['tasks']:
            expected={v['query_id'] for v in read(root/'EVAL_ROWS.json')}|{task['rephrase']['query_id']}
            for arm in [a['arm'] for a in frozen['arms'] if a['task']==task['key']]:
                files=list((part/task['key']/'outputs'/arm).glob('*.json'));assert {p.stem for p in files}==expected
                assert all(read(p)['raw_token_ids'] for p in files)
            assert read(part/task['key']/'BASE_REUSE_ADMISSION.json')['matched']==len(expected)
    for i in (0,1):
        part=root/str(i)
        for task in read(root/'DATA_ADMISSION.json')['tasks']:
            arm='BALANCEDIT' if i==0 else 'BALANCEDIT_CE'
            expected={v['query_id'] for v in read(root/'EVAL_ROWS.json')}|{task['rephrase']['query_id']}
            assert {p.stem for p in (part/task['key']/'routing'/arm).glob('*.json')}==expected
    paths=[root/'NEW_FISHER.pt']
    for i in (0,1):
        part=root/str(i)
        paths+=list(part.glob('E*/active_*.pt'))+list(part.glob('E*/finals/*.pt'))+list(part.glob('E*/BALANCEDIT*.pt'))
    assert len(paths)==29 and all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) for p in paths)
    rows=[dict(path=str(p),bytes=p.stat().st_size) for p in paths]
    receipt=dict(time=time.time(),reason='All16 terminals generated complete frozen consumers, save/load and Base identity audits complete, both workers exited',files=rows,total_bytes=sum(r['bytes'] for r in rows),historical_assets_touched=False,weights_reconstruction_requires_training=True)
    (root/'RETIREMENT_PLAN.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for p in paths:p.unlink()
    (root/'RETIREMENT_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return dict(files=len(rows),bytes=receipt['total_bytes'])
if __name__=='__main__':print(json.dumps(retire(os.environ['OUTPUT_ROOT'])))
