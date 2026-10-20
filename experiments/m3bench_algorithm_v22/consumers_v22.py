"""Exact finite output coverage, missing/duplicate rejection and bounded lifecycle."""
import json
from pathlib import Path
ARMS=('SOURCE','CE','KL','CE_PROJ','RESERVE','BALANCEDIT','BALANCEDIT_CE')
def read(p):return json.loads(Path(p).read_text())
def coverage(actual,expected):
    assert len(actual)==len(set(actual)) and set(actual)==set(expected),'Missing, extra or duplicate output consumer'
    return len(actual)
def validate(root):
    config=read(root/'DATA_ADMISSION.json');queries=read(root/'QUERY_ROWS.json');expected=[];actual=[]
    for task in config['tasks']:
        part=(task['order']-1)%5;folder=root/str(part)/task['key']
        for arm in ARMS:
            expected.extend((task['key'],arm,r['query_id']) for r in task['eval_rows'])
            for p in (folder/'outputs'/arm).glob('*.json'):actual.append((task['key'],arm,p.stem))
    return dict(method_outputs=coverage(actual,expected),base_outputs=coverage([p.stem for p in (root/'base').glob('*/*.json')],queries),scientific_scoring='REQUIRES_COMPLETE_HIDDEN_VERDICTS')
def retire_edit(folder,write):
    import torch
    assert read(folder/'MODEL_CONSUMERS_COMPLETE.json')['status']=='PASS'
    # Preserve SOURCE active/final for the registered source-bank and recovery consumers.
    paths=[folder/a/n for a in ARMS[1:5] for n in ('active.pt','final.pt')]+[folder/(a+'.pt') for a in ARMS[5:]]
    anchors={a:torch.load(folder/(a+'.pt'),map_location='cpu',weights_only=True)['router'] for a in ARMS[5:]}
    torch.save(anchors,folder/'R0_ANCHORS.pt')
    assert all(p.is_file() for p in paths)
    deleted=[dict(file=str(p.relative_to(folder)),bytes=p.stat().st_size) for p in paths]
    for p in paths:p.unlink()
    write(folder/'LIFECYCLE.json',dict(status='RETIRED_NON_SOURCE',deleted=deleted,SOURCE='Retained only for registered bank consumer',last_consumers='all seven-arm answers, bindings, routes, save-load'))
def selfcheck():
    assert coverage([('e','a','q')],[('e','a','q')])==1
    for bad in [[],[('e','a','q')]*2,[('e','a','z')]]:
        try:coverage(bad,[('e','a','q')])
        except AssertionError:pass
        else:raise AssertionError('coverage accepted invalid')
    assert len(ARMS)==7
    return dict(status='PASS',missing_duplicate_extra_rejected=True,all_seven_required=True)
