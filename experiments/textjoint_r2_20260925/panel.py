"""Freeze Base-eligible DEV/VERIFY panels before inspecting edit outcomes."""
import json
import os
from pathlib import Path
import time

ROOT=Path(os.environ['RUN_ROOT'])


def freeze():
    tasks=json.loads((ROOT/'private/TASKS_R2.json').read_text())['tasks']
    masks=json.loads((ROOT/'private/BASE_MASKS.json').read_text())['rows']
    base={(r['edit'],r['task'],r['query_id']):r['base_correct'] for r in masks}
    panels={}
    for split,target,minimum in [('validation',50,15),('test',60,20)]:
        audit=json.loads((ROOT/f'BASE_COVERAGE_{split.upper()}.json').read_text())
        if audit['status']!='READY' or audit['pending'] or audit['base_correct']<target:
            raise ValueError(f'{split} Base panel not complete or below target')
        rows=json.loads((ROOT/f'private/PRESSURE_{split.upper()}_FROZEN.json').read_text())['rows']
        if len(rows)!=target or len({r['source_group'] for r in rows})<minimum:
            raise ValueError(f'{split} source/unique target not met')
        panels[split]=rows
    locked=[];official_counts={};pressure_counts={}
    for task in tasks:
        order=task['order'];eligible=[]
        for row in task['evaluation']:
            identity=(task['canonical_edit_id'],row['task'],row['query_id'])
            if identity not in base:raise ValueError('Frozen Base mask missing')
            if base[identity] is row['task'].endswith('L'):
                eligible.append(row)
        split='validation' if order<=24 else 'test'
        index=order-1 if order<=24 else order-25
        source=panels[split]
        # DEV 50: 2 each plus two extras. VERIFY 60: 2 each plus twelve extras.
        pressure=[source[index*2],source[index*2+1]]
        extra=2 if split=='validation' else 12
        if index<extra:pressure.append(source[48+index])
        pressure=[dict(row,task='T2L_PRESSURE') for row in pressure]
        official_counts[order]={k:sum(r['task']==k for r in eligible) for k in ['T0','T1G','T1L','T2G','T2L']}
        pressure_counts[order]=len(pressure)
        locked.append(dict(task,evaluation=eligible+pressure,official_evaluation_full=task['evaluation'],
                           pressure_split=split))
    assert sum(pressure_counts[i] for i in range(1,25))==50
    assert sum(pressure_counts[i] for i in range(25,49))==60
    (ROOT/'private/TASKS_R2_LOCKED.json').write_text(json.dumps(dict(tasks=locked,CONFIRM=None,
        role='DEV24 validation-derived pressure; VERIFY24 test-derived pressure; no independent CONFIRM'),ensure_ascii=False,indent=2))
    counts=dict(status='FROZEN_BEFORE_FULL_EDIT_SCREEN',DEV24=dict(pressure_inputs=50,
                pressure_sources=len({r['source_group'] for r in panels['validation']}),
                official_eligible=sum(sum(official_counts[i].values()) for i in range(1,25))),
                VERIFY24=dict(pressure_inputs=60,pressure_sources=len({r['source_group'] for r in panels['test']}),
                official_eligible=sum(sum(official_counts[i].values()) for i in range(25,49))),
                official_counts=official_counts,pressure_counts=pressure_counts,created_epoch=time.time(),
                original_unchanged=str(ROOT/'private/TASKS_MATRIX.json'))
    (ROOT/'PANEL_LOCK.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps({k:counts[k] for k in ['status','DEV24','VERIFY24']}))


if __name__=='__main__':freeze()
