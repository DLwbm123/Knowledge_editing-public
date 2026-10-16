"""Finite C/M13 generation, exact scoring reuse, report; original Fisher untouched."""
import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import cm_eval146 as protocol
from interim20_controller import read,write

RUN=Path(os.environ['RUN_ROOT']);TOOLS=RUN/'private/tools'


def execute(name,action,part=0,operation=None):
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(entry).write_text('import runpy\nrunpy.run_path('+repr(str(TOOLS/name))+',run_name="__main__")\n')
    env=dict(os.environ,ACTION=action,PARTITION=str(part),GPU='1',CUDA_VISIBLE_DEVICES='1' if name=='cm_runtime146.py' else '')
    if operation is not None:env['QUEUE_REQUEST']=json.dumps(operation)
    label=action+'_'+str(part)
    with (RUN/'logs'/f'{label}.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-u',entry],env=env,stdout=log,stderr=subprocess.STDOUT)
        write(RUN/'private'/f'START_{label}.json',dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],entry=entry,epoch=time.time()))
        assert child.wait()==0,'Child failed; no automatic retry: '+label


def merge(phase):
    gate=read(RUN/'private/CM_ADMISSION1.json');parent=Path(gate['parent'])
    own=read(RUN/'RESOURCE_LEDGER.json');prior=read(RUN/'private/INHERITED_COST.json')
    assert all(s.get('ended_epoch') for s in own['gpu_sessions'])
    key='cm_eval146_v1_'+phase+'_merged'
    with (parent/'RESOURCE_LEDGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(parent/'RESOURCE_LEDGER.json')
        if key in ledger:
            assert ledger[key]['run']==str(RUN);return
        field='gpu_seconds_used' if phase=='GPU' else 'Judge_attempts'
        delta=own[field]-prior[field];assert delta>=0
        ledger[field]+=delta
        if phase=='GPU':ledger['gpu_sessions']+=own['gpu_sessions']
        else:assert ledger[field]<=read(parent/'RUN_MANIFEST.json')['Judge_limit']
        ledger[key]=dict(run=str(RUN),increment=delta,epoch=time.time())
        write(parent/'RESOURCE_LEDGER.json',ledger);write(RUN/f'private/COST_TRANSFER_{phase}1.json',ledger[key])


def main():
    protocol.admit(RUN,check_weights=True)
    assert protocol.read(RUN/'private/CM_CPU_CHECK1.json')['status']=='PASS'
    assert protocol.read(RUN/'private/CM_SOURCE1.json')['commit']!='CPU_PREPARATION_NOT_LAUNCHED'
    with (RUN/'private/CM_CONTROLLER_ONCE1').open('x'):pass
    for action in ('m13_146_single','m13_146_bank'):
        for part in (0,1):execute('cm_runtime146.py',action,part)
    counts=collections.Counter()
    for path in (RUN/'private').glob('COUNTS_*.json'):counts.update(read(path))
    assert counts['generation_attempts']==16375 and counts['non_generation_forwards']<=protocol.NON_GENERATION_FORWARDS
    assert counts['backwards']==counts['updates']==counts['operator_backwards']==counts['local_solves']==0
    assert len(list((RUN/'private/outputs').glob('*/*/*.json')))==protocol.CONSUMERS
    write(RUN/'private/CM_GENERATION_COMPLETE1.json',dict(status='COMPLETE',outputs=protocol.CONSUMERS,reused=706,counts=dict(counts),epoch=time.time()))
    merge('GPU');execute('cm_queue146.py','cm_ingest',operation=dict(action='ingest'))
    queue=RUN/'private'/protocol.QUEUE
    while not ((queue/'ALL_WORKERS_COMPLETE.json').exists() and (queue/'workers/0/SCORER_DONE.json').exists()):
        protocol.project_budget(RUN)
        assert time.time()<read(RUN/'RUN_MANIFEST.json')['deadline_epoch']
        assert not list((queue/'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(20)
    merge('Judge');execute('cm_report146.py','cm_report')
    assert (RUN/'private/CM_REPORT_COMPLETE1.json').exists()
    write(RUN/'private/CM_COMPLETE1.json',dict(status='COMPLETE_PUBLICATION_PENDING',epoch=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;write(RUN/'private/FAILURE_cm_controller1.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,epoch=time.time()));raise
    finally:
        for phase in ('GPU','Judge'):
            try:merge(phase)
            except BaseException as error:write(RUN/f'private/COST_TRANSFER_{phase}_ERROR.json',dict(error=repr(error)))
        write(RUN/'private/CM_CONTROLLER_EXIT1.json',dict(exit_code=code,epoch=time.time()))
