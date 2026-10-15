"""Finite generation/scoring/report chain with additive, once-only cost transfers."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import wording_pair as protocol

RUN=Path(os.environ['RUN_ROOT']); TOOLS=RUN/'private/tools'
def read(p): return json.loads(p.read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def execute(name,part=None,operation=None):
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(entry).write_text('import runpy\nrunpy.run_path('+repr(str(TOOLS/name))+',run_name="__main__")\n')
    env=dict(os.environ)
    if part is not None: env.update(PARTITION=str(part),GPU='2',ACTION='m13_146_single')
    if operation is not None: env['QUEUE_REQUEST']=json.dumps(operation)
    label=str(part) if part is not None else name.replace('.py','')
    with (RUN/'logs'/f'{label}.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-u',entry],env=env,stdout=log,stderr=subprocess.STDOUT)
        write(RUN/'private'/f'START_{label}.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],entry=entry,epoch=time.time()))
        assert p.wait()==0,'Child failed; no automatic retry: '+name


def merge(phase):
    gate=read(RUN/'private/WORDING_ADMISSION1.json');parent=Path(gate['parent'])
    own=read(RUN/'RESOURCE_LEDGER.json');prior=read(RUN/'private/INHERITED_COST.json')
    assert all(s.get('ended_epoch') for s in own['gpu_sessions'])
    key=gate.get('cost_namespace','wording_pair_v1')+'_'+phase+'_merged'
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
    protocol.check(RUN)
    with (RUN/'private/WORDING_CONTROLLER_ONCE1').open('x'):pass
    for part in (0,1): execute('wording_runtime.py',part=part)
    counts=collections_counts()
    assert counts['generation_attempts']==protocol.CONSUMERS
    assert len(list((RUN/'private/outputs').glob('*/*/*.json')))==protocol.CONSUMERS
    write(RUN/'private/WORDING_GENERATION_COMPLETE1.json',dict(status='COMPLETE',outputs=protocol.CONSUMERS,counts=counts,epoch=time.time()))
    merge('GPU');execute('wording_queue.py',operation=dict(action='ingest'))
    queue=RUN/'private'/protocol.QUEUE
    while not ((queue/'ALL_WORKERS_COMPLETE.json').exists() and (queue/'workers/0/SCORER_DONE.json').exists()):
        assert time.time()<read(RUN/'RUN_MANIFEST.json')['deadline_epoch']
        assert not list((queue/'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(20)
    merge('Judge');execute('wording_report.py')
    write(RUN/'private/WORDING_COMPLETE1.json',dict(status='COMPLETE_PUBLICATION_PENDING',epoch=time.time()))


def collections_counts():
    import collections
    counts=collections.Counter()
    for part in (0,1): counts.update(read(RUN/f'private/COUNTS_m13_146_single_{part}.json'))
    return dict(counts)


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;write(RUN/'private/FAILURE_wording_controller1.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        # Charge finished work on failure too; preserve partial outputs and original failures.
        for phase in ('GPU','Judge'):
            try:merge(phase)
            except BaseException as error:write(RUN/f'private/COST_TRANSFER_{phase}_ERROR.json',dict(error=repr(error)))
        write(RUN/'private/WORDING_CONTROLLER_EXIT1.json',dict(exit_code=code,epoch=time.time()))
