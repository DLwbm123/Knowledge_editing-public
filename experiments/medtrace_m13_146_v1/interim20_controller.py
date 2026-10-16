"""One finite interim chain; archive failures and reuse only exact score identities."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

RUN=Path(os.environ['RUN_ROOT'])
PARENT=Path(os.environ['INTERIM20_PARENT'])
TOOLS=RUN/'private/tools'


def read(p):return json.loads(p.read_text())
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)


def execute(name,operation=None):
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(entry).write_text('import runpy\nrunpy.run_path('+repr(str(TOOLS/name))+',run_name="__main__")\n')
    env=dict(os.environ)
    if operation is not None:env['QUEUE_REQUEST']=json.dumps(operation)
    log=RUN/'logs'/('worker.log' if operation is None else 'queue.log')
    with log.open('ab') as handle:
        child=subprocess.Popen([sys.executable,'-u',entry],env=env,stdout=handle,stderr=subprocess.STDOUT)
        write(RUN/'private'/('GPU_START1.json' if operation is None else 'QUEUE_START1.json'),dict(pid=child.pid,
            start_ticks=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),entry=entry,epoch=time.time()))
        assert child.wait()==0, name+' failed; no automatic retry'


def merge_cost(phase):
    own=read(RUN/'RESOURCE_LEDGER.json');prior=read(RUN/'private/INHERITED_COST.json')
    assert all(v.get('ended_epoch') for v in own['gpu_sessions'])
    with (PARENT/'RESOURCE_LEDGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=read(PARENT/'RESOURCE_LEDGER.json');key='interim20_'+phase+'_merged'
        assert key not in ledger,'No repeated cost transfer'
        if phase=='GPU':
            delta=own['gpu_seconds_used']-prior['gpu_seconds_used']
            ledger['gpu_seconds_used']+=delta;ledger['gpu_sessions']+=own['gpu_sessions']
        else:
            delta=own['Judge_attempts']-prior['Judge_attempts'];ledger['Judge_attempts']+=delta
            assert ledger['Judge_attempts']<=read(PARENT/'RUN_MANIFEST.json')['Judge_limit']
        ledger[key]=dict(run=str(RUN),increment=delta,epoch=time.time())
        write(PARENT/'RESOURCE_LEDGER.json',ledger)
        write(RUN/'private'/('COST_TRANSFER_'+phase+'1.json'),ledger[key])


def finish():
    merge_cost('GPU')
    execute('interim20_queue.py',dict(action='ingest'))
    db=RUN/'private/judge_interim20_astra_medium/queue.sqlite'
    assert db.is_file()
    # The full controller reads this existing allowlist only when it later ingests.
    with (PARENT/'private/INTERIM20_REUSE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=PARENT/'private/M13_146_REUSE_QUEUES.json';paths=read(path)
        assert str(db) not in paths;write(path,paths+[str(db)])
        write(RUN/'private/FULL146_REUSE_REGISTERED1.json',dict(status='EXACT_IDENTITY_ONLY',queue=str(db),epoch=time.time()))
    queue=db.parent
    while not ((queue/'ALL_WORKERS_COMPLETE.json').exists() and all((queue/f'workers/{i}/SCORER_DONE.json').exists() for i in range(4))):
        assert time.time()<read(RUN/'RUN_MANIFEST.json')['deadline_epoch']
        assert not list((queue/'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(20)
    merge_cost('Judge')
    execute('interim20_report.py',{})
    assert (RUN/'private/INTERIM20_REPORT_COMPLETE1.json').exists()
    write(RUN/'private/INTERIM20_COMPLETE1.json',dict(status='COMPLETE_PUBLICATION_PENDING',training_modified=False,epoch=time.time()))


def main():
    with (RUN/'private/CONTROLLER_ONCE1').open('x'):pass
    execute('interim20_runtime.py')
    assert read(RUN/'private/INTERIM20_EXIT1.json')['exit_code']==0
    finish()


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;write(RUN/'private/FAILURE_interim20_controller1.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,epoch=time.time()))
        raise
    finally:write(RUN/'private/CONTROLLER_EXIT1.json',dict(exit_code=code,epoch=time.time()))
