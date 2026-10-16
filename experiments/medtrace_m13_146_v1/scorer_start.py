"""Four new isolated scorer instances; never restore an older scorer/watch root."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def configs(root,run,source,task,queue_file='queue146.py',
        result_folder='judge_m13_146_astra_medium',epoch='MEDTRACE_M13_NATIVE_FIT146_20261014_V1'):
    return [dict(RUN_ROOT=run,SCORER_STATE=str(root/str(worker)),SCORER_SOURCE=source,TASK_FILE=task,
        JUDGE_QUEUE_FILE=queue_file,JUDGE_RESULT_FOLDER=f'{result_folder}/workers/{worker}',
        JUDGE_EPOCH=epoch,JUDGE_EFFORT='medium',
        JUDGE_WORKER=str(worker),JUDGE_SHARED_ROOT=str(root)) for worker in range(4)]


def start(queue_file='queue146.py',result_folder='judge_m13_146_astra_medium',
        epoch='MEDTRACE_M13_NATIVE_FIT146_20261014_V1',admission_file='M13_146_RUNTIME_ADMISSION.json'):
    run=os.environ['M13_REMOTE_RUN'];root=Path(os.environ['M13_LOCAL_SCORER_ROOT'])
    source=os.environ['SCORER_SOURCE'];task=os.environ['SCORER_TASK']
    assert root.is_dir() and not any(root.iterdir()),'New empty isolated root required'
    assert admission_file in ('M13_146_RUNTIME_ADMISSION.json','SELECTED_RUNTIME_ADMISSION1.json',
                              'SELECTED_PREFILL_RECOVERY_ADMISSION1.json')
    code="import json,pathlib; r=pathlib.Path("+repr(run)+"); a=json.loads((r/'private'/"+repr(admission_file)+").read_text()); assert a['status']=='PASS' and a['launch_authorized']; s=json.loads((r/'private/SOURCE_ADMISSION146.json').read_text()); assert s['status']=='PASS'; print('PASS')"
    p=subprocess.run(['ssh','pro5000','python3 -'],input=code,text=True,capture_output=True,check=True)
    assert p.stdout.strip()=='PASS'
    with (root/'SCORER_START_ONCE').open('x'):pass
    jobs=[]
    for config in configs(root,run,source,task,queue_file,result_folder,epoch):
        work=Path(config['SCORER_STATE']);work.mkdir();(work/'config.json').write_text(json.dumps(config))
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
        Path(entry).write_text('import os,json,runpy\nfrom pathlib import Path\nos.environ.update(json.loads(Path('+repr(str(work/'config.json'))+').read_text()))\nrunpy.run_path(os.environ["TASK_FILE"],run_name="__main__")\n')
        process=subprocess.Popen([sys.executable,'-u',entry],stdout=(work/'scorer.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(dict(pid=process.pid,worker=int(config['JUDGE_WORKER']),entry=entry,epoch=time.time()))
    (root/'START.json').write_text(json.dumps(jobs,indent=2)+'\n')


if __name__=='__main__':start()
