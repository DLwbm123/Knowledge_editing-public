"""One local launch after both native-reload barriers; permanent server evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    root = Path(__file__).resolve().parents[2]
    state = root/'.local_run/fixed_input_context_pin8_v3_20261017'
    startup = json.loads((state/'STARTUP.json').read_text())
    run = startup['run_root']
    assert run.endswith('/fixed-input-context-pin8-v3-20261017/run')
    code = 'import os,runpy\nos.environ.update(RUN_ROOT='+repr(run)+',QUEUE_REQUEST=\'{"action":"status"}\')\nrunpy.run_path(os.environ["RUN_ROOT"]+"/private/tools/judge_queue.py",run_name="__main__")\n'
    status = json.loads(subprocess.run(['ssh','pro5000','python3 -'],input=code,capture_output=True,text=True,check=True).stdout)
    assert status['status'] == 'GENERATION_COMPLETE'
    # Create the once-only receipt before spawning; uncertain launches require inspection.
    receipt = state/'SCORERS.json'
    with receipt.open('x') as handle:json.dump(dict(status='STARTING',epoch=time.time(),queue=status,workers=[]),handle)
    temporary = Path(tempfile.mkdtemp(prefix='e.',dir='/private/tmp'))
    base = root/'experiments/medtrace_full_method_comparison_v2_20261003'
    workers = []
    for i in range(4):
        entry = temporary/f'{i}.py'
        entry.write_text('import os,runpy\nrunpy.run_path(os.environ["TASK_FILE"],run_name="__main__")\n')
        log = temporary/f'{i}.log'
        env = dict(os.environ,RUN_ROOT=run,SCORER_STATE=str(temporary/f's{i}'),SCORER_SOURCE=str(base/'execution_source'),
                   SCORER_BASE=str(base/'scorer.py'),TASK_FILE=str(root/'experiments/fixed_input_guard8_v1/score.py'),
                   JUDGE_QUEUE_FILE='judge_queue.py',JUDGE_RESULT_FOLDER=f'judge_fixed8/worker{i}',
                   JUDGE_EPOCH='FIXED_INPUT_CONTEXT_PIN8_V3_20261017',JUDGE_EFFORT='medium',JUDGE_WORKER=str(i),
                   HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
        with log.open('w') as handle:
            p = subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,cwd=temporary)
        argv = subprocess.check_output(['ps','-ww','-p',str(p.pid),'-o','args='],text=True).strip()
        assert not any(word in argv for word in ('wangbomin','Knowledge_editing','guard'))
        workers.append(dict(worker=i,pid=p.pid,argv=argv,log=str(log),state=env['SCORER_STATE']))
        receipt.write_text(json.dumps(dict(status='STARTED' if i==3 else 'STARTING',epoch=time.time(),queue=status,workers=workers),indent=2))
    print(receipt.read_text())


if __name__=='__main__':main()
