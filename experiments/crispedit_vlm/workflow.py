"""Detached dependency-ordered editing, grading and report workflow; neutral argv."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


def main():
    root=Path(os.environ['RUN_DIR']);c=json.loads((root/'CONFIG.private.json').read_text())
    def write(path,value):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    def status(state,**kwargs):write(root/'WORKFLOW.json',dict(state=state,updated_epoch=time.time(),**kwargs))
    def gate(required):
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        row=next(x for x in rows if x.split(',')[1].strip()==c['gpu_uuid'])
        if int(row.split(',')[0])!=c['gpu_index'] or int(row.split(',')[2])<required:
            raise RuntimeError('Frozen GPU identity/free-memory gate failed')
        if time.time()>=c['deadline_epoch']:raise TimeoutError('Original wall deadline expired')
    def stage(name,python,code,run_dir):
        entry=root/(name+'.entry.private.txt');entry.write_text(code)
        env=dict(os.environ,RUN_DIR=str(run_dir),CUDA_VISIBLE_DEVICES=str(c['gpu_index']),
            PYTHONPATH=str(root/'source')+':'+c['judge_support_source'],
            CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
            VLLM_WORKER_MULTIPROC_METHOD='spawn',
            CPATH='/data/bmw/envs/s1/headers/usr/include/python3.12:/data/bmw/envs/s1/headers/usr/include')
        started=time.time()
        with entry.open() as inp,(root/(name+'.private.log')).open('a') as log:
            p=subprocess.Popen([python,'-'],stdin=inp,stdout=log,stderr=subprocess.STDOUT,
                cwd=root,env=env,start_new_session=True)
        status(name.upper(),pid=p.pid,started_epoch=started)
        try:code=p.wait(timeout=max(1,c['deadline_epoch']-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
        receipt=dict(exit_code=code,started_epoch=started,finished_epoch=time.time(),pid=p.pid,argv=[python,'-'])
        write(run_dir/'CONTROLLER_RESULT.json',receipt)
        write(root/(name+'.EXECUTION.json'),receipt)
        if code:raise RuntimeError(name+' exited with code '+str(code)+'; preserve state, no blind retry')
    try:
        if not (root/'COMPLETE.json').exists():
            gate(65000)
            stage('editing','/data/bmw/envs/s0/bin/python','from experiments.crispedit_vlm.run import main\nmain()\n',root)
        if not (root/'scoring').exists():
            stage('prepare','/data/bmw/envs/s1/bin/python',
                'import os\nfrom experiments.crispedit_vlm.score import prepare\nprint(prepare(os.environ["RUN_DIR"]))\n',root)
        if not (root/'scoring/CONTROLLER_RESULT.json').exists():
            gate(57000)
            stage('scoring','/data/bmw/envs/s1/bin/python','from experiments.crispedit_vlm.score import worker\nworker()\n',root/'scoring')
        elif json.loads((root/'scoring/CONTROLLER_RESULT.json').read_text())['exit_code']:
            raise RuntimeError('Failed scoring state requires explicit operational repair; accepted decisions stay preserved')
        stage('report','/data/bmw/envs/s1/bin/python',
            'import os,json\nfrom experiments.crispedit_vlm.score import report\nprint(json.dumps(report(os.environ["RUN_DIR"])))\n',root)
        status('COMPLETE_AWAITING_PUBLICATION')
    except BaseException as error:
        status('STOPPED_NEEDS_REPAIR',error=repr(error),traceback=traceback.format_exc());raise
