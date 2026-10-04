"""Detached environment setup and dependency-ordered finite throughput run."""
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import traceback
from .run import write


def main():
    root=Path(os.environ['RUN_DIR']);config=json.loads((root/'CONFIG.private.json').read_text())
    def state(value,**kw):write(root/'STATUS.json',dict(state=value,updated_epoch=time.time(),**kw))
    try:
        envroot=Path(config['environment'])
        if os.environ.get('RESUME_SETUP')=='1':
            previous=json.loads((root/'STATUS.json').read_text())
            if previous['state']!='INSTALLING_VLLM' or list(root.glob('c*.STATUS.json')):
                raise RuntimeError('Setup resume requires an interrupted pre-inference installation')
            if not (envroot/'bin/python').is_file():raise FileNotFoundError('Missing isolated environment')
        else:
            state('CREATING_ENVIRONMENT')
            if envroot.exists():raise FileExistsError('New isolated environment already exists')
            subprocess.run(['/usr/bin/python3','-m','venv','--without-pip',str(envroot)],check=True)
        exe=str(envroot/'bin/python')
        state('INSTALLING_VLLM')
        with (root/'install.private.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pip','--python',exe,'install','--disable-pip-version-check',
                'vllm==0.10.2','transformers==4.55.2','numpy==2.2.6','openai==1.99.9',
                '--index-url',os.environ.get('PIP_INDEX_URL','https://pypi.org/simple')],stdout=log,stderr=subprocess.STDOUT,check=True)
        state('WAITING_FOR_MODEL_COPY')
        while not (root/'MODEL_READY.json').exists():
            if (root/'MODEL_TRANSFER_FAILED.json').exists():raise RuntimeError('Model transfer failed')
            time.sleep(5)
        model=Path(config['model_path'])
        index=json.loads((model/'model.safetensors.index.json').read_text())
        if any(not(model/name).is_file() for name in set(index['weight_map'].values())):raise ValueError('Missing model shard')
        state('READY_FOR_BENCHMARK')
        entry=Path(config['neutral_entry'])
        entry.write_text('from experiments.qwen_speed.run import main\nif __name__ == "__main__": main()\n')
        for concurrency in (2,32):
            gpu=subprocess.check_output(['nvidia-smi','-i',config['gpu_uuid'],'--query-gpu=uuid,memory.free,index','--format=csv,noheader,nounits'],text=True).strip().split(', ')
            if gpu[0]!=config['gpu_uuid'] or int(gpu[1])<57000:raise RuntimeError('GPU identity/free memory gate failed')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu[2],CUDA_DEVICE_ORDER='PCI_BUS_ID',CONCURRENCY=str(concurrency),
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
                     VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
            with (root/f'c{concurrency}.private.log').open('w') as log:
                process=subprocess.Popen([exe,str(entry)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                state('BENCHMARKING',concurrency=concurrency,pid=process.pid,argv=[exe,str(entry)])
                try:process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
                if process.returncode:raise RuntimeError(f'Benchmark concurrency {concurrency} failed: {process.returncode}')
        state('COMPLETE',results=[json.loads((root/f'c{n}.RESULT.json').read_text()) for n in (2,32)])
    except Exception as error:
        state('FAILED',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
