"""Launch a bounded dependency graph; neutral argv for every GPU process."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    root=Path(os.environ['CAMPAIGN_DIR']);c=json.loads((root/'CONFIG.private.json').read_text())
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    devices={int(p[0]):(p[1],int(p[2])) for p in ([x.strip() for x in line.split(',')] for line in rows.splitlines())}
    if time.time()>=c['deadline_epoch']:raise RuntimeError('deadline expired')
    for gpu in (5,6,7):
        if devices[gpu][0]!=c['gpu_uuids'][str(gpu)] or devices[gpu][1]<52000:raise RuntimeError('GPU identity/memory gate failed')
    jobs={};receipts=[]
    for mode,gpu in [('statistics',5),('single_layer',6),('multi_layer',7)]:
        out=root/mode;out.mkdir(exist_ok=False)
        entry=out/'worker.private.py';entry.write_text('from experiments.alphaedit_stage17.run import main\nmain()\n')
        env=dict(os.environ,RUN_DIR=str(out),WORKER_MODE=mode,CUDA_VISIBLE_DEVICES=devices[gpu][0],
                 CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        with entry.open() as inp,(out/'stdout.private.log').open('w') as log:
            p=subprocess.Popen([sys.executable,'-'],stdin=inp,stdout=log,stderr=subprocess.STDOUT,cwd=root/'source',env=env,start_new_session=True)
        jobs[mode]=p;receipts.append(dict(mode=mode,pid=p.pid,gpu=gpu,uuid=devices[gpu][0],argv=[sys.executable,'-'],free_MiB_before=devices[gpu][1]))
    (root/'LAUNCH.json').write_text(json.dumps(dict(jobs=receipts,started_epoch=time.time(),deadline_epoch=c['deadline_epoch']),indent=2)+'\n')
    while any(p.poll() is None for p in jobs.values()):
        if time.time()>c['deadline_epoch']+5:
            for p in jobs.values():
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            time.sleep(5)
            for p in jobs.values():
                if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
            break
        time.sleep(2)
    codes={mode:p.wait() for mode,p in jobs.items()}
    result=dict(state='GENERATED_NOT_SCORED' if all(v==0 for v in codes.values()) else 'STOPPED_ON_ERROR',exit_codes=codes,finished_epoch=time.time())
    if all(codes.get(mode)==0 for mode in ('single_layer','multi_layer')):
        paths=[root/'statistics'/f'basis-{l}.private.pt' for l in c['statistics_layers']]
        receipt=dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),reason='Both complete single and sequential consumers finished',retained_copy=False,reconstruction='recompute frozen corpus statistics')
        for p in paths:p.unlink()
        (root/'statistics/DELETION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
