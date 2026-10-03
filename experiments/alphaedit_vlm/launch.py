"""Launch the frozen three-GPU dependency graph using neutral process arguments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(os.environ['CAMPAIGN_DIR'])
    config=json.loads((root/'CONFIG.private.json').read_text())
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                                  '--format=csv,noheader,nounits'],text=True)
    devices={int(parts[0]):(parts[1],int(parts[2])) for parts in
             ([p.strip() for p in line.split(',')] for line in rows.splitlines())}
    if time.time()>=config['deadline_epoch']:raise RuntimeError('campaign deadline elapsed')
    if any(devices[g][0]!=config['gpu_uuids'][str(g)] or devices[g][1]<52000 for g in (5,6,7)):
        raise RuntimeError('GPU identity or free-memory gate failed')
    jobs={};receipts=[]
    for mode,gpu in [('statistics',5),('single',6),('multi',7)]:
        out=root/mode;out.mkdir(exist_ok=False)
        entry=out/'worker.private.py'
        entry.write_text('from experiments.alphaedit_vlm.run import main\nmain()\n')
        env=dict(os.environ,RUN_DIR=str(out),WORKER_MODE=mode,
                 CUDA_VISIBLE_DEVICES=devices[gpu][0],CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        with entry.open() as source,(out/'stdout.private.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-'],stdin=source,stdout=log,
                stderr=subprocess.STDOUT,cwd=root/'source',env=env,start_new_session=True)
        jobs[mode]=process
        receipt=dict(mode=mode,pid=process.pid,gpu=gpu,uuid=devices[gpu][0],
                     argv=[sys.executable,'-'],free_MiB_before_launch=devices[gpu][1])
        receipts.append(receipt)
    (root/'LAUNCH.json').write_text(json.dumps(dict(jobs=receipts,deadline_epoch=config['deadline_epoch']),indent=2)+'\n')
    print(json.dumps(receipts),flush=True)
    codes={mode:process.wait() for mode,process in jobs.items()}
    result=dict(state='COMPLETE' if all(code==0 for code in codes.values()) else 'STOPPED_ON_ERROR',
                exit_codes=codes,finished_epoch=time.time())
    if result['state']=='COMPLETE':
        stats=json.loads((root/'statistics/STATUS.json').read_text())
        for arm in ('single','multi'):
            status=json.loads((root/arm/'STATUS.json').read_text())
            cases=status['cases'];probes=[p for c in cases for p in c['probes']]
            corrected=sum(not c['target_tokens_exact_before'] and c['target_tokens_exact_after'] for c in cases)
            drops=sum(p['score_change']<-.1 for p in probes)
            result[arm]=dict(completed=len(cases),full_target_corrected=corrected,
                exact_before=sum(c['target_tokens_exact_before'] for c in cases),
                exact_after=sum(c['target_tokens_exact_after'] for c in cases),
                probe_slots=len(probes),probe_score_drops=drops,probe_KL_above_001=sum(p['KL']>.001 for p in probes),
                worst_probe_score_change=min(p['score_change'] for p in probes),
                joint_criterion=corrected>=2 and drops==0 and all(c['clean_reload']['generation_identical'] for c in cases),
                elapsed_seconds=status['elapsed_seconds'],sum_case_seconds=sum(c['elapsed_seconds'] for c in cases))
        files=[root/'statistics'/f'basis-{layer}.private.pt' for layer in config['statistics_layers']]
        receipt=dict(files=len(files),bytes=sum(p.stat().st_size for p in files),
                     reason='Both independent-edit arms and every native consumer completed',
                     retained_copy=False,reconstruction='Recompute from frozen statistics manifest and Base')
        for path in files:path.unlink()
        (root/'statistics/DELETION.json').write_text(json.dumps(receipt,indent=2)+'\n')
        result['statistics_seconds']=stats['elapsed_seconds']
    (root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
