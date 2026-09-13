"""One assigned LoRA worker and a finite small-output relay to the original queue."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.medtrace.astra_judge_bundle import read, write_new


def state(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def validate_phase(root, cfg, mode):
    from scripts.medtrace.stage17_campaign import prefixes
    directory=root/'private'/f'lora_{mode}'
    receipt=read(directory/'COMPLETE.json')
    expected=dict(freeze_id=cfg['freeze_id'],method='lora',mode=mode,
        runtime=cfg['runtime_lock'],code_commit=cfg['code_commit'],method_lock=cfg['methods'],
        order=cfg['order'],prefixes=prefixes(cfg['N']))
    if (receipt['status']!='GENERATED_NOT_SCORED' or receipt['N']!=cfg['N']
            or receipt['phase']!=expected or read(directory/'BINDING.json')!=expected):
        raise ValueError('External phase provenance mismatch')
    if read(directory/'CLEANUP.json')['status']!='DELETED':
        raise ValueError('External checkpoint lifecycle incomplete')
    return receipt


def worker(cfg):
    from scripts.medtrace.stage17_campaign import cleanup, check_space, role_map
    root=Path(cfg['run'])
    with (root/'private/campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_new(root/'private/PREFIX_ACTIVE_TARGETS.json',role_map(
            read(Path(cfg['source_run'])/'private/COHORT_AND_SUPPORT_LEDGER.json')))
        for mode in ('single','sequential'):
            check_space(root)
            free=int(subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu']),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<22000: raise RuntimeError('Insufficient GPU peak margin')
            state(root/'public/PROGRESS.json',dict(status='RUNNING',method='lora',mode=mode,completed=0,N=cfg['N']))
            env=dict(os.environ,JOB_ACTION='phase',JOB_METHOD='lora',JOB_MODE=mode)
            with (root/f'lora_{mode}.log').open('ab') as log:
                subprocess.run([cfg['python'],'-u',cfg['entry']],env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            cleanup(cfg,'lora',mode)
            validate_phase(root,cfg,mode)
        state(root/'public/PROGRESS.json',dict(status='GPU_GENERATED_NOT_SCORED',method='lora',N=cfg['N']))


def gate(root, mode):
    """Called instead of GPU training by the existing queue's next child only."""
    cfg=read(root/'private/EXTERNAL_LORA.json')
    while not (root/'private/EXTERNAL_LORA_IMPORTED.json').exists():
        if (root/'private/EXTERNAL_LORA_FAILURE.json').exists():
            raise RuntimeError('External worker/relay failed; see EXTERNAL_LORA_FAILURE.json')
        state(root/'public/PROGRESS.json',dict(status='RUNNING',method='lora',mode=mode,
            phase='WAITING_FOR_ASSIGNED_EXTERNAL_WORKER',N=cfg['N']))
        time.sleep(60)
    if read(root/'private/EXTERNAL_LORA_IMPORTED.json')!=cfg:
        raise ValueError('External assignment/import mismatch')
    validate_phase(root,cfg,mode)


def receive(root):
    cfg=read(root/'private/EXTERNAL_LORA.json')
    incoming=root/'private/external-incoming'
    for mode in ('single','sequential'):
        validate_phase(incoming,cfg,mode)
        if (root/'private'/f'lora_{mode}').exists():
            raise RuntimeError('Refusing to overwrite existing phase')
    for mode in ('single','sequential'):
        (incoming/'private'/f'lora_{mode}').rename(root/'private'/f'lora_{mode}')
    write_new(root/'private/EXTERNAL_LORA_IMPORTED.json',cfg)


def remote_python(ssh, program):
    return subprocess.run([*ssh,'python3 -'],input=program,text=True,check=True,timeout=60)


def relay(cfg):
    """Mac transport only; never retrains or judges. Failure is explicit on both hosts."""
    import shlex
    local=Path(cfg['local']); local.mkdir(exist_ok=True)
    status=local/'RELAY.json'
    try:
        state(status,dict(status='WAITING_FOR_EXTERNAL_GPU'))
        while True:
            result=json.loads(subprocess.check_output([*cfg['source_ssh'],
                'cat '+shlex.quote(cfg['source_root']+'/public/PROGRESS.json')],text=True,timeout=30))
            if result['status']=='GPU_GENERATED_NOT_SCORED': break
            if result['status']!='RUNNING': raise RuntimeError('External GPU stopped: '+str(result))
            time.sleep(60)
        state(status,dict(status='TRANSFERRING_SMALL_OUTPUTS'))
        for mode in ('single','sequential'):
            name='lora_'+mode
            args=['rsync','-a','--include=*/','--include=*.json','--include=*.jsonl','--exclude=*']
            subprocess.run([*args,'-e',shlex.join(cfg['source_ssh'][:-1]),
                cfg['source_ssh'][-1]+':'+cfg['source_root']+'/private/'+name+'/',str(local/name)+'/'],check=True)
        assignment=read(local/'ASSIGNMENT.json')
        # Validate locally using the same directory shape as the GPU run.
        incoming=local/'payload'; (incoming/'private').mkdir(parents=True,exist_ok=True)
        for mode in ('single','sequential'):
            (local/('lora_'+mode)).rename(incoming/'private'/('lora_'+mode))
            validate_phase(incoming,assignment,mode)
        destination=cfg['destination_root']+'/private/external-incoming'
        remote_python(cfg['destination_ssh'],'from pathlib import Path\nPath('+repr(destination)+').mkdir(exist_ok=False)\n')
        subprocess.run(['rsync','-a','-e',shlex.join(cfg['destination_ssh'][:-1]),str(incoming)+'/',
            cfg['destination_ssh'][-1]+':'+destination+'/'],check=True)
        program='import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(cfg['destination_source'])+')\n'
        program+='from scripts.medtrace.stage17_external import receive\nreceive(Path('+repr(cfg['destination_root'])+'))\n'
        # Use the recorded runtime, not the server's unrelated system Python.
        subprocess.run([*cfg['destination_ssh'],cfg['destination_python']+' -'],input=program,text=True,check=True,timeout=60)
        state(status,dict(status='IMPORTED_FOR_EXISTING_FOLLOWER'))
    except Exception as error:
        failure=dict(status='STOPPED_NO_COMPUTE_RETRY',error=repr(error))
        state(status,failure)
        path=cfg['destination_root']+'/private/EXTERNAL_LORA_FAILURE.json'
        remote_python(cfg['destination_ssh'],'from pathlib import Path\nPath('+repr(path)+').write_text('+repr(json.dumps(failure))+')\n')
        raise


if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG'])
    if os.environ.get('JOB_ACTION')=='relay': relay(cfg)
    else:
        try: worker(cfg)
        except Exception as error:
            state(Path(cfg['run'])/'public/PROGRESS.json',dict(status='STOPPED',error=repr(error)))
            raise
