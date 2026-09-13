"""Assigned baseline workers and finite small-output relays to the original queue."""
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


def assigned_phases(cfg):
    methods=cfg.get('assigned_methods',['lora'])
    modes=cfg.get('assigned_modes',['single','sequential'])
    if methods not in (['lora'],['grace','belora']):
        raise ValueError('Unapproved external method assignment')
    if modes!=['single','sequential'] and not (methods==['lora'] and modes==['sequential']
            and cfg.get('precision_variant')=='LORA_SEQUENTIAL_BF16_STABILITY_V1'):
        raise ValueError('Unapproved external phase selection')
    return [(method,mode) for method in methods for mode in modes]


def phase_config(cfg, method, mode):
    if 'phase_assignments' not in cfg: return cfg
    amendment=cfg.get('acceptance_amendment',{})
    if (method!='lora' or amendment.get('id')!='LORA_SINGLE_FP16_SEQUENTIAL_BF16_V1'
            or set(cfg['phase_assignments'])!={'single','sequential'}):
        raise ValueError('Unrecognized mixed-precision acceptance amendment')
    selected=cfg['phase_assignments'][mode]
    if any(selected[k]!=cfg[k] for k in ('freeze_id','N','order')):
        raise ValueError('Amended phase cohort/order mismatch')
    expected='float16' if mode=='single' else 'bfloat16'
    if selected['methods']['generation']['dtype']!=expected:
        raise ValueError('Amended phase precision mismatch')
    if mode=='sequential' and selected.get('precision_variant')!='LORA_SEQUENTIAL_BF16_STABILITY_V1':
        raise ValueError('Missing BF16 variant provenance')
    return selected


def validate_phase(root, cfg, mode, *, method='lora', require_cleanup=True):
    from scripts.medtrace.stage17_campaign import prefixes
    cfg=phase_config(cfg,method,mode)
    directory=root/'private'/f'{method}_{mode}'
    receipt=read(directory/'COMPLETE.json')
    expected=dict(freeze_id=cfg['freeze_id'],method=method,mode=mode,
        runtime=cfg['runtime_lock'],code_commit=cfg['code_commit'],method_lock=cfg['methods'],
        order=cfg['order'],prefixes=prefixes(cfg['N']))
    if (receipt['status']!='GENERATED_NOT_SCORED' or receipt['N']!=cfg['N']
            or receipt['phase']!=expected or read(directory/'BINDING.json')!=expected):
        raise ValueError('External phase provenance mismatch')
    if require_cleanup and read(directory/'CLEANUP.json')['status']!='DELETED':
        raise ValueError('External checkpoint lifecycle incomplete')
    return receipt


def worker(cfg):
    from scripts.medtrace.stage17_campaign import cleanup, check_space, role_map, CheckpointVisibilityError
    root=Path(cfg['run'])
    if 'phase_assignments' in cfg: raise ValueError('Acceptance amendment is import-only, not a training dispatch')
    phases=assigned_phases(cfg)
    with (root/'private/campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        roles=role_map(read(Path(cfg['source_run'])/'private/COHORT_AND_SUPPORT_LEDGER.json'))
        rolefile=root/'private/PREFIX_ACTIVE_TARGETS.json'
        if rolefile.exists():
            if read(rolefile)!=roles: raise ValueError('Existing prefix role map changed')
        else: write_new(rolefile,roles)
        pending=[]
        for method,mode in phases:
            check_space(root)
            directory=root/'private'/f'{method}_{mode}'
            if not (directory/'COMPLETE.json').exists():
                free=int(subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu']),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<22000: raise RuntimeError('Insufficient GPU peak margin')
                state(root/'public/PROGRESS.json',dict(status='RUNNING',method=method,mode=mode,completed=0,N=cfg['N']))
                env=dict(os.environ,JOB_ACTION='phase',JOB_METHOD=method,JOB_MODE=mode)
                with (root/f'{method}_{mode}.log').open('ab') as log:
                    subprocess.run([cfg['python'],'-u',cfg['entry']],env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True)
            validate_phase(root,cfg,mode,method=method,require_cleanup=False)
            try: cleanup(cfg,method,mode)
            except CheckpointVisibilityError as error:
                # No deletion plan is written before this check; retain bounded state.
                state(directory/'CLEANUP_BLOCKED.json',dict(status='BLOCKED_OPEN_FILE_VISIBILITY',
                    error=str(error),retry='next authorized hourly maintenance',no_deletion_performed=True))
                pending.append(mode if method=='lora' else f'{method}_{mode}')
        state(root/'public/PROGRESS.json',dict(status='GPU_GENERATED_CLEANUP_PENDING' if pending else 'GPU_GENERATED_NOT_SCORED',
            phases=phases,N=cfg['N'],cleanup_pending=pending))


def gate(root, mode, *, method='lora', assignment='EXTERNAL_LORA'):
    """Called instead of GPU training by the existing queue's next child only."""
    cfg=read(root/'private'/f'{assignment}.json')
    if (method,mode) not in assigned_phases(cfg): raise ValueError('Phase not externally assigned')
    while not (root/'private'/f'{assignment}_IMPORTED.json').exists():
        if (root/'private'/f'{assignment}_FAILURE.json').exists():
            raise RuntimeError('External worker/relay failed; see '+assignment+'_FAILURE.json')
        state(root/'public/PROGRESS.json',dict(status='RUNNING',method=method,mode=mode,
            phase='WAITING_FOR_ASSIGNED_EXTERNAL_WORKER',N=cfg['N']))
        time.sleep(60)
    if read(root/'private'/f'{assignment}_IMPORTED.json')!=cfg:
        raise ValueError('External assignment/import mismatch')
    validate_phase(root,cfg,mode,method=method)


def receive(root, assignment='EXTERNAL_LORA'):
    cfg=read(root/'private'/f'{assignment}.json')
    incoming=root/'private'/('external-incoming' if assignment=='EXTERNAL_LORA' else 'external-incoming-baselines')
    phases=assigned_phases(cfg)
    for method,mode in phases:
        validate_phase(incoming,cfg,mode,method=method)
        if (root/'private'/f'{method}_{mode}').exists():
            raise RuntimeError('Refusing to overwrite existing phase')
    for method,mode in phases:
        (incoming/'private'/f'{method}_{mode}').rename(root/'private'/f'{method}_{mode}')
    write_new(root/'private'/f'{assignment}_IMPORTED.json',cfg)


def remote_python(ssh, program):
    return subprocess.run([*ssh,'python3 -'],input=program,text=True,check=True,timeout=60)


def split_source_status(ssh, roots):
    """Inspect only the assigned phase, not an unrelated failure in its source run."""
    program='from pathlib import Path\nimport json\nroots='+repr(roots)+'\nresult={}\n'
    program+='for name,root in roots.items():\n p=Path(root)/"private"/name\n result[name]=("COMPLETE" if (p/"CLEANUP.json").exists() and json.loads((p/"CLEANUP.json").read_text()).get("status")=="DELETED" else "CLEANUP_PENDING") if (p/"COMPLETE.json").exists() else ("FAILED" if (p/"FAILURE.json").exists() else "RUNNING")\nprint(json.dumps(result))\n'
    return json.loads(subprocess.check_output([*ssh,'python3 -'],input=program,text=True,timeout=30))


def relay(cfg):
    """Mac transport only; never retrains or judges. Failure is explicit on both hosts."""
    import shlex
    local=Path(cfg['local']); local.mkdir(exist_ok=True)
    status=local/'RELAY.json'
    assignment_name=cfg.get('assignment_name','EXTERNAL_LORA')
    assignment=read(local/'ASSIGNMENT.json')
    phases=assigned_phases(assignment)
    if 'phase_source_roots' in cfg and set(cfg['phase_source_roots'])!={f'{m}_{mode}' for m,mode in phases}:
        raise ValueError('Split source phase coverage mismatch')
    try:
        state(status,dict(status='WAITING_FOR_EXTERNAL_GPU'))
        while True:
            if 'phase_source_roots' in cfg:
                result=split_source_status(cfg['source_ssh'],cfg['phase_source_roots'])
                if any(v=='FAILED' for v in result.values()): raise RuntimeError('Assigned phase failed: '+str(result))
                if all(v=='COMPLETE' for v in result.values()): break
                state(status,dict(status='WAITING_FOR_ASSIGNED_PHASES',phases=result))
                time.sleep(60)
                continue
            result=json.loads(subprocess.check_output([*cfg['source_ssh'],
                'cat '+shlex.quote(cfg['source_root']+'/public/PROGRESS.json')],text=True,timeout=30))
            if result['status']=='GPU_GENERATED_NOT_SCORED': break
            if result['status'] not in ('RUNNING','GPU_GENERATED_CLEANUP_PENDING'):
                raise RuntimeError('External GPU stopped: '+str(result))
            time.sleep(60)
        state(status,dict(status='TRANSFERRING_SMALL_OUTPUTS'))
        for method,mode in phases:
            name=method+'_'+mode
            source_root=cfg.get('phase_source_roots',{}).get(name,cfg['source_root'])
            args=['rsync','-a','--include=*/','--include=*.json','--include=*.jsonl','--exclude=*']
            subprocess.run([*args,'-e',shlex.join(cfg['source_ssh'][:-1]),
                cfg['source_ssh'][-1]+':'+source_root+'/private/'+name+'/',str(local/name)+'/'],check=True)
        # Validate locally using the same directory shape as the GPU run.
        incoming=local/'payload'; (incoming/'private').mkdir(parents=True,exist_ok=True)
        for method,mode in phases:
            (local/(method+'_'+mode)).rename(incoming/'private'/(method+'_'+mode))
            validate_phase(incoming,assignment,mode,method=method)
        destination=cfg['destination_root']+'/private/'+('external-incoming' if assignment_name=='EXTERNAL_LORA' else 'external-incoming-baselines')
        remote_python(cfg['destination_ssh'],'from pathlib import Path\nPath('+repr(destination)+').mkdir(exist_ok=False)\n')
        subprocess.run(['rsync','-a','-e',shlex.join(cfg['destination_ssh'][:-1]),str(incoming)+'/',
            cfg['destination_ssh'][-1]+':'+destination+'/'],check=True)
        program='import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(cfg['destination_source'])+')\n'
        program+='from scripts.medtrace.stage17_external import receive\nreceive(Path('+repr(cfg['destination_root'])+'),'+repr(assignment_name)+')\n'
        # Use the recorded runtime, not the server's unrelated system Python.
        subprocess.run([*cfg['destination_ssh'],cfg['destination_python']+' -'],input=program,text=True,check=True,timeout=60)
        state(status,dict(status='IMPORTED_FOR_EXISTING_FOLLOWER'))
    except Exception as error:
        failure=dict(status='STOPPED_NO_COMPUTE_RETRY',error=repr(error))
        state(status,failure)
        path=cfg['destination_root']+'/private/'+assignment_name+'_FAILURE.json'
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
