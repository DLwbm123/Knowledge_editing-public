"""Operational split: pinned BELoRA phases on one host, receipts bridged to its queue.

The phase interpreter imports the frozen JOB_SOURCE checkout. This controller
never changes scientific configuration or imports its own checkout as runtime.
"""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

MODES = ('single', 'sequential')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def frozen_api():
    sys.path.insert(0, os.environ['JOB_SOURCE'])
    from scripts.medtrace import stage17_campaign, stage17_external
    return stage17_campaign, stage17_external


def selected(cfg):
    if cfg.get('operational_assignment') != 'BELORA_GPU3_SPLIT_V1':
        raise ValueError('Explicit split assignment required')
    return [('belora', mode) for mode in MODES]


def worker(cfg):
    campaign, external = frozen_api()
    root = Path(cfg['run'])
    with (root / 'private/campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roles = campaign.role_map(read(Path(cfg['source_run']) / 'private/COHORT_AND_SUPPORT_LEDGER.json'))
        rolefile = root / 'private/PREFIX_ACTIVE_TARGETS.json'
        if rolefile.exists() and read(rolefile) != roles:
            raise ValueError('Prefix roles changed')
        write(rolefile, roles)
        pending = []
        for method, mode in selected(cfg):
            campaign.check_space(root)
            directory = root / 'private' / f'{method}_{mode}'
            if not (directory / 'COMPLETE.json').exists():
                free = int(subprocess.check_output(['nvidia-smi', '-i', str(cfg['gpu']),
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                if free < 22000:
                    raise RuntimeError('Insufficient GPU peak margin')
                write(root / 'public/PROGRESS.json', dict(status='RUNNING', method=method, mode=mode, completed=0, N=cfg['N']))
                env = dict(os.environ, JOB_ACTION='phase', JOB_METHOD=method, JOB_MODE=mode)
                with (root / f'{method}_{mode}.log').open('ab') as log:
                    subprocess.run([cfg['python'], '-u', cfg['entry']], env=env,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            external.validate_phase(root, cfg, mode, method=method, require_cleanup=False)
            try:
                campaign.cleanup(cfg, method, mode)
            except campaign.CheckpointVisibilityError as error:
                write(directory / 'CLEANUP_BLOCKED.json', dict(status='BLOCKED_OPEN_FILE_VISIBILITY', error=str(error)))
                pending.append(mode)
        write(root / 'public/PROGRESS.json', dict(status='GPU_GENERATED_CLEANUP_PENDING' if pending else 'GPU_GENERATED_NOT_SCORED',
            phases=selected(cfg), N=cfg['N'], cleanup_pending=pending))


def gate(cfg):
    _, external = frozen_api()
    method, mode = os.environ['JOB_METHOD'], os.environ['JOB_MODE']
    if (method, mode) not in selected(cfg):
        raise ValueError('Unassigned remote phase')
    root = Path(cfg['run'])
    while not (root / 'private' / f'{method}_{mode}' / 'CLEANUP.json').exists():
        if (root / 'STOP').exists() or (root / 'private/SPLIT_FAILURE.json').exists():
            raise RuntimeError('Split stopped; inspect STOP/SPLIT_FAILURE')
        write(root / 'public/PROGRESS.json', dict(status='RUNNING', method=method, mode=mode,
            phase='WAITING_FOR_GPU3_RESULTS', N=cfg['N']))
        time.sleep(60)
    external.validate_phase(root, cfg, mode, method=method)


def bridge(cfg):
    _, external = frozen_api()
    local = Path(cfg['local'])
    assignment = read(local / 'ASSIGNMENT.json')
    selected(assignment)
    status = local / 'RELAY.json'
    write(status, dict(status='WAITING_FOR_GPU3'))
    while True:
        result = json.loads(subprocess.check_output([*cfg['source_ssh'],
            'cat ' + shlex.quote(cfg['source_root'] + '/public/PROGRESS.json')], text=True, timeout=30))
        if result['status'] == 'GPU_GENERATED_NOT_SCORED':
            break
        if result['status'] not in ('RUNNING', 'GPU_GENERATED_CLEANUP_PENDING'):
            raise RuntimeError('GPU3 stopped: ' + str(result))
        time.sleep(60)
    incoming = local / 'payload'
    (incoming / 'private').mkdir(parents=True, exist_ok=True)
    write(status, dict(status='TRANSFERRING_SMALL_OUTPUTS'))
    args = ['rsync', '-a', '--include=*/', '--include=*.json', '--include=*.jsonl', '--exclude=*']
    for _, mode in selected(assignment):
        name = 'belora_' + mode
        subprocess.run([*args, '-e', shlex.join(cfg['source_ssh'][:-1]),
            cfg['source_ssh'][-1] + ':' + cfg['source_root'] + '/private/' + name + '/',
            str(incoming / 'private' / name) + '/'], check=True)
        external.validate_phase(incoming, assignment, mode, method='belora')
    destination = cfg['destination_root'] + '/private/split-incoming'
    subprocess.run([*cfg['destination_ssh'], 'mkdir -p ' + shlex.quote(destination)], check=True)
    subprocess.run(['rsync', '-a', '-e', shlex.join(cfg['destination_ssh'][:-1]),
        str(incoming) + '/', cfg['destination_ssh'][-1] + ':' + destination + '/'], check=True)
    program = 'import sys,json\nfrom pathlib import Path\nsys.path.insert(0,' + repr(cfg['destination_source']) + ')\n'
    program += 'from scripts.medtrace.stage17_external import validate_phase\n'
    program += 'root=Path(' + repr(cfg['destination_root']) + ')\ncfg=json.loads((root/"private/DISPATCH.json").read_text())\n'
    program += 'incoming=root/"private/split-incoming"\n'
    program += 'for mode in ("single","sequential"):\n validate_phase(incoming,cfg,mode,method="belora")\n'
    program += 'for mode in ("single","sequential"):\n name="belora_"+mode\n target=root/"private"/name\n if target.exists(): validate_phase(root,cfg,mode,method="belora")\n else: (incoming/"private"/name).rename(target)\n'
    subprocess.run([*cfg['destination_ssh'], cfg['destination_python'] + ' -'], input=program, text=True, check=True, timeout=60)
    write(status, dict(status='IMPORTED_TO_BASELINE_QUEUE'))


if __name__ == '__main__':
    config = read(os.environ['JOB_CONFIG'])
    action = os.environ['JOB_ACTION']
    try:
        {'split_worker': worker, 'split_gate': gate, 'split_bridge': bridge}[action](config)
    except Exception as error:
        failure = dict(status='STOPPED', action=action, error=repr(error))
        if action == 'split_worker':
            write(Path(config['run']) / 'public/PROGRESS.json', failure)
        elif action == 'split_bridge':
            write(Path(config['local']) / 'RELAY.json', failure)
            program = 'from pathlib import Path\nPath(' + repr(config['destination_root'] + '/private/SPLIT_FAILURE.json') + ').write_text(' + repr(json.dumps(failure)) + ')\n'
            subprocess.run([*config['destination_ssh'], 'python3 -'], input=program, text=True, timeout=30)
        raise
