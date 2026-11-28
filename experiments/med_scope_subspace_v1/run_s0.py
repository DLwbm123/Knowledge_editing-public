"""Run S0 using two GPU workers and a common deadline, with isolated artifacts."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def read(name):
    return json.loads(Path(name).read_text())


def record(name, data):
    with Path(name).open('x') as out:
        json.dump(data,out,allow_nan=False,ensure_ascii=False)


def owned_bytes(root):
    return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def run_group(name, entries, devices, deadline, cap):
    jobs = []
    root = Path.cwd()
    def limit():
        resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**3,4*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    try:
        for i,(args,gpu) in enumerate(zip(entries,devices,strict=True)):
            argv = ['w',*args]
            payload = ('import sys;from experiments.med_scope_subspace_v1.experiment import main;'
                       'sys.argv='+repr(argv)+';main()')
            source = Path(f'{name}.{i}.in')
            source.write_text(payload)
            with source.open() as stdin,Path(f'{name}.{i}.log').open('x') as out:
                jobs.append(subprocess.Popen(['python3','-'],stdin=stdin,stdout=out,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),preexec_fn=limit))
        record(f'{name}.start.json',dict(pids=[x.pid for x in jobs],gpus=devices,started=time.time()))
        while any(p.poll() is None for p in jobs):
            if time.time()>=deadline-30 or owned_bytes(root)>=cap or any(
                    p.poll() not in (None,0) for p in jobs):
                raise RuntimeError(f'{name} failed, reached its deadline, or reached its storage cap')
            time.sleep(10)
        codes = [p.wait() for p in jobs]
        record(f'{name}.exit.json',dict(codes=codes,ended=time.time(),owned_bytes=owned_bytes(root)))
        if any(codes):
            raise RuntimeError(f'{name} exited with {codes}')
    finally:
        for p in jobs:
            if p.poll() is None:
                p.terminate()
        for p in jobs:
            if p.poll() is None:
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


def main():
    settings = read('chain.json')
    deadline,cap = settings['deadline'],settings['storage_cap']
    devices = settings['gpus']
    assert devices==[6,7]
    if owned_bytes(Path.cwd())>=cap:
        raise ValueError('Storage budget already exhausted')

    def gpu(name, stage, extra, manifest='panel.json', extra_each=None):
        entries = []
        for shard in range(2):
            const = [stage,'--manifest',manifest,'--contract',f'config{shard}.json',
                     '--shards','2','--shard',str(shard)]
            entries.append(const+extra+['--output',f'{name}{shard}.'+('pt' if stage=='collect' else 'json')]
                           +(extra_each(shard) if extra_each else []))
        run_group(name,entries,devices,deadline,cap)

    gpu('parity','collect',['--part','development','--roles','native','fit'],
        extra_each=lambda i:['--references',f'ref{i}.pt'])
    gpu('prefix','prefix',['--part','development','--roles','train_vis','val_neg','bg'])
    invalid = [r for i in range(2) for r in read(f'prefix{i}.json')['rejected']]
    background = {r['id'] for r in invalid if r['role']=='bg'}
    rejected_roles = sorted({r['role'] for r in invalid})
    record('prefix_admission.json',dict(rejected=len(invalid),background_rejected=len(background),
                                     rejected_roles=rejected_roles))
    if len(invalid)!=len(background) or len(background)>50:
        raise ValueError('Visual prefix failed or >5% background prefixes are unusable')
    manifest = read('panel.json')
    manifest['rows'] = [r for r in manifest['rows'] if r['id'] not in background]
    record('panel_run.json',manifest)
    # Keep an admitted-only prefix bundle after checking the digest of the original.
    from experiments.med_scope_subspace_v1.runtime import FrozenRuntime
    for i in range(2):
        p = read(f'prefix{i}.json')
        if FrozenRuntime.digest(p['outputs'])!=p['digest']:
            raise ValueError('Preflight prefix payload changed')
        kept = {k:v for k,v in p['outputs'].items() if k not in background}
        record(f'prefix_admitted{i}.json',dict(digest=FrozenRuntime.digest(kept),outputs=kept))
    gpu('dev','collect',['--part','development','--roles','train_eq','train_ind','train_vis',
         'bg','val_pos','val_neg','--base-prefixes','prefix_admitted0.json','prefix_admitted1.json'],
        manifest='panel_run.json')
    run_group('solve',[['solve','--manifest','panel_run.json','--features',
        'parity0.pt','parity1.pt','dev0.pt','dev1.pt','--references','ref0.pt','ref1.pt',
        '--device','cuda:0','--output','solution.pt']],[6],deadline,cap)
    gpu('held','collect',['--part','evaluation','--solution','solution.pt','--roles',
        'hold_pos','hold_ind','hold_vis','official_ind'],manifest='panel_run.json')
    run_group('report',[['s0','--manifest','panel_run.json','--features','held0.pt','held1.pt',
                         '--solution','solution.pt','--output','s0_results.private.json']],
              [''],deadline,cap)
    record('complete.json',dict(state='S0_COMPLETE',ended=time.time(),owned_bytes=owned_bytes(Path.cwd())))


if __name__=='__main__':
    main()
