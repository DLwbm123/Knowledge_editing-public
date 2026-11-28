"""Exact S0 acceleration using a shared background Gram and disjoint device shards."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch

from .features import load_manifest, read_features
from .run_s0 import owned_bytes, record, run_group
from .scope_solver import CovarianceMoment, Moment, row_basis
from .solve_panel import solve


def read(path):
    return json.loads(Path(path).read_text())


def atomic_state(path, obj):
    tmp = path.with_suffix('.pending')
    with tmp.open('xb') as out:
        torch.save(obj,out)
    os.link(tmp,path)
    tmp.unlink()


def inputs():
    manifest = load_manifest('panel_run.json')
    expected = {r['id']:r for r in manifest['rows']}
    data = read_features(['parity0.pt','parity1.pt','dev0.pt','dev1.pt'])
    for key,v in data.items():
        r = expected[key]
        if (v['role'],v['owner'],v['image'])!=(r['role'],r.get('owner'),r['source_sha256']):
            raise ValueError('Feature or image binding changed')
    parity = {k for k,v in data.items() if v.get('projection_parity')}
    if len(parity)!=5 or parity!=set(manifest['projection_parity']):
        raise ValueError('Frozen MED projection parity missing')
    if not any(v.get('zero_residual_parity') for v in data.values()):
        raise ValueError('Frozen Base zero-residual parity missing')
    return manifest,data


def shared_matrix(path):
    _,data = inputs()
    queries = [v['feature'] for v in data.values() if v['role']=='bg']
    rows = Moment.from_queries(queries)
    gram = rows.rows.T @ rows.rows
    if not torch.isfinite(gram).all():
        raise ValueError('Nonfinite background Gram matrix')
    atomic_state(path/'background.pt',dict(matrix=gram,queries=len(queries),
                          positions=len(rows.rows),manifest=manifest_digest()))
    print('BACKGROUND_READY',len(queries),len(rows.rows),flush=True)


def manifest_digest():
    return sha256(Path('panel_run.json').read_bytes()).hexdigest()


def worker(path, lane, ceiling, sample_owner=None):
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Expected precisely one shard GPU')
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(ceiling/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(12)
    manifest,data = inputs()
    refs = {}
    for i in range(2):
        out = torch.load(f'ref{i}.pt',map_location='cpu',weights_only=True)
        if set(refs)&set(out):
            raise ValueError('Overlapping reference owners')
        refs.update(out)
    owner_list = sorted({r['owner'] for r in manifest['rows'] if r['role']=='native'})
    assert len(owner_list)==30 and set(refs)==set(owner_list)
    cached = torch.load(path/'background.pt',map_location='cpu',weights_only=True)
    if cached['manifest']!=manifest_digest():
        raise ValueError('Background matrix belongs to a different manifest')
    gram = cached['matrix']
    expected = owner_list[lane::2]
    sample = expected[0] if sample_owner is None else sample_owner
    if sample not in expected:
        raise ValueError("Parity edit must come from the specified device shard")
    test_g4 = row_basis(refs[sample]['MED']['G4'][:,:,0].double().to(device))
    background = Moment.from_queries([x['feature'] for x in data.values() if x['role']=='bg'],device)
    compressed = CovarianceMoment(gram.to(device))
    torch.testing.assert_close(compressed.tt_gram(test_g4,112),background.tt_gram(test_g4,112),rtol=2e-8,atol=1e-9)
    candidate = refs[sample]['MED']['G3'].double().to(device)
    columns = (candidate @ test_g4).flatten(1).T
    torch.testing.assert_close(compressed.gram(columns),background.gram(columns),rtol=2e-8,atol=1e-9)
    del compressed,background
    print('GRAM_PARITY',len(expected),flush=True)

    def checkpoint(owner,value):
        atomic_state(path/f'owner-{owner}.pt',value)
        print('DONE',owner,flush=True)
    if sample_owner is not None:
        profile = dict.fromkeys(('g3','compression','adam','unrestricted'), 0.)
        started = time.monotonic()
        solve(data,refs,device,owners=[sample],background_gram=gram,save_owner=checkpoint,profile=profile)
        profile['total'] = time.monotonic() - started
        return profile

    remaining = [owner for owner in expected if not (path/f'owner-{owner}.pt').is_file()]
    if remaining:
        solve(data,refs,device,owners=remaining,background_gram=gram,save_owner=checkpoint)
    record(path/f'lane{lane}.done.json',dict(owners=expected,ended=time.time()))


def check(path, lane, ceiling, owner, reference):
    profile = worker(path, lane, ceiling, owner)
    print('PROFILE', json.dumps(profile), flush=True)
    old = torch.load(reference,map_location='cpu',weights_only=True)
    new = torch.load(path/f'owner-{owner}.pt',map_location='cpu',weights_only=True)
    differences = {}
    for variant in sorted(old):
        if not variant.startswith('K-'):
            continue
        for field in ('gamma','visual_weight','objective','eigenvalues','trace','val_gap'):
            if field not in old[variant]:
                continue
            before = torch.tensor(old[variant][field], dtype=torch.float64)
            after = torch.tensor(new[variant][field], dtype=torch.float64)
            torch.testing.assert_close(after,before,rtol=1e-9,atol=1e-12,msg=f'{variant} {field}')
            error = float(((after - before).abs() / before.abs().clamp_min(1e-12)).max())
            differences[field] = max(differences.get(field,0.),error)
    record(path/'compression.parity.json',dict(owner=owner,profile=profile,max_relative=differences))
    print('COMPRESSION_PARITY', json.dumps(differences), flush=True)


def launch(path, stage, args, device=None, threads=12):
    overlay = str(path/'source')
    command = ('import sys;sys.path.insert(0,'+repr(overlay)+');'
               'from experiments.med_scope_subspace_v1.parallel_s0 import main;'
               'sys.argv='+repr(['w',stage,str(path),*args])+';main()')
    script = path/f'{stage}.in'
    script.write_text(command)
    def limit():
        resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**3,4*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    with script.open() as stdin,(path/f'{stage}.log').open('x') as out:
        return subprocess.Popen(['python3','-'],stdin=stdin,stdout=out,stderr=subprocess.STDOUT,
           env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if device is None else str(device),
                    OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads)),
           preexec_fn=limit)


def wait(jobs,cap):
    try:
        while any(p.poll() is None for p in jobs):
            if owned_bytes(Path.cwd())>=cap or any(p.poll() not in (None,0) for p in jobs):
                raise RuntimeError('Parallel worker failed or reached storage cap')
            if Path('solve.continued.exit.json').exists() and Path('solution.pt').is_file():
                return False
            time.sleep(12)
        if any(p.wait()!=0 for p in jobs):
            raise RuntimeError('Parallel worker failed')
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
    return True


def manager(path):
    info = read('handoff.released.json')
    cap = info['storage_cap']
    assert info['deadline'] is None and owned_bytes(Path.cwd())<cap-2_000_000_000
    if not (path/'background.pt').is_file():
        b = launch(path,'background',[],threads=16)
        record(path/'background.start.json',dict(pid=b.pid,started=time.time()))
        if not wait([b],cap):
            return
    free = [int(v) for v in subprocess.check_output(['nvidia-smi','-i','6,7',
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(free)==2 and min(free)>=14000
    jobs = [launch(path,f'lane{i}',[str(i),str(min(f-4500,14000)*1024**2)],device=gpu)
            for i,(gpu,f) in enumerate(zip((6,7),free,strict=True))]
    record(path/'workers.json',dict(pids=[p.pid for p in jobs],gpus=[6,7],started=time.time()))
    if not wait(jobs,cap):
        return
    manifest=load_manifest('panel_run.json')
    expected={r['owner'] for r in manifest['rows'] if r['role']=='native'}
    got = {}
    for i in range(2):
        for o in read(path/f'lane{i}.done.json')['owners']:
            assert o not in got
            got[o] = torch.load(path/f'owner-{o}.pt',map_location='cpu',weights_only=True)
            assert {'K-TT','K-U','K-P4','K-MED','K-MDU','K-TT-med','K-TT-p4','K-TT-nobg','K-TT-novis'}<=got[o].keys()
    assert set(got)==expected and len(got)==30
    candidate=path/'solution.pt'
    atomic_state(candidate,dict(manifest_digest=manifest_digest(),solutions=got))
    try:
        os.link(candidate,Path('solution.pt'))
    except FileExistsError:
        record(path/'delivery.json',dict(selected='original',ours_completed=len(got),ended=time.time()))
        return
    record(path/'delivery.json',dict(selected='parallel_verified',owners=len(got),ended=time.time()))
    run_evaluation(cap)


def run_evaluation(cap):
    while True:
        if Path('config0.active.json').exists():
            return
        if owned_bytes(Path.cwd())>=cap:
            raise RuntimeError('Evaluation reached storage cap')
        free = [int(value) for value in subprocess.check_output(['nvidia-smi','-i','6,7',
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
        if len(free)==2 and min(free)>=43500:
            break
        time.sleep(30)
    held=[]
    for shard,memory in enumerate(free):
        config = Path(f'config{shard}.active.json')
        values = read(f'config{shard}.continued.json')
        values['memory_bytes'] = min(values['memory_bytes'], (memory-3500)*1024**2)
        try:
            record(config, values)
        except FileExistsError:
            if shard:
                raise
            return
        held.append(['collect','--manifest','panel_run.json','--contract',str(config),
                     '--shards','2','--shard',str(shard),'--part','evaluation','--solution','solution.pt',
                     '--roles','hold_pos','hold_ind','hold_vis','official_ind','--output',f'held{shard}.pt'])
    run_group('held',held,[6,7],math.inf,cap)
    run_group('report',[['s0','--manifest','panel_run.json','--features','held0.pt','held1.pt',
        '--solution','solution.pt','--output','s0_results.private.json']],[''],math.inf,cap)
    record('complete.json',dict(state='S0_COMPLETE',ended=time.time(),owned_bytes=owned_bytes(Path.cwd()),
                                deadline_cancelled=True,parallel=True))


def main():
    stage,root,*args=sys.argv[1:]
    path=Path(root)
    if stage=='background':
        torch.set_num_threads(16)
        shared_matrix(path)
    elif stage=='manager':
        manager(path)
    elif stage=='check':
        lane,ceiling,owner = map(int,args[:3])
        check(path,lane,ceiling,owner,args[3])
    else:
        worker(path,*map(int,args))


if __name__=='__main__':
    main()
