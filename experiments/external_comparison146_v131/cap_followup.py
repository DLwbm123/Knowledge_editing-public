"""Finite dependency tail for the human amendment to three total GPUs.

Let the retained disjoint shards finish. Settle the four externally interrupted
requests as UNKNOWN without retry, then resume only unattempted consumers under
the same clocks and cumulative GPU-process/operation/storage limits.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def cid(row):
    return json.dumps([row[key] for key in ('method','mode','prefix','owner')]
                      +[row['query']['query_id']], separators=(',', ':'))


def interrupted(out, actual_ids):
    enforcement = read(out/'GPU3_CAP_ENFORCEMENT.json')
    assert enforcement['maximum_total_GPUs'] == 3
    assert enforcement['allowed'] == [0,2,4]
    result = []
    for stopped in enforcement['stopped']:
        jobdir = out/'parallel/prefix50'/str(stopped['gpu'])
        start, end, counts = (read(jobdir/(name+'.json')) for name in ('START','END','COUNTS'))
        failure = read(jobdir/'FAILURE.json')
        assert start['pid'] == stopped['pid'] and end['code'] == 1
        assert 'SystemExit: Guard stop' in failure['error']
        assert failure['counts'] == counts
        assert counts['generation_attempts'] == counts['formal_consumers']+1
        rows = read(jobdir/'JOB.json')['rows']
        assert all(cid(row) in actual_ids for row in rows[:counts['formal_consumers']])
        row = rows[counts['formal_consumers']]
        identity = cid(row)
        assert identity not in actual_ids
        result.append(dict(consumer_id=identity, gpu=stopped['gpu'], pid=start['pid'],
            status='UNKNOWN_USER_GPU_CAP_INTERRUPTION', formal_retry_allowed=False,
            failure_record=str((jobdir/'FAILURE.json').relative_to(out))))
    assert len(result) == len({value['consumer_id'] for value in result}) == 4
    return result


def identity(pid, ticks):
    path = Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split()[21] == ticks


def main(cfg):
    root, neutral = Path(cfg['root']), Path(cfg['neutral'])
    out = root/'MEMIT_MM'
    assert cfg['maximum_total_GPUs'] == 3 and cfg['source_public_verified'] is True
    assert read(out/'CAP3_SOURCE_DEPLOYED.json')['source'] == cfg['source']
    with (out/'CAP3_FOLLOWUP_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original deadline')))
    signal.alarm(max(1, int(cfg['method_original_deadline']-time.time()-30)))
    while identity(cfg['old_guard_pid'], cfg['old_guard_start_ticks']):
        assert not (out/'CAP3_CANCEL.json').exists()
        time.sleep(5)
    assert not (out/'CAP3_CANCEL.json').exists()
    failure, ended = read(out/'FAILURE.json'), read(out/'GUARD_EXIT.json')
    assert 'Stop on any formal generation failure; no implicit retry' in failure['error']
    merged = read(out/'parallel/prefix50/MERGED.json')
    counts = failure['counts']
    assert merged['counters_including_prior_attempts'] == counts
    assert counts['edits'] == 196 and counts['inserted'] == 50
    assert counts['updates'] == 3304 and counts['backwards'] == 20484
    actual = {read(path)['consumer_id'] for path in (out/'outputs').glob('*.json')}
    assert len(actual) == counts['formal_consumers']
    aborted = interrupted(out, actual)
    assert counts['generation_attempts'] == counts['formal_consumers']+len(aborted)
    write(out/'CAP3_ABORTED_CONSUMERS.json', dict(time=time.time(), maximum_total_GPUs=3,
        formal_retry_allowed=False, consumers=aborted, complete=False,
        original_plan_consumers=7494, source='Direct human resource-cap amendment'))
    old = read(out/'PARALLEL2_LOCK.json')
    prior = max(ended['total_method_process_seconds'], ended.get('total_GPU_process_seconds',0))
    assert prior < old['method_process_seconds']['MEMIT_MM']-200
    assert time.time() < cfg['method_original_deadline']-200
    # A resource wait consumes the original wall clock, never resets it.
    while True:
        assert not (out/'CAP3_CANCEL.json').exists()
        uuid, free = subprocess.check_output(['nvidia-smi','-i','4',
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert uuid == cfg['primary_UUID']
        if int(free) >= 45*1024:
            break
        assert time.time() < cfg['method_original_deadline']-200
        time.sleep(30)
    archive = out/'gpu3_cap_phase_settlement'
    archive.mkdir(exist_ok=False)
    moved = []
    for name in ['GUARD_ONCE','WORKER_ONCE','WORKER_IDENTITY.json','PROGRESS.json','FAILURE.json',
                 'WORKER_EXIT.json','GUARD_EXIT.json','GUARD_HEALTH.json','STOP.json',
                 'ALGEBRA_CHECK.json','PREFILL_ADMISSION.json','PRODUCTION_MATRIX_ADMISSION.json',
                 'RESUME_ADMISSION.json','worker.log']:
        path = out/name
        if path.exists():
            assert not path.is_symlink()
            moved.append(dict(name=name,bytes=path.stat().st_size))
            shutil.move(str(path),str(archive/name))
    write(out/'CAP3_ARCHIVE_RECEIPT.json',dict(time=time.time(),files=moved,
        metadata_only=True,weights_outputs_training_preserved=True))
    lock = dict(old)
    lock.update(execution_identity=cfg['source'], source_public_verified=True,
        prior_counts={'MEMIT_MM':counts}, prior_process_seconds={'MEMIT_MM':prior},
        maximum_parallel_GPUs=3, parallel_devices=[0,2], parallel_epoch='cap3',
        aborted_consumers=aborted, aborted_failure_receipt='CAP3_ABORTED_CONSUMERS.json')
    for key in ['maximum_counts','MEMIT_MM','parameter_edit_layers','method_process_seconds',
                'hard_end','method_original_deadline','storage_limit_bytes','storage_pre_stop_bytes']:
        assert lock[key] == old[key], ('Frozen field changed', key)
    write(out/'CAP3_LOCK.json',lock)
    env = read(neutral/'n.json')
    env.update(NEUTRAL_ENTRY=str(neutral/'t.py'),
        V131_GUARD_FILE=str(neutral/'r5/guard.py'),
        V131_JOB_FILE=str(neutral/'r5/parameter_run.py'),
        V131_PROTOCOL_FILE=str(out/'CAP3_LOCK.json'),V131_PARALLEL_DEVICES='[0,2]',
        M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES='0,2,4')
    (neutral/'t.py').write_text((neutral/'n.py').read_text())
    write(neutral/'t.json',env)
    start = time.time()
    with (neutral/'t.log').open('x') as log:
        child = subprocess.Popen([env['TRAIN_PYTHON'],'-u',str(neutral/'t.py')],
            cwd=neutral,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'CAP3_DISPATCH.json',dict(time=start,pid=child.pid,
        start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],source=cfg['source'],
        public_verified=True,maximum_total_GPUs=3,primary_GPU=4,helper_GPUs=[0,2],
        prior_total_GPU_process_seconds=prior,prior_counts=counts,
        interrupted_consumers_unknown=len(aborted),no_completed_consumer_repeated=True,
        original_method_deadline=cfg['method_original_deadline'],hard_end=old['hard_end']))


if __name__ == '__main__':
    config = read(os.environ['V131_CAP_FOLLOWUP_CONFIG'])
    destination = Path(config['root'])/'MEMIT_MM'
    try:
        main(config)
    except BaseException:
        write(destination/'CAP3_FOLLOWUP_BLOCKED.json',dict(time=time.time(),error=traceback.format_exc(),
            no_automatic_retry=True))
        raise
