"""One finite, identity-bound handoff for the explicit completion amendment."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def identity(pid, ticks):
    p = Path('/proc') / str(pid) / 'stat'
    return p.exists() and p.read_text().split()[21] == str(ticks)


def main():
    cfg = read(Path(__file__).parent / 'COMPLETION_HANDOFF_CONFIG.json')
    root, neutral = Path(cfg['root']), Path(cfg['neutral'])
    out = root / 'MEMIT_MM'
    assert cfg['human_request'] == '不要有 gpu 时间预算，跑完所有的'
    assert identity(cfg['worker_pid'], cfg['worker_ticks'])
    assert identity(cfg['guard_pid'], cfg['guard_ticks'])
    initial = read(out / 'PROGRESS.json')
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        progress = read(out / 'PROGRESS.json')
        if progress['phase'] == 'INSERTION_COMPLETE' and progress['time'] > initial['time']:
            break
        assert identity(cfg['worker_pid'], cfg['worker_ticks'])
        time.sleep(.02)
    else:
        raise TimeoutError('No completed insertion boundary; healthy worker unchanged')
    assert identity(cfg['worker_pid'], cfg['worker_ticks'])
    # Stop at the next existing budget callback; its exception records exact costs.
    os.kill(cfg['worker_pid'], signal.SIGSTOP)
    write(out / 'STOP.json', dict(reason='HUMAN_COMPLETION_LIMIT_AMENDMENT_HANDOFF', time=time.time()))
    os.kill(cfg['worker_pid'], signal.SIGCONT)
    deadline = time.monotonic() + 90
    while identity(cfg['guard_pid'], cfg['guard_ticks']) and time.monotonic() < deadline:
        time.sleep(.2)
    assert not identity(cfg['guard_pid'], cfg['guard_ticks'])
    assert not identity(cfg['worker_pid'], cfg['worker_ticks'])
    end, failure = read(out / 'WORKER_EXIT.json'), read(out / 'FAILURE.json')
    assert end['code'] == 1 and 'AssertionError' in failure['error']
    counts = end['counts']
    assert counts == failure['counts']
    assert all(counts[k] == progress['counts'][k] for k in ('edits','inserted','generation_attempts','formal_consumers')), 'No in-flight formal generation or completed edit may be lost'
    assert all(counts[k] >= progress['counts'][k] for k in counts)
    assert len(list((out / 'outputs').glob('*.json'))) == counts['formal_consumers']
    assert counts['generation_attempts'] == counts['formal_consumers'] + 4
    import torch
    saved = torch.load(out / 'sequential_resume.pt', map_location='cpu', weights_only=True)
    assert saved['owner'] == progress['owner'] == counts['inserted']
    assert saved['source'] == cfg['old_source']
    overhead = {k: counts[k] - saved['counts'][k] for k in ('updates', 'backwards', 'edits', 'inserted')}
    assert all(v >= 0 for v in overhead.values()) and not overhead['edits'] and not overhead['inserted']
    # A saved state precedes its final generation; that completed answer is reused.
    assert counts['formal_consumers'] >= saved['counts']['formal_consumers']
    guard_end = read(out / 'GUARD_EXIT.json')
    lock = read(out / 'CAP3_LOCK.json')
    lock.update(run_until_completion=True, execution_identity=cfg['new_source'],
                resume_source_execution_identity=saved['source'],
                handoff_uncommitted_counts=overhead, parallel_epoch='completion',
                completion_amendment=dict(human_request=cfg['human_request'], time=time.time(),
                    GPU_time_limit_removed=True, wall_deadline_removed=True,
                    original_start_preserved=True, maximum_total_GPUs=3))
    lock['prior_counts']['MEMIT_MM'] = counts
    lock['prior_process_seconds']['MEMIT_MM'] = guard_end['total_GPU_process_seconds']
    lock['parallel_total_GPU_process_budget_unchanged'] = False
    write(out / 'COMPLETION_LOCK.json', lock)
    archive = out / 'completion_limit_handoff'
    archive.mkdir(exist_ok=False)
    names = ['GUARD_ONCE','WORKER_ONCE','STOP.json','FAILURE.json','WORKER_EXIT.json',
             'GUARD_EXIT.json','GUARD_HEALTH.json','WORKER_IDENTITY.json','PROGRESS.json',
             'worker.log','guard.log','RESUME_ADMISSION.json','PRODUCTION_MATRIX_ADMISSION.json']
    moved = []
    for name in names:
        p = out / name
        if p.exists():
            p.rename(archive / name)
            moved.append(name)
    env = read(neutral / 't.json')
    env.update(NEUTRAL_ENTRY=str(neutral / 'u.py'), V131_GUARD_FILE=str(neutral / 'r6/guard.py'),
               V131_JOB_FILE=str(neutral / 'r6/parameter_run.py'),
               V131_PROTOCOL_FILE=str(out / 'COMPLETION_LOCK.json'))
    write(neutral / 'u.json', env)
    (neutral / 'u.py').write_text((neutral / 't.py').read_text())
    memory = subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    assert int(memory.strip()) >= 45 * 1024
    assert os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 8 * 1024**3
    log = (out / 'guard.log').open('x')
    child = subprocess.Popen([sys.executable,'-u',str(neutral / 'u.py')], stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    write(out / 'COMPLETION_DISPATCH.json', dict(time=time.time(), pid=child.pid,
          start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],
          source=cfg['new_source'], resume_owner=saved['owner'], prior_counts=counts,
          handoff_uncommitted_counts=overhead, prior_GPU_process_seconds=guard_end['total_GPU_process_seconds'],
          archived_metadata=moved, run_until_completion=True, maximum_total_GPUs=3))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(Path(__file__).parent / 'COMPLETION_HANDOFF_BLOCKED.json',
              dict(time=time.time(), error=traceback.format_exc()))
        raise
