"""Dispatch only audited missing work; every child has an independent hard guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['V131_ROOT'])
lock = json.loads((ROOT / 'PROTOCOL_LOCK.json').read_text())
entries = json.loads((ROOT / 'ENTRY_BINDINGS.json').read_text())


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = ROOT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    with (ROOT / 'COORDINATOR_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original wall')))
    signal.alarm(max(1, int(lock['hard_end'] - time.time() - 10)))
    gate = read(ROOT / 'NO_DUPLICATE_EXECUTION_GATE.json')
    assert gate['new_storage_limit'] == 100000000000 and lock['storage_pre_stop_bytes'] == 98000000000
    assert set(entries) == {'M13P_160','STATISTICS','MEMIT_MM'}
    assert lock['completed_methods_no_new_GPU'] == ['BALANCEDIT','ALPHAEDIT_MM','GRACE']
    jobs = {}
    def launch(method):
        assert method not in jobs and not (ROOT / method / 'GUARD_ONCE').exists()
        entry = entries[method]
        config = read(Path(entry).with_suffix('.json'))
        gpu = config['CUDA_VISIBLE_DEVICES']
        current = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.free',
            '--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert current[0] == config['M3BENCH_FORMAL_EXPECTED_GPU_UUID']
        assert int(current[1]) >= 45 * 1024, 'Admission waits for available memory; no unrelated process is changed'
        with (ROOT/method/'guard.log').open('x') as log:
            process = subprocess.Popen([sys.executable,'-u',entry],stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ,V131_ACTION='guard'))
        identity = dict(pid=process.pid,start_ticks=Path('/proc',str(process.pid),'stat').read_text().split()[21],
            GPU=gpu,UUID=current[0],free_MiB_before=int(current[1]),start=time.time(),entry=entry)
        jobs[method] = (process,identity)
        write('LAUNCH_RECEIPT.json',dict(status='MISSING_WORK_DISPATCHED',
            jobs={m:identity for m,(_,identity) in jobs.items()},time=time.time()))
    launch('M13P_160')
    launch('STATISTICS')
    while time.time() < lock['hard_end'] - 20:
        if (ROOT/'STATISTICS/COMPLETE.json').exists() and 'MEMIT_MM' not in jobs:
            assert read(ROOT/'STATISTICS/READY.json')['counts']['documents'] == 100000
            try:
                launch('MEMIT_MM')
            except AssertionError as error:
                write('MEMIT_ADMISSION_WAIT.json',dict(reason=str(error),time=time.time()))
        states = {method:process.poll() for method,(process,_) in jobs.items()}
        write('COORDINATOR_HEALTH.json',dict(states=states,time=time.time(),
            new_GPU_methods_only=['M13P_missing287','MEMIT_MM'],
            completed_BalancEdit_AlphaEdit_GRACE_repeated=False))
        if all(code is not None for code in states.values()):
            if 'MEMIT_MM' in jobs or states.get('STATISTICS') != 0:
                break
        time.sleep(10)
    write('COORDINATOR_EXIT.json',dict(states={m:p.poll() for m,(p,_) in jobs.items()},
        all_registered_missing_GPU_complete=all((ROOT/m/'COMPLETE.json').exists() for m in entries),time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write('COORDINATOR_FAILURE.json',dict(error=traceback.format_exc(),time=time.time()))
        raise
