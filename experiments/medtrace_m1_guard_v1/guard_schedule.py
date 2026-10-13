"""Adopt an existing prefix worker and dispatch only the remaining partition."""
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import guard_probe as g
from module_recovery import live
import pipeline


def admission(memory):
    assert memory['stage']=='TRAIN_AND_NODE_CHECKPOINT' and not memory['peak_counters_reset']
    assert memory.get('process_sample_valid',memory['sampled_process_peak_bytes']>0) and memory['includes_loading']
    required=memory['peak_reserved_bytes']+max(1024**3,memory['sampled_nonallocator_max_bytes'])+4*1024**3
    assert required<=39*1024**3, 'Measured peak plus margin exceeds proposed admission'
    return dict(status='PASS',required_free_MiB=39*1024,measured=memory,
                nonallocator_reserve_bytes=max(1024**3,memory['sampled_nonallocator_max_bytes']),
                additional_reserve_bytes=39*1024**3-memory['peak_reserved_bytes']-max(1024**3,memory['sampled_nonallocator_max_bytes']))


def main():
    c,run=g.c,g.RUN
    marker=run/'private/SCHEDULE_HANDOFF1.json'
    assert not marker.exists()
    parent=c.read(run/'private/TAKEOVER_START1.json')
    adopted=c.read(run/'private/START_CHAIN_guard_probe_1.json')
    assert live(parent) and live(adopted)
    assert all((run/'private/PREFIX_COMPLETE'/f'{owner}.json').exists() for owner in (1,4,7))
    assert not (run/'private/START_CHAIN_guard_probe_2.json').exists()
    memory=c.read(run/'private/MEMORY_guard_probe_1.json')
    assert memory['pid']==adopted['pid']
    # Both arms and same-state protection diagnostics must precede lower admission.
    assert (run/'private/diagnostics/2_20.json').exists()
    node=c.read(run/'private/diagnostics/2_20.json')
    assert node['owner']==2 and node['node']==20 and node['shared_state']=='GUARD_PREFIX'
    assert memory['epoch']>=(run/'private/diagnostics/2_20.json').stat().st_mtime
    policy=admission(memory)
    gpu=4
    expected=c.read(run/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    def available():
        row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        assert row[0]==expected
        return int(row[1])>=policy['required_free_MiB']
    assert available(), 'GPU4 does not meet measured admission'
    with marker.open('x') as handle:
        import json
        json.dump(dict(status='RESERVED_ONCE',epoch=time.time()),handle)
    # Pause only the waiting CPU scheduler to exclude a duplicate dispatch race.
    os.kill(parent['pid'],signal.SIGSTOP)
    try:
        assert live(parent) and live(adopted)
        assert not (run/'private/START_CHAIN_guard_probe_2.json').exists()
        assert not (run/'private/TAKEOVER_EXIT1.json').exists()
        assert available()
        c.write(marker,dict(status='SCHEDULER_ONLY_SUPERSEDED',original_parent=parent,
            adopted_worker=adopted,pending_partition=2,epoch=time.time(),model_workers_restarted=0))
        c.write(run/'private/MEMORY_ADMISSION.json',policy)
        os.kill(parent['pid'],signal.SIGTERM)
        os.kill(parent['pid'],signal.SIGCONT)
    except BaseException:
        if live(parent):os.kill(parent['pid'],signal.SIGCONT)
        raise
    job=pipeline.launch('guard_probe.py','guard_probe',gpu,2)
    while job.poll() is None or live(adopted):
        c.budget()
        assert job.poll() is None or job.returncode==0, 'Remaining partition failed; retain state'
        if not live(adopted):
            assert all((run/'private/PREFIX_COMPLETE'/f'{owner}.json').exists() for owner in (2,5,8)), 'Adopted worker stopped unfinished'
        time.sleep(30)
    assert job.returncode==0
    assert all((run/'private/PREFIX_COMPLETE'/f'{owner}.json').exists() for owner in range(1,9))
    pipeline.wait([pipeline.launch('guard_report.py','guard_report')])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        g.c.write(g.RUN/'private/FAILURE_guard_schedule1.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
