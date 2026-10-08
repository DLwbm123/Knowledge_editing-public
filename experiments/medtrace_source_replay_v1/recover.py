"""Adopt completed/live evaluation shards; resume only the admitted failed shard."""
import json
import os
import time
from pathlib import Path
import replay as exp
import pipeline

c,RUN=exp.c,exp.RUN


def alive(start):
    path=Path('/proc')/str(start['pid'])/'stat'
    return path.exists() and path.read_text().split()[21]==start['start_ticks'] and path.read_text().split()[2]!='Z'


def main():
    archive=RUN/'private/engineering_recovery1'
    admission=c.read(archive/'ADMISSION.json')
    assert admission['generated_queries_before_failure']==0 and not admission['validation_relaxed']
    assert not alive(c.read(archive/'START_CHAIN_replay_eval_7.json'))
    assert not (RUN/'private/START_CHAIN_replay_eval_7.json').exists(), 'Never duplicate a resumed shard'
    pipeline.launch('replay.py','replay_eval',7,7)
    while not all((RUN/f'private/EVAL_{i}.json').exists() for i in range(8)):
        c.budget()
        for i in range(8):
            if not (RUN/f'private/EVAL_{i}.json').exists():
                assert alive(c.read(RUN/f'private/START_CHAIN_replay_eval_{i}.json')) or (RUN/f'private/EVAL_{i}.json').exists(), 'A shard stopped; inspect before another recovery'
        time.sleep(5)
    # The old controller exits after observing its original failed child's exit code.
    while alive(c.read(RUN/'private/START.json')):
        c.budget();time.sleep(5)
    error=RUN/'private/FAILURE_replay_controller_all.json'
    if error.exists():error.rename(archive/error.name)
    assert not list((RUN/'private').glob('FAILURE_*.json'))
    exp.consumers();exp.p.done('GENERATION_COMPLETE')
    pipeline.wait([pipeline.launch('replay_queue.py','replay_ingest')])
    removed=[]
    for folder in ('weights','teacher'):
        for path in (RUN/'private'/folder).rglob('*.pt'):
            assert not path.is_symlink() and path.resolve().is_relative_to((RUN/'private'/folder).resolve())
            removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    exp.p.done('DELETION',dict(files=removed,all_consumers_complete=True,bindings_durable=True,historical_weights_untouched=True,rebuild='retrain'))
    exp.p.done('ENGINEERING_RECOVERY',dict(resumed_shards=[7],unchanged_GPU_source=True,unchanged_validation=True,
        original_failure_and_cost_retained=True,retraining=False,score_retry=False))
    exp.p.progress('ASTRA_SCORING');root=RUN/'private/judge_replay_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('replay_report.py','replay_report')])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        c.write(RUN/'private/RECOVERY_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
