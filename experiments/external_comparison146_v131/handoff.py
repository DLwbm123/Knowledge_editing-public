"""One-shot, identity-checked handoff in the existing CPU prefill boundary."""
import json
import os
from pathlib import Path
import signal
import time


def read(path):
    return json.loads(Path(path).read_text())


def identity(pid, ticks):
    path = Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split()[21] == ticks


def main():
    root = Path(os.environ['V131_ROOT'])
    out = root/'MEMIT_MM'
    cfg = read(os.environ['V131_HANDOFF_CONFIG'])
    assert cfg['human_authorized_parallel_inference'] is True
    assert cfg['source_public_verified'] is True
    plan = [row for row in read(root/'EVAL_PLAN.json')
            if row['method'] == 'MEMIT_MM' and row['mode'] != 'single' and row['prefix'] == cfg['prefix']]
    ids = [json.dumps([row[k] for k in ('method', 'mode', 'prefix', 'owner')]+[row['query']['query_id']],
                      separators=(',', ':')) for row in plan]
    positions = {key: index for index, key in enumerate(ids)}
    target = cfg['not_before_index']
    while time.time() < cfg['hard_end']-120:
        assert identity(cfg['worker_pid'], cfg['worker_start_ticks'])
        assert identity(cfg['guard_pid'], cfg['guard_start_ticks'])
        progress = read(out/'PROGRESS.json')
        index = positions.get(progress.get('last_consumer'), -1)
        # evaluate() prepares the next 24 prompts only after this durable
        # output/progress pair. No language generation is in flight then.
        if progress['phase'] == 'GENERATE' and target <= index < len(plan)-24 and (index+1)%24 == 0:
            assert time.time()-progress['time'] < .5
            os.kill(cfg['worker_pid'], signal.SIGINT)
            trigger = dict(time=time.time(), row_index=index,
                           last_complete_counts=progress['counts'], planned_engineering_handoff=True)
            (out/'PARALLEL_HANDOFF_TRIGGER.json').write_text(json.dumps(trigger, indent=2)+'\n')
            break
        time.sleep(.02)
    else:
        raise TimeoutError('No safe prefill handoff boundary before original deadline')
    until = time.monotonic()+40
    while time.monotonic() < until and not (out/'GUARD_EXIT.json').exists():
        time.sleep(.1)
    failure, ended = read(out/'FAILURE.json'), read(out/'WORKER_EXIT.json')
    counts = failure['counts']
    assert counts == ended['counts']
    assert 'KeyboardInterrupt' in failure['error']
    assert counts['generation_attempts'] == counts['formal_consumers']
    assert counts['formal_consumers'] == len(list((out/'outputs').glob('*.json')))
    for key in ('updates', 'backwards', 'edits', 'inserted'):
        assert counts[key] == cfg['unchanged_training_counts'][key]
    assert read(out/'GUARD_EXIT.json')['code'] != 0
    result = dict(status='PASS', time=time.time(), counts=counts,
                  no_inflight_generation_lost=True, no_completed_consumer_repeated=True,
                  prior_total_process_seconds=read(out/'GUARD_EXIT.json')['total_method_process_seconds'])
    (out/'PARALLEL_HANDOFF_ADMISSION.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
