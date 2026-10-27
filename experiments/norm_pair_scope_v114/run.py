"""CPU-only norm-and-coordinate relation classifier with fixed group validation."""
import json
import os
from pathlib import Path
import signal
import time
import traceback

ROOT = Path(os.environ['V114_ROOT'])
FEATURE_ROOT = Path(os.environ['PARENT_FEATURE_ROOT'])
START = time.time()
CPU_START = time.process_time()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 2000000


def fit(torch, rows):
    groups = sorted({r['task_id'] for r in rows})
    x = torch.stack([r['x'] for r in rows])
    y = torch.tensor([r['y'] for r in rows], dtype=torch.float64)
    weights = torch.tensor([1 / (2 * len(groups) * sum(v['task_id'] == r['task_id'] and v['y'] == r['y'] for v in rows))
                            for r in rows], dtype=torch.float64)
    assert abs(float(weights.sum()) - 1) < 1e-12
    mean_x = (weights[:, None] * x).sum(0)
    mean_y = (weights * y).sum()
    a = (x - mean_x) * weights.sqrt()[:, None]
    q = (y - mean_y) * weights.sqrt()
    beta = torch.linalg.solve(a @ a.T + torch.eye(len(rows), dtype=torch.float64), q)
    w = a.T @ beta
    b = mean_y - mean_x @ w
    assert torch.isfinite(w).all() and torch.isfinite(b)
    return w, b


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    lock = read(ROOT / 'PROTOCOL_LOCK.json')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Five-minute CPU deadline')))
    signal.alarm(max(1, int(min(300, lock['hard_end'] - time.time() - 2))))
    write(ROOT / 'WORKER_IDENTITY.json', dict(pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().split()[21], start=START))
    import torch
    torch.set_num_threads(1)
    synthetic = [dict(task_id=g, y=y, x=torch.tensor(x, dtype=torch.float64))
        for g in ('A', 'B') for y, x in ((1, [.1, .05]), (-1, [.9, .8]))]
    test_w, test_b = fit(torch, synthetic)
    assert all(float(r['x'] @ test_w + test_b) * r['y'] > 0 for r in synthetic) and test_b > 0
    write(ROOT / 'CPU_SELF_CHECK.json', dict(status='PASS', weighted_dual_and_intercept=True,
        synthetic_only=True, model_calls=0))
    tasks = read(ROOT / 'TASKS.json')['tasks']
    features = torch.load(FEATURE_ROOT / 'FEATURES.pt', map_location='cpu', weights_only=True)
    assert len(features) == 126 and all(v.shape == (14336,) and torch.isfinite(v).all() for v in features.values())
    def pair(row, anchors):
        value = features[row['query_id']]
        distances = torch.stack([(value - features[v['query_id']]).norm() for v in anchors])
        nearest = anchors[int(distances.argmin())]
        difference = value - features[nearest['query_id']]
        return torch.cat((difference.norm().reshape(1), difference.abs()))
    rows = []
    for task in tasks:
        if task['role'] != 'DEV_CANDIDATE':
            continue
        for index, row in enumerate(task['train']):
            anchors = task['train'][:index] + task['train'][index + 1:]
            rows.append(dict(task_id=task['task_id'], query_id=row['query_id'],
                             y=1, x=pair(row, anchors)))
        for row in task['protect']:
            assert row['relation'] != task['target']['relation']
            rows.append(dict(task_id=task['task_id'], query_id=row['query_id'],
                             y=-1, x=pair(row, task['train'])))
    assert len(rows) == 58 and sum(r['y'] == 1 for r in rows) == 40
    predictions = []
    for group in sorted({r['task_id'] for r in rows}):
        training = [r for r in rows if r['task_id'] != group]
        held = [r for r in rows if r['task_id'] == group]
        w, b = fit(torch, training)
        for row in held:
            score = float(row['x'] @ w + b)
            predictions.append(dict(task_id=group, query_id=row['query_id'], label=row['y'],
                score=score, correct=score * row['y'] > 0))
        write(ROOT / 'PROGRESS.json', dict(folds_completed=len({r['task_id'] for r in predictions}),
            evaluated=len(predictions), errors=sum(not r['correct'] for r in predictions), time=time.time()))
    assert len(predictions) == 58
    errors = sum(not r['correct'] for r in predictions)
    write(ROOT / 'GROUP_CV_FROZEN.json', dict(predictions=predictions, errors=errors, rows=58,
        groups=8, source_labels_only=True, fit_held_six_inputs_used=False,
        source_commit=lock['source_commit'], time=time.time()))
    if errors:
        write(ROOT / 'COMPLETE.json', dict(status='NORM_PAIR_GROUP_CV_FAILED', CV_errors=errors,
            final_head_fitted=False, cache_replay=False, new_GPU_seconds=0, new_generations=0, time=time.time()))
        return
    w, b = fit(torch, rows)
    if not b > 0:
        write(ROOT / 'COMPLETE.json', dict(status='NORM_PAIR_SELF_RECOGNITION_FAILED',
            b=float(b), cache_replay=False, new_GPU_seconds=0, new_generations=0, time=time.time()))
        return
    torch.save(dict(w=w, b=b), ROOT / 'HEAD.pt')
    write(ROOT / 'HEAD_FROZEN.json', dict(parameters=w.numel() + 1, lambda_value=1,
        class_and_group_balanced=True, b=float(b), CV_errors=0, time=time.time()))
    routes = {}
    for task in tasks:
        routes[task['task_id']] = {}
        for row in task['train'] + task['heldout'] + task['protect']:
            score = float(pair(row, task['train']) @ w + b)
            routes[task['task_id']][row['query_id']] = dict(score=score, ON=score > 0)
        assert all(routes[task['task_id']][r['query_id']]['ON'] for r in task['train'])
    write(ROOT / 'ALL_ROUTING_FROZEN.json', dict(routes=routes, time=time.time(),
        source='fixed_Base_features_and_frozen_linear_head', output_answers_used=False))
    assert not torch.cuda.is_initialized()
    write(ROOT / 'COMPLETE.json', dict(status='CACHE_REPLAY_READY', CV_errors=0,
        final_head_parameters=w.numel() + 1, new_GPU_seconds=0, new_generations=0,
        final_head_fitted=True, cache_replay=True, time=time.time()))


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(ROOT / 'FAILURE.json', dict(error=traceback.format_exc(), time=time.time()))
        raise
    finally:
        write(ROOT / 'WORKER_EXIT.json', dict(code=code, wall_seconds=time.time() - START,
            CPU_process_seconds=time.process_time() - CPU_START, GPU_seconds=0, time=time.time()))
