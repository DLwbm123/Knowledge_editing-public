"""Fresh campaign runtime; local snapshots only, with an independent resource ledger."""
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ['RUN_ROOT'])
GPU = os.environ.get('PINNED_GPU_UUID', '')
LAYER = 'model.layers.30.mlp.down_proj'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def check_budget(*, training=False, p0=False):
    manifest = read(ROOT/'RUN_MANIFEST.json')
    deadline = manifest['p0_deadline_at' if p0 else 'no_new_training_after' if training else 'deadline_at']
    if (ROOT/'STOP').exists() or time.time() >= epoch(deadline):
        raise TimeoutError('STOP or persisted campaign deadline')
    if shutil.disk_usage(ROOT).free < 8*1024**3:
        raise OSError('8 GiB disk reserve reached')
    ledger = read(ROOT/'RESOURCE_LEDGER.json')
    used = ledger['gpu_seconds_used']
    used += sum(time.time()-s['started_epoch'] for s in ledger['gpu_sessions'] if s.get('ended_epoch') is None)
    if used >= ledger['gpu_seconds_limit']:
        raise TimeoutError('GPU resident budget exhausted')


@contextlib.contextmanager
def ledger_lock():
    with (ROOT/'RESOURCE_LEDGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read(ROOT/'RESOURCE_LEDGER.json')
        yield ledger
        write(ROOT/'RESOURCE_LEDGER.json', ledger)


@contextlib.contextmanager
def gpu_session(purpose):
    check_budget(p0=purpose=='P0_RUNTIME')
    actual = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    lines = [x.split(', ') for x in actual.strip().splitlines()]
    selected = [x for x in lines if x[0] == GPU]
    if len(selected) != 1 or int(selected[0][1]) < 20000:
        raise RuntimeError('Pinned GPU missing or less than 20000 MiB free')
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    started = time.time()
    with ledger_lock() as ledger:
        if sum(x.get('ended_epoch') is None for x in ledger['gpu_sessions']) >= 2 or any(x.get('ended_epoch') is None and x['gpu_uuid'] == GPU for x in ledger['gpu_sessions']):
            raise RuntimeError('Both authorized GPU slots busy or same GPU already active')
        ledger['gpu_sessions'].append(dict(pid=os.getpid(), purpose=purpose, gpu_uuid=GPU, started_epoch=started))
        ledger['status'] = 'GPU_ACTIVE'
    try:
        yield
    finally:
        ended = time.time()
        with ledger_lock() as ledger:
            session = next(x for x in ledger['gpu_sessions'] if x['started_epoch'] == started)
            session.update(ended_epoch=ended, resident_seconds=ended-started)
            ledger['gpu_seconds_used'] += ended-started
            ledger['status'] = 'GPU_SESSION_CLOSED'


def load_runtime(run_root, seed):
    source = ROOT/'source/third_party/LLaVA-Med'
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.environ.update(M3BENCH_LLAVA_SOURCE=str(source), M3BENCH_EXPECTED_LLAVA_SOURCE=str(source),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    from m3bench_repro.editors.llava_runtime import LlavaMedEditorRuntime
    runtime = LlavaMedEditorRuntime(device='cuda:0', run_root=run_root,
        model_path=ROOT/'models/llava-med-v1.5-mistral-7b', vision_path=ROOT/'models/vision',
        generation_config_path=ROOT/'source/freshstart/generation.json', loader_mode='official_native')
    runtime.load_frozen_backbone(seed=seed)
    runtime.resolve_module_inventory(freeze=True)
    module = runtime.get_module(LAYER)
    if tuple(module.weight.shape) != (4096, 14336):
        raise ValueError('L30 down_proj shape mismatch')
    return runtime


def smoke():
    import torch
    receipt = dict(status='STARTING', pid=os.getpid(), training=False, gpu_uuid=GPU)
    write(ROOT/'GPU_SMOKE.json', receipt)
    try:
        with gpu_session('P0_RUNTIME'):
            runtime = load_runtime(ROOT/'runtime_check', 20260923)
            task = read(ROOT/'private/FRESH_TASKS.json')['tasks'][0]
            from m3bench_repro.editors.llava_runtime import EditorRecord
            row = task['native']
            record = EditorRecord(task['canonical_edit_id'], row['dataset'], row['question'], row['reference'],
                task['fit_questions'][0], Path(row['image_path']), '', task['order'], 'SOURCE_LABEL', 'FRESH_DEV16')
            batch = runtime.build_edit_batch(record)
            with torch.inference_mode():
                loss = runtime.compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite model forward')
            assert runtime.base_guard.verify()['unchanged']
            modules = {}
            for name in ['m3bench_repro.editors.llava_runtime', 'm3bench_repro.inference.llava_med',
                         'llava.model.language_model.llava_mistral', 'methods.medtrace.selective_write', __name__]:
                path = Path(importlib.import_module(name).__file__).resolve()
                modules[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            receipt.update(status='PASS', target_layer=LAYER, shape=list(runtime.get_module(LAYER).weight.shape),
                dtype=str(next(runtime.model.parameters()).dtype), tokenizer=type(runtime.adapter.tokenizer).__name__,
                finite_forward=True, base_unchanged=True, modules=modules,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                git_head=subprocess.check_output(['git', '-C', str(ROOT/'source'), 'rev-parse', 'HEAD'], text=True).strip())
    except Exception as error:
        receipt.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['completed_at'] = datetime.now(timezone.utc).isoformat()
        write(ROOT/'GPU_SMOKE.json', receipt)


if __name__ == '__main__':
    smoke()
