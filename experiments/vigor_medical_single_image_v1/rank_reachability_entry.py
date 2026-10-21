"""Load the admitted development runtime and execute the frozen two-arm comparison."""
import json
import os
from pathlib import Path
import signal
import time

import torch

from .dual_execution import run_bound_comparison, FREEZES
from .dual_guard import spawn_guard
from .editor import paired_batches
from .real_smoke import GPU, MODULE, validate_binding


def main(config_name='vigor_rankReachability_v1_20261021', endpoint_consumer=None):
    root = Path(os.environ['RUN_PARENT'])
    manifest = json.loads((root / 'private/binding.json').read_text())
    validate_binding(manifest)
    kv = json.loads(Path(os.environ['RUN_KV_RECEIPT']).read_text())
    if kv['status'] != 'PASS' or kv['source_binding'] != manifest['source']['binding_id']:
        raise PermissionError('Actual same-source smoke PASS required')
    config = json.loads((Path(__file__).resolve().parents[2] / 'reports' / config_name / 'FREEZE.json').read_text())
    config['source_binding'] = manifest['source']['binding_id']
    continuation_file=root/'private/CONTINUATION.json'
    continuation=json.loads(continuation_file.read_text()) if continuation_file.exists() else None
    start = time.time() if continuation is None else continuation['start_epoch']
    with (root / 'private/RUN_RESERVED.json').open('x') as f:
        json.dump({'start_epoch': start}, f)
    guard = spawn_guard(os.environ['RUN_GUARD_DIR'], start, config)
    def terminate(signum, frame):
        raise TimeoutError('Independent dual guard TERM')
    signal.signal(signal.SIGTERM, terminate)
    deployment = json.loads((root / 'private/deployment.json').read_text())['config']
    if os.environ['CUDA_VISIBLE_DEVICES'] != GPU or torch.cuda.mem_get_info()[0] < 48 * 1024**3:
        raise MemoryError('Fixed GPU memory admission')
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    from m3bench_repro.editors.llava_runtime import LlavaMedEditorRuntime, EditorRecord
    runtime = LlavaMedEditorRuntime(device='cuda:0', run_root=Path(manifest['runtime_root']),
        model_path=Path(manifest['model_path']), vision_path=Path(manifest['vision_path']),
        generation_config_path=Path(manifest['generation_config_path']), loader_mode='official_native')
    runtime.load_frozen_backbone(seed=20260828)
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise MemoryError('Workspace reserve')
    row = manifest['source']['record']
    record = EditorRecord(record_id='CURRENT', dataset=manifest['source']['dataset'],
        question=row['question'], target=row['target'], image_path=Path(row['image_path']),
        official_rephrase=row['question'], relative_image_path='', formal_sequence_position=0, question_type='BOUND')
    calibration = torch.load(deployment['calibration'], map_location='cpu', weights_only=True)
    if calibration['binding'] != manifest['F0_binding']:
        raise PermissionError('Original calibration binding mismatch')
    factor = calibration['factors'][MODULE]
    manifest.update(KV=kv, F0=dict(model_snapshot=manifest['model_snapshot'], module=MODULE,
        native_dtype='float16'), start_epoch=start, guard=guard)
    def memory_guard(model, args):
        if torch.cuda.memory_allocated() > 36 * 1024**3:
            raise MemoryError('Owned memory cap')
    hook = runtime.model.register_forward_pre_hook(memory_guard)
    try:
        from .rank_reachability import ReachBudget
        from .rank_reachability import run
        from .dual_guard import verify_guard
        verify_guard(guard)
        if runtime.generation_config != manifest['generation'] or manifest['generation'] != kv['generation']:
            raise PermissionError('Generation configuration mismatch')
        (root/'private/EXECUTION_BINDING.json').write_text(json.dumps(dict(config=config,manifest=manifest)))
        budget=ReachBudget(config,start)
        if continuation is not None:
            budget.forwards=continuation['forwards']
            budget.backwards=continuation['backward_API']
            budget.gradient_paths=continuation['model_gradient_paths']
            budget.requests=continuation['generation_requests']
        counter=runtime.model.register_forward_pre_hook(budget.before_forward)
        try:
            run(runtime,runtime.get_module(MODULE),record,paired_batches(runtime,record),factor,config,budget,root,replay_plan=None if continuation is None else continuation['accepted_path'],endpoint_consumer=endpoint_consumer)
        finally:
            counter.remove()
        peak = torch.cuda.max_memory_allocated()
        if peak > 36 * 1024**3:
            raise MemoryError('Peak owned memory cap')
        with (root / 'private/RUN_COMPLETE.json').open('x') as f:
            json.dump({'elapsed_seconds': time.time()-start, 'peak_allocated_bytes': peak}, f)
    finally:
        hook.remove()


if __name__ == '__main__':
    main()
