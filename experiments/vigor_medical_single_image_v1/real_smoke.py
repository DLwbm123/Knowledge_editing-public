"""Bound official runtime loading, coefficient diagnosis, KV and recovery in one call."""
import copy
import json
import inspect
import threading
import os
from pathlib import Path
import signal
import time

import torch
from .coefficient_runtime import coefficient_forward, compose, native_weight, validate
from .dual_execution import ExecutionBudget, Journal
from .dual_guard import spawn_guard, verify_guard
from .editor import editable_fp32, finite, nll, paired_batches
from .kv_caller import paired_kv_generation

GPU = 'GPU-4b56ab28-a018-652c-b8f6-2d9dc3dc45d2'
MODULE = 'model.layers.6.mlp.down_proj'
CONFIG = dict(model_forward_max=320, work_forward_max=300, backward_API_max=4,
              model_gradient_paths_max=4, work_seconds=240, TERM_seconds=270,
              KILL_seconds=300, per_generation_forward_max=128,
              complete_generation_requests_max=2)


class SmokeBudget(ExecutionBudget):
    def __init__(self, start):
        super().__init__(CONFIG, start)
        self.non_generation = 0

    def before_forward(self, model, args):
        if not self.generating and not self.restoring:
            if self.non_generation >= 44:
                raise TimeoutError('Non-generation cap')
            self.non_generation += 1
        if torch.cuda.memory_allocated() > 36 * 1024**3:
            raise MemoryError('Owned GPU allocation cap')
        super().before_forward(model, args)


def validate_binding(m):
    source = m.get('source', {})
    if not all(source.get(k) is True for k in ('current_use_permitted', 'reserved_exclusion_certified', 'fit_exclusion_certified')):
        raise PermissionError('Source admission missing')
    if source.get('split') != 'train' or not source.get('binding_id'):
        raise PermissionError('Concrete source binding missing')
    if not all(m.get('resources', {}).get(k) is True for k in ('mount_probe', 'disk_baseline', 'neutral_argv')):
        raise PermissionError('Resource receipts missing')
    if m.get('GPU_UUID') != GPU or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise PermissionError('Fixed device visibility required')
    if not m.get('input') or m.get('model_snapshot') != '91bb16c122001ddc9cf1fd36ce1dae09448943a2' or not m.get('F0_binding'):
        raise PermissionError('Input/model/F0 binding missing')
    for key in ('runtime_root','model_path','vision_path','generation_config_path','runtime_class_file','model_class_file'):
        if not m.get(key):
            raise PermissionError('Reviewed loading binding missing')
    record = source.get('record', {})
    if not all(record.get(k) for k in ('image_path', 'question', 'target')):
        raise PermissionError('Concrete source record missing')
    if m.get('generation', {}).get('max_new_tokens') != 1024 or m['generation'].get('do_sample') is not False or m['generation'].get('num_beams') != 1:
        raise PermissionError('Frozen generation required')


def run(manifest, output_root, neutral_directory):
    """Linux neutral process-group worker; no launch on import or absent binding."""
    validate_binding(manifest)  # No model imports or GPU operations before rejection.
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Guarded worker requires main thread')
    start = time.time()
    journal = Journal(output_root, 256 * 1024**2)
    journal.private('LAUNCH_RESERVED.json', {'start_epoch': start, 'manifest': manifest})
    guard = spawn_guard(neutral_directory, start, CONFIG)
    verify_guard(guard)
    journal.private('GUARD.json', guard)
    old_handler = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise TimeoutError('Independent smoke guard TERM')
    signal.signal(signal.SIGTERM, terminate)
    budget = SmokeBudget(start)
    runtime = None
    try:
        # The existing launch PYTHONPATH supplies the pinned package and custom LLaVA.
        from m3bench_repro.editors.llava_runtime import LlavaMedEditorRuntime, EditorRecord
        if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 48 * 1024**3:
            raise MemoryError('Frozen GPU free-memory admission failed')
        import transformers
        if transformers.__version__ != '4.51.3' or Path(inspect.getfile(LlavaMedEditorRuntime)).resolve() != Path(manifest['runtime_class_file']).resolve():
            raise RuntimeError('Reviewed runtime package mismatch')
        lock = json.loads((Path(manifest['runtime_root']) / 'locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json').read_text())
        if lock.get('selected_runtime') != 'runtime_b_official_native':
            raise PermissionError('Official native runtime lock required')
        torch.cuda.reset_peak_memory_stats()
        runtime = LlavaMedEditorRuntime(device='cuda:0', run_root=Path(manifest['runtime_root']),
            model_path=Path(manifest['model_path']), vision_path=Path(manifest['vision_path']),
            generation_config_path=Path(manifest['generation_config_path']), loader_mode='official_native')
        runtime.load_frozen_backbone(seed=20260828)
        budget.check_time()
        if type(runtime.model).__name__ != 'LlavaMistralForCausalLM' or runtime.generation_config != manifest['generation']:
            raise RuntimeError('Actual loaded interface mismatch')
        if Path(inspect.getfile(type(runtime.model))).resolve() != Path(manifest['model_class_file']).resolve():
            raise RuntimeError('Actual custom model source mismatch')
        if torch.cuda.mem_get_info()[0] < 12 * 1024**3 or torch.cuda.memory_allocated() > 36 * 1024**3:
            raise MemoryError('Post-load workspace admission failed')
        row = manifest['source']['record']
        record = EditorRecord(record_id='CURRENT', dataset=manifest['source'].get('dataset','BOUND'),
            question=row['question'], target=row['target'], image_path=Path(row['image_path']),
            official_rephrase=row['question'], relative_image_path='', formal_sequence_position=0, question_type='BOUND')
        layer = runtime.get_module(MODULE)
        validate(runtime.model, layer)
        if layer.weight.dtype != torch.float16 or list(layer.weight.shape) != [4096, 14336]:
            raise RuntimeError('Frozen native matrix mismatch')
        result = diagnose(runtime, layer, record, manifest, budget, journal)
        journal.private('RESULT.json', dict(result, source_binding=manifest['source']['binding_id'],
            model_snapshot=manifest['model_snapshot'], module=MODULE, native_dtype='float16',
            actual_class=type(runtime.model).__name__, generation=manifest['generation']))
        return result
    except BaseException as error:
        journal.event({'status':'FAILED_NO_RETRY', 'exception_type':type(error).__name__, 'forwards':budget.forwards})
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def diagnose(runtime, layer, record, manifest, budget, journal):
    model = runtime.model
    base = layer.weight.detach().clone()
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    config = copy.deepcopy(runtime.generation_config)
    hook_state = [(m, dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in model.modules()]
    baseline = pair = None
    hook = model.register_forward_pre_hook(budget.before_forward)
    try:
        model.eval()
        pair = paired_batches(runtime, record)
        budget.check_time()
        raw = runtime.adapter.prepare_inputs(record.image_path, record.question, None)
        expected = manifest['input']
        if raw['input_ids'].tolist() != expected['prompt_ids'] or raw['attention_mask'].tolist() != expected['attention_mask'] or raw['image_sha256'] != expected['existing_image_binding']:
            raise PermissionError('Actual input binding mismatch')
        with torch.no_grad():
            baseline = [float(nll(model,b)) for b in pair]
        with editable_fp32(model, layer, readonly=True):
            loss = nll(model, pair[0])
            original_loss = float(loss.detach())
            gradient = finite(budget.grad(loss, (layer.weight,), 1)[0].detach())
        j = int(gradient.square().sum(0).argmax())
        u = gradient[:, j:j+1].clone()
        u = finite(u / u.norm())
        v = finite(gradient.T @ u)
        v = finite(v / v.norm())
        expected_derivative = float((u.T @ gradient @ v).item())
        del gradient, loss
        a = torch.zeros((1,1), device=base.device, requires_grad=True)
        with coefficient_forward(model, layer, u, a, v):
            loss = nll(model, pair[0])
            derivative = float(budget.grad(loss, (a,), 1)[0].item())
            if abs(float(loss.detach())-original_loss) > 1e-5 or abs(derivative-expected_derivative) > 1e-6+.001*abs(expected_derivative):
                raise RuntimeError('Coefficient derivative/loss mismatch')
        del loss
        coefficient = torch.full((1,1), float(base.float().norm())*1e-4, device=base.device)
        candidate = compose(base,u,coefficient,v).half()
        if torch.equal(candidate,base):
            raise RuntimeError('Native diagnostic collapsed')
        with native_weight(model,layer,candidate):
            kv = paired_kv_generation(runtime,raw,budget,
                lambda cached: journal.private('CACHE_'+str(cached)+'_RESERVED.json',{'epoch':time.time()}))
        result = dict(kv, status='PASS', coefficient_gradient_abs_error=abs(derivative-expected_derivative))
    finally:
        budget.generating = False
        budget.restoring = True
        try:
            with torch.no_grad():
                layer.weight.data = base.clone()
            for p,flag in parameters:
                p.requires_grad_(flag)
            # Loss restoration uses the same eval mode as the recorded W0 baseline.
            model.eval()
            runtime.generation_config.clear(); runtime.generation_config.update(config)
            if not torch.equal(layer.weight,base) or any(p.grad is not None for p,_ in parameters):
                raise RuntimeError('Native/gradient restoration failed')
            if baseline is not None:
                with torch.no_grad():
                    restored = [float(nll(model,b)) for b in pair]
                if any(abs(a-b)>1e-5 for a,b in zip(restored,baseline)):
                    raise RuntimeError('W0 NLL restoration failed')
        finally:
            for m,mode in modes:
                m.training = mode
            hook.remove()
        if any(dict(m._forward_hooks)!=h or dict(m._forward_pre_hooks)!=p for m,h,p in hook_state) or 'forward' in layer.__dict__:
            raise RuntimeError('Hook/forward restoration failed')
        journal.event({'status':'RESTORED' if baseline is not None else 'WEIGHTS_RESTORED_NO_BASELINE','forwards':budget.forwards})
    if torch.cuda.max_memory_allocated() > 36 * 1024**3:
        raise MemoryError('Peak allocated memory exceeded frozen cap')
    result.update(forwards=budget.forwards, backward_API=budget.backwards,
        elapsed_seconds=time.monotonic()-budget.started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


def main():
    # Neutral launcher sets these through its environment; no private argv fields.
    manifest = json.loads(Path(os.environ['RUN_BINDING']).read_text())
    run(manifest, os.environ['RUN_OUTPUT'], os.environ['RUN_GUARD_DIR'])


if __name__ == '__main__':
    main()
