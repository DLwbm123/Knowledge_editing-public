"""Production caller and private one-shot score handoff; no implicit launch/data reads."""
import copy
import json
import math
import os
import signal
from pathlib import Path
import time
import threading

import torch

from .dual_solver import Budget, endpoint_native, solve_pair_comparison
from .editor import nll

ARMS = ('IND2', 'JOINT2')

def validate_source(source):
    if source.get('mode') == 'existing_development_source':
        if (source.get('binding_id') != 'DEV_E01_20261021'
                or source.get('original_edit_alias') != 'E01'
                or source.get('current_use_permitted') is not True
                or source.get('original_record_exact') is not True
                or source.get('calibration_exclusion_certified') is not True
                or source.get('independent_evaluation_claim') is not False):
            raise PermissionError('Existing development source binding incomplete')
        return
    required = ('current_use_permitted', 'reserved_exclusion_certified', 'fit_exclusion_certified')
    if not all(source.get(k) is True for k in required) or source.get('split') != 'train':
        raise PermissionError('Source exclusion/use binding incomplete')
PUBLIC_KEYS = {
    'arm', 'step', 'last_step', 'pre_update_losses', 'objective', 'forwards',
    'backward_API', 'model_gradient_paths', 'status', 'exception_type',
    'endpoint', 'generation_forwards', 'generated_token_count', 'termination',
}


def public_event(value):
    if isinstance(value, dict):
        if set(value) - PUBLIC_KEYS:
            raise ValueError('Non-public journal field')
        return {k: public_event(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_event(v) for v in value]
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('Non-scalar public journal value')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite public journal value')
    return value


class Journal:
    def __init__(self, root, cap):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        (self.root / 'private').mkdir(mode=0o700)
        (self.root / 'public').mkdir()
        self.cap = cap - 16 * 1024**2

    def size(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def private(self, name, value):
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()
        if self.size() + len(data) > self.cap:
            raise OSError('Disk budget exhausted')
        # Exclusive writes leave durable reservation on uncertain failure: never retry.
        with (self.root / 'private' / name).open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    def event(self, row):
        data = (json.dumps(public_event(row), allow_nan=False) + '\n').encode()
        if self.size() + len(data) > self.cap:
            raise OSError('Disk budget exhausted')
        with (self.root / 'public' / 'EVENTS.jsonl').open('ab') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())


def validate_manifest(manifest, config, record):
    frozen_path = Path(__file__).resolve().parents[2] / 'reports/vigor_dualStable_v1_20261021/FREEZE.json'
    frozen = json.loads(frozen_path.read_text())
    if {k: v for k, v in config.items() if k != 'source_binding'} != {k: v for k, v in frozen.items() if k != 'source_binding'}:
        raise PermissionError('Frozen method parameters changed')
    source, f0, kv = (manifest[k] for k in ('source', 'F0', 'KV'))
    validate_source(source)
    binding = source.get('binding_id')
    if not binding or config['source_binding'] != binding or kv.get('source_binding') != binding:
        raise PermissionError('Source/KV/config binding mismatch')
    if kv.get('status') != 'PASS' or kv.get('actual_class') != 'LlavaMistralForCausalLM':
        raise PermissionError('Need actual production KV smoke PASS')
    for key in ('model_snapshot', 'module', 'native_dtype'):
        if not f0.get(key) or f0[key] != kv.get(key):
            raise PermissionError('F0/KV runtime identity mismatch')
    if f0['module'] != config['module'] or f0['native_dtype'] != 'float16':
        raise PermissionError('Frozen module/dtype mismatch')
    expected = source['record']
    if (str(Path(record.image_path).resolve()) != expected['image_path']
            or record.question != expected['question'] or record.target != expected['target']):
        raise PermissionError('Concrete private source record mismatch')
    if manifest['generation'] != kv['generation']:
        raise PermissionError('Generation config differs from admitted smoke')
    generation = manifest['generation']
    if (generation.get('do_sample') is not False or generation.get('num_beams') != 1
            or generation.get('max_new_tokens') != 1024 or generation.get('use_cache') is not True):
        raise PermissionError('Frozen complete generation config required')
    if (config['arms'] != list(ARMS) or len(config['init_Z_E11']) != 2
            or config['complete_generation_requests_max'] != 4):
        raise PermissionError('Need fixed two arms/two endpoints')
    return binding


class ExecutionBudget(Budget):
    def __init__(self, config, start_epoch):
        super().__init__(config)
        if not math.isfinite(start_epoch) or start_epoch > time.time():
            raise ValueError('Invalid original clock')
        self.started -= time.time() - start_epoch
        self.generating = False
        self.request_forwards = 0
        self.requests = 0

    def before_forward(self, model, args):
        if self.generating and self.request_forwards >= self.config['per_generation_forward_max']:
            raise TimeoutError('Generation request forward cap')
        super().before_forward(model, args)
        if self.generating:
            self.request_forwards += 1

    def begin_generation(self):
        self.check_time()
        if self.requests >= self.config['complete_generation_requests_max']:
            raise TimeoutError('Fixed generation count exhausted')
        self.requests += 1
        self.request_forwards = 0
        self.generating = True


def run_bound_comparison(runtime, layer, record, pair, factor, config, manifest, root):
    """Caller loads admitted runtime/pair under the same pre-existing process guard."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Signal-controlled worker must use main thread')
    validate_manifest(manifest, config, record)
    from .dual_guard import verify_guard
    verify_guard(manifest['guard'])
    if any(abs(manifest['guard'][name + '_epoch'] - (manifest['start_epoch'] + config[key])) > 1e-6 for name, key in [('TERM', 'TERM_seconds'), ('KILL', 'KILL_seconds')]):
        raise PermissionError('Guard deadlines differ from original clock')
    if (type(runtime.model).__name__ != manifest['KV']['actual_class']
            or runtime.get_module(config['module']) is not layer
            or list(layer.weight.shape) != [4096, 14336]
            or runtime.generation_config != manifest['generation']):
        raise PermissionError('Actual runtime differs from bound interface')
    budget = ExecutionBudget(config, manifest['start_epoch'])
    budget.check_time()
    journal = Journal(root, config['disk_bytes_max'])
    journal.private('LAUNCH_RESERVED.json', {'epoch': time.time(), 'binding_id': config['source_binding']})
    journal.private('EXECUTION_BINDING.json', {'manifest': manifest, 'config': config})
    base = layer.weight.detach().clone()
    modes = [(m, m.training) for m in runtime.model.modules()]
    generation_config = copy.deepcopy(runtime.generation_config)
    hook = runtime.model.register_forward_pre_hook(budget.before_forward)
    result = None
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise TimeoutError('Independent guard TERM')
    signal.signal(signal.SIGTERM, terminate)
    try:
        runtime.model.eval()
        admission = dict(source_bound=True, input_bound=True, F0_bound=True, KV_smoke_pass=True, resource_guard_ready=True)
        result = solve_pair_comparison(runtime, layer, pair, factor, config, admission, journal.event, budget=budget)
        public_arms = {}
        u, v = result['basis_private']
        for arm in ARMS:
            value = result['arms'][arm]
            public_arms[arm] = {k: value[k] for k in ('finals', 'native_distinct', 'native_distance', 'numeric_pair_pass')}
            public_arms[arm]['generations'] = []
            for endpoint, z in enumerate(value['coefficients_private']):
                alias = f'{arm}_{endpoint}'
                journal.private(alias + '_RESERVED.json', {'epoch': time.time()})
                budget.begin_generation()
                try:
                    with endpoint_native(runtime.model, layer, base, u, v, z, config), torch.inference_mode():
                        raw = runtime.adapter.prepare_inputs(record.image_path, record.question, None)
                        if raw['input_ids'].tolist() != manifest['input']['prompt_ids'] or raw['attention_mask'].tolist() != manifest['input']['attention_mask'] or raw['image_sha256'] != manifest['input']['existing_image_binding']:
                            raise PermissionError('Actual generation input differs from private binding')
                        generated = runtime.adapter.generate_prepared_with_result(raw, generation_config)
                    ids = list(generated.raw_token_ids)
                    eos = runtime.model.generation_config.eos_token_id
                    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
                    complete = bool(ids) and ids[-1] in eos and len(ids) <= config['max_new_tokens']
                    packet = {'alias': alias, 'status': 'COMPLETE' if complete else 'INCOMPLETE_NO_RETRY', 'question': record.question, 'gold_answer': record.target, 'raw_base_answer': generated.decoded_text, 'raw_token_ids': ids}
                    journal.private(alias + '_GENERATION.json', packet)
                    event = {'arm': arm, 'endpoint': endpoint, 'status': packet['status'], 'generation_forwards': budget.request_forwards, 'generated_token_count': len(ids), 'termination': 'EOS' if complete else 'UNRESOLVED'}
                    journal.event(event)
                    public_arms[arm]['generations'].append(event)
                    if not complete:
                        raise RuntimeError('Incomplete generation; no replacement endpoint')
                finally:
                    budget.generating = False
        journal.private('NUMERIC_TERMINAL.json', {'status': 'COMPLETE', 'arms': public_arms, 'semantic_status': 'NOT_SCORED'})
        journal.event({'status': 'GENERATIONS_COMPLETE_SCORE_NOT_RUN'})
        return public_arms
    except BaseException as error:
        journal.event({'status': 'FAILED_NO_RETRY', 'exception_type': type(error).__name__, 'forwards': budget.forwards})
        raise
    finally:
        budget.generating = False
        budget.restoring = True
        try:
            with torch.no_grad():
                layer.weight.copy_(base)
            if not torch.equal(layer.weight, base):
                raise RuntimeError('Outer native restoration failed')
            if result is not None:
                with torch.no_grad():
                    restored = [float(nll(runtime.model, batch)) for batch in pair]
                if any(abs(a - b) > 1e-5 for a, b in zip(restored, result['baseline'])):
                    raise RuntimeError('Outer W0 loss restoration failed')
            journal.event({'status': 'RESTORED' if result is not None else 'WEIGHTS_RESTORED_NO_BASELINE', 'forwards': budget.forwards, 'backward_API': budget.backwards, 'model_gradient_paths': budget.gradient_paths})
        finally:
            hook.remove()
            signal.signal(signal.SIGTERM, previous_term_handler)
            runtime.generation_config.clear()
            runtime.generation_config.update(generation_config)
            for m, mode in modes:
                m.training = mode
