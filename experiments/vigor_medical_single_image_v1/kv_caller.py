"""Read-only KV observations around official generation; not a launch admission gate."""
import copy
import inspect
import math

import torch


def cache_length(past):
    if past is None:
        return 0
    # Legacy tuples require a separately certified custom implementation.
    if not callable(getattr(past, 'get_seq_length', None)):
        raise RuntimeError('Unresolved cache implementation')
    length = int(past.get_seq_length())
    if length < 0:
        raise RuntimeError('Invalid cache length')
    return length


def validate_trace(rows, cached):
    if len(rows) < 2:
        raise RuntimeError('CACHE_NOT_EXERCISED')
    if rows[0]['past'] != 0 or rows[0]['length'] < 1:
        raise RuntimeError('Missing initial prefill')
    for previous, row in zip(rows, rows[1:]):
        if cached:
            if row['past'] <= previous['past'] or row['length'] != 1:
                raise RuntimeError('Missing cached single-token decode')
        elif row['past'] != 0 or row['length'] <= previous['length']:
            raise RuntimeError('Missing uncached growing sequence')


class Observer:
    def __init__(self, model, cached, reference=None):
        self.signature = inspect.signature(model.forward)
        self.cached = cached
        self.reference = reference
        self.rows, self.logits = [], []
        self.max_abs = self.error_square = self.reference_square = 0.0
        self.elements = 0

    def before(self, model, args, kwargs):
        bound = self.signature.bind_partial(*args, **kwargs).arguments
        for name, parameter in self.signature.parameters.items():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                bound.update(bound.pop(name, {}))
        # Read cache length BEFORE forward: DynamicCache mutates in-place.
        length = cache_length(bound.get('past_key_values'))
        value = bound.get('inputs_embeds')
        if value is None:
            value = bound.get('input_ids')
        if value is None or value.shape[0] != 1:
            raise RuntimeError('Unresolved batch/input shape')
        if bound.get('use_cache') is not self.cached:
            raise RuntimeError('Actual cache flag differs from request')
        self.rows.append({'past': length, 'length': int(value.shape[1])})
        if len(self.rows) > 128:
            raise TimeoutError('KV observer request cap')

    def after(self, model, args, kwargs, output):
        logits = output.logits[:, -1, :].detach().float().cpu().clone()
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite generation logits')
        if self.reference is None:
            self.logits.append(logits)
        else:
            index = len(self.rows) - 1
            if index >= len(self.reference) or logits.shape != self.reference[index].shape:
                raise RuntimeError('Unaligned generation logits')
            ref = self.reference[index]
            error = logits.double() - ref.double()
            self.max_abs = max(self.max_abs, float(error.abs().max()))
            self.error_square += float(error.square().sum())
            self.reference_square += float(ref.double().square().sum())
            self.elements += logits.numel()

    def metrics(self):
        if not self.elements:
            raise RuntimeError('No aligned logits')
        relative = math.sqrt(self.error_square / self.elements) / max(math.sqrt(self.reference_square / self.elements), 1e-12)
        if self.max_abs > .05 or relative > .001:
            raise RuntimeError('Frozen logits tolerance failed')
        return {'logits_max_abs': self.max_abs, 'logits_relative_RMS': relative}


def paired_kv_generation(runtime, raw, budget, reserve):
    """Called within admitted native-weight/restoration scope and shared budget hook.

    reserve(cache_flag) must durably reserve each attempt before generation. This
    function does not load a model, authorize a source, or emit a full smoke PASS.
    """
    config = copy.deepcopy(runtime.generation_config)
    if config.get('do_sample') is not False or config.get('num_beams') != 1 or config.get('max_new_tokens') != 1024:
        raise ValueError('Frozen generation configuration required')
    if type(runtime.model).__name__ != 'LlavaMistralForCausalLM':
        raise ValueError('Unexpected actual model class')
    eos = runtime.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    observers, sequences = [], []
    for cached in (True, False):
        reserve(cached)
        budget.begin_generation()
        observer = Observer(runtime.model, cached, observers[0].logits if observers else None)
        pre = post = None
        try:
            pre = runtime.model.register_forward_pre_hook(observer.before, with_kwargs=True)
            post = runtime.model.register_forward_hook(observer.after, with_kwargs=True)
            with torch.inference_mode():
                generated = runtime.adapter.generate_prepared_with_result(raw, dict(config, use_cache=cached))
            ids = tuple(generated.raw_token_ids)
            if not ids or ids[-1] not in eos or len(ids) > 1024:
                raise RuntimeError('Incomplete generation; no retry')
            if len(ids) != len(observer.rows):
                raise RuntimeError('Unresolved continuation/forward alignment')
            validate_trace(observer.rows, cached)
            observers.append(observer)
            sequences.append(ids)
        finally:
            if post is not None:
                post.remove()
            if pre is not None:
                pre.remove()
            budget.generating = False
    if sequences[0] != sequences[1] or len(observers[0].rows) != len(observers[1].rows):
        raise RuntimeError('Cached/uncached full continuation mismatch')
    metrics = observers[1].metrics()
    return dict(status='KV_GENERATION_COMPONENT_PASS', requests=2,
                token_count=len(sequences[0]), **metrics)
