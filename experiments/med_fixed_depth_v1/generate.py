"""Generate one frozen single-owner event without reading a reference answer."""
import copy
from pathlib import Path
import numpy as np
import torch


def generate_one(runtime, r, task, query, route, prior_input, prior_route, state_file, arm, generation_rng, budget):
    required = {'query_id', 'question', 'image_path', 'image_sha256'}
    if set(query) != required:
        raise ValueError('Generation accepts only approved input fields')
    if arm not in ('C21', 'CDEEP') or set(route) != {'activated', 'logical_edit_id'}:
        raise ValueError('Invalid arm or frozen single-owner route')
    selected = route['logical_edit_id']
    if bool(route['activated']) != bool(selected) or selected not in (None, task['edit_id']):
        raise ValueError('Route is inconsistent with the single owner')
    if route != prior_route:
        raise ValueError('Single-owner route differs from the original')
    if set(prior_input) != {'question', 'image_sha256', 'prompt_ids', 'attention_mask', 'generation'}:
        raise ValueError('Only approved original input binding fields are accepted')
    state = torch.load(state_file, map_location='cpu', weights_only=True)
    expected = 'model.layers.21.mlp.down_proj' if arm == 'C21' else f'model.layers.{len(runtime.model.model.layers)-2}.mlp.down_proj'
    if state['module'] != expected or state['steps'] != 160:
        raise ValueError('State is not bound to the arm and completed step')
    budget()
    runtime.model.half()
    try:
        raw = runtime.adapter.prepare_inputs(Path(r.c.local_path(query['image_path'])), query['question'], None)
        if raw['image_sha256'] != query['image_sha256']:
            raise ValueError('Image binding mismatch')
        binding = dict(question=query['question'], image_sha256=raw['image_sha256'],
                       prompt_ids=raw['input_ids'][0].tolist(), attention_mask=raw['attention_mask'][0].tolist(),
                       generation=runtime.generation_config)
        if binding != prior_input:
            raise ValueError('Actual input binding differs from original MED')
        model = runtime.llava_model()
        with torch.no_grad():
            expanded = model.prepare_inputs_labels_for_multimodal(raw['input_ids'], None, raw['attention_mask'], None, None,
                                                                   raw['images'], image_sizes=None)
        if expanded[4].dtype != torch.float16:
            raise ValueError('Unexpected prefill precision')
        expanded = tuple(x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in expanded)
    finally:
        r.old.fp32(runtime)
    hook = None
    try:
        if selected:
            expert = r.p.expert(copy.deepcopy(state['state']), task['seed'], runtime.device).requires_grad_(False)
            hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(expected), expert)
            hook.attach()
        budget()
        r.c.restore_rng(generation_rng['rng'])
        np.random.set_state(generation_rng['numpy_rng'])
        answer, trace = r.s.fit.short.generate(runtime, hook, raw, expanded)
        observed = r.c.rng()
        if observed['python_rng'] != generation_rng['rng']['python_rng']:
            raise ValueError('Generation changed the locked Python RNG')
        if any(not torch.equal(observed[k], generation_rng['rng'][k]) for k in ('torch_rng', 'cuda_rng')):
            raise ValueError('Generation changed the locked Torch RNG')
        tokens = list(answer.raw_token_ids)
        if not tokens:
            raise ValueError('Empty generation')
        return dict(query_id=query['query_id'], arm=arm, route=route, module=expected, state_file=Path(state_file).name,
                    question=query['question'], image_sha256=raw['image_sha256'], prompt_ids=raw['input_ids'][0].tolist(),
                    attention_mask=raw['attention_mask'][0].tolist(), generation=runtime.generation_config,
                    raw_answer=answer.decoded_text, raw_token_ids=tokens, trace=trace,
                    eos=tokens[-1] == runtime.adapter.tokenizer.eos_token_id, at_cap=len(tokens) >= 1024)
    finally:
        if hook:
            hook.detach()
