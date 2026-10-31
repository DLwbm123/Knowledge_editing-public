"""Use fixed-depth generation with the frozen source-teacher candidate."""
from pathlib import Path
import torch
import generate as fixed_generate
from train_pair import state_digest


def one(runtime, r, task, query, original, state_path, training_result, rng, budget):
    if original['owner'] != task['order'] or original['query_id'] != query['query_id']:
        raise ValueError('Original event identity differs')
    state = torch.load(Path(state_path), map_location='cpu', weights_only=True)
    if training_result.get('steps') != 160 or state_digest(state['state']) != training_result['final_digest']:
        raise ValueError('Candidate does not match completed training')
    original_route = original['route']
    route = dict(activated=original_route['activated'],
                 logical_edit_id=original_route['logical_edit_id'] if original_route['activated'] else None)
    binding = {k: original['input'][k] for k in ('question', 'image_sha256', 'prompt_ids', 'attention_mask', 'generation')}
    result = fixed_generate.generate_one(runtime, r, task, query, route, binding, route,
                                         state_path, 'CDEEP', rng, budget)
    result.update(arm='SOURCE_TEACHER', owner=task['order'], original_consumer_id=original['consumer_id'],
                  original_execution_binding=original['prediction_binding'], original_route=original_route,
                  prediction_reused_from=None)
    return result
