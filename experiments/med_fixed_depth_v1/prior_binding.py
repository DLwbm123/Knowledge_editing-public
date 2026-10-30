"""Select original input, routing and execution metadata without decoding answers."""
import json
from pathlib import Path
import stream_extract as s


def skip(f, token):
    if token == b'"':
        s.string(f)
    elif token in (b'{', b'['):
        s.skip_compound(f, token)
    else:
        s.scalar(f, token)


def decode(f, token):
    start = f.tell() - 1
    skip(f, token)
    end = f.tell()
    if end - start > 2_000_000:
        raise ValueError('Binding field exceeds size limit')
    f.seek(start)
    value = json.loads(f.read(end - start))
    f.seek(end)
    return value


def select(f, token, allowed):
    if token != b'{':
        raise ValueError('Expected binding object')
    result = {}
    while True:
        token = s.nonspace(f)
        if token == b'}':
            return result
        if token != b'"':
            raise ValueError('Expected binding field')
        key = s.string(f, True)
        if s.nonspace(f) != b':':
            raise ValueError('Expected binding colon')
        token = s.nonspace(f)
        if key in allowed:
            if key in result:
                raise ValueError('Duplicate binding field')
            rule = allowed[key]
            result[key] = select(f, token, rule) if isinstance(rule, dict) else decode(f, token)
        else:
            skip(f, token)
        token = s.nonspace(f)
        if token == b'}':
            return result
        if token != b',':
            raise ValueError('Expected binding delimiter')


def read(path, expected_consumer):
    allowed = {
        'consumer_id': True,
        'route': True,
        'effective_expert': True,
        'binding': {'judge_input': {'question': True, 'image_sha256': True, 'prompt_ids': True,
                                    'attention_mask': True, 'runtime': True, 'generation': True}},
        'prediction_binding': {'model_source': True, 'execution_source': True,
                               'deterministic_greedy': True, 'generation_rng_binding': True},
    }
    with Path(path).open('rb') as f:
        result = select(f, s.nonspace(f), allowed)
    if result.get('consumer_id') != expected_consumer:
        raise ValueError('Original consumer identity mismatch')
    b = result.get('binding', {}).get('judge_input', {})
    if set(b) != {'question', 'image_sha256', 'prompt_ids', 'attention_mask', 'runtime', 'generation'}:
        raise ValueError('Original input binding incomplete')
    pb = result.get('prediction_binding', {})
    if set(pb) != {'model_source', 'execution_source', 'deterministic_greedy', 'generation_rng_binding'}:
        raise ValueError('Original execution binding incomplete')
    route = result.get('route')
    if not isinstance(route, dict) or type(route.get('activated')) is not bool:
        raise ValueError('Original route invalid')
    if (route.get('logical_edit_id') if route['activated'] else None) != result.get('effective_expert'):
        raise ValueError('Original effective expert disagrees with route')
    return result
