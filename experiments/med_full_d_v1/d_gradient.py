"""D source gradient for the original MED constrained update."""
import torch
from robust_objective import weights_and_value, view_loss
from span_train import terms


def source_gradients(r, runtime, hook, expert, batches, rows, lock, step):
    if len(batches) != 5 or len(rows) != 5 or len(lock['views']) != 5:
        raise ValueError('Expected one source and four FIT views')
    if lock.get('status') != 'WEAK_TOKEN_MATCH' or step < 1:
        raise ValueError('Invalid fixed D span lock')
    chosen = (0, 1 + (step - 1) % 4)
    expert.zero_grad(set_to_none=True)
    def calculate(i):
        view = lock['views'][i]
        return terms(runtime, hook, batches[i], rows[i], lock['answer_tokens'], view['medical'], view)
    with torch.no_grad():
        first = [calculate(i) for i in chosen]
    weights, value = weights_and_value(first[0][1], first[1][1], 1.0)
    for j, i in enumerate(chosen):
        ce, spans = calculate(i)
        if not torch.allclose(ce.detach(), first[j][0], atol=1e-5, rtol=1e-5):
            raise ValueError('Recomputed CE mismatch')
        if not torch.allclose(spans.detach(), first[j][1], atol=1e-5, rtol=1e-5):
            raise ValueError('Recomputed span mismatch')
        view_loss(ce, spans, weights[j], 1.0).backward()
        r.COUNTS['backwards'] += 1
    return None, [float(x[0]) for x in first], float(value)
