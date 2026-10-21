"""Fixed IND2/JOINT2 production core; launch and semantic qualification are separate."""
from contextlib import contextmanager
import math
import time

import torch

from .coefficient_runtime import coefficient_forward, compose, native_weight, validate
from .editor import editable_fp32, finite, nll


class Budget:
    def __init__(self, config):
        self.config = config
        self.started = time.monotonic()
        self.forwards = self.backwards = self.gradient_paths = 0
        self.restoring = False

    def before_forward(self, model, args):
        limit = self.config['model_forward_max'] if self.restoring else self.config['work_forward_max']
        self.check_time()
        if self.forwards >= limit:
            raise TimeoutError('Forward budget exhausted')
        self.forwards += 1

    def check_time(self):
        limit = self.config['KILL_seconds'] if self.restoring else self.config['work_seconds']
        if time.monotonic() - self.started >= limit:
            raise TimeoutError('Time budget exhausted')

    def grad(self, loss, parameters, paths):
        self.check_time()
        if self.backwards >= self.config['backward_API_max'] or self.gradient_paths + paths > self.config['model_gradient_paths_max']:
            raise TimeoutError('Gradient budget exhausted')
        self.backwards += 1
        self.gradient_paths += paths
        return torch.autograd.grad(loss, parameters)


def orthogonalize(columns, tolerance):
    vectors = []
    reference = float(columns[:, 0].norm())
    for column in columns.T:
        q = column.clone()
        for _ in range(2):
            for previous in vectors:
                q = q - previous * (previous @ q)
        norm = float(q.norm())
        if not math.isfinite(norm) or norm <= tolerance * reference or norm == 0:
            raise ValueError('Fixed basis rank unavailable')
        vectors.append(q / norm)
    return torch.stack(vectors, dim=1)


def source_basis(gradient, a, b, config):
    vectors = []
    norms = gradient.square().sum(0)
    reference = float(norms.max().sqrt())
    for _ in range(config['rank']):
        residual_norms = norms.clone()
        for q in vectors:
            residual_norms -= (q @ gradient).square()
        # argmax chooses the first index on a tie; no random directions.
        q = gradient[:, int(residual_norms.argmax())].clone()
        for _ in range(2):
            for previous in vectors:
                q -= previous * (previous @ q)
        norm = float(q.norm())
        if not math.isfinite(norm) or norm == 0 or norm <= config['basis_relative_rank_tol'] * reference:
            raise ValueError('Fixed source rank unavailable')
        vectors.append(q / norm)
    u = torch.stack(vectors, dim=1)
    v = orthogonalize(gradient.T @ u, config['basis_relative_rank_tol'])
    def whiten(q, metric):
        gram = q.double().T @ (metric.double()[:, None] * q.double())
        lower = torch.linalg.cholesky(gram)
        inverse_transpose = torch.linalg.solve_triangular(lower.T, torch.eye(q.shape[1], device=q.device, dtype=torch.float64), upper=True)
        result = (q.double() @ inverse_transpose).float()
        error = (result.double().T @ (metric.double()[:, None] * result.double()) - torch.eye(q.shape[1], device=q.device)).abs().max()
        if float(error) > config['metric_orthogonality_atol']:
            raise ValueError('F0 basis whitening failed')
        return finite(result)
    return whiten(u, b), whiten(v, a)


def endpoint_objective(visual, null, z, baseline, config):
    null_violation = torch.relu((null - baseline[1]).abs() / config['null_abs_budget'] - 1)
    trust_violation = torch.relu(z.square().sum() - 1)
    return visual / max(baseline[0], config['source_normalizer_floor']) + config['constraint_penalty'] * (null_violation.square() + trust_violation.square())


def pair_objective(losses, zs, baseline, arm, config):
    total = sum(endpoint_objective(v, n, z, baseline, config) for (v, n), z in zip(losses, zs))
    if arm == 'JOINT2':
        distance = torch.linalg.vector_norm(zs[0] - zs[1])
        total = total + config['joint_diversity_penalty'] * torch.relu((config['diversity_min'] - distance) / config['diversity_min']).square()
    elif arm != 'IND2':
        raise ValueError('Unknown frozen arm')
    return finite(total)


def separated_gradients(losses, zs, baseline, arm, config, budget):
    # Apply large penalty adjoints in FP32, never across the FP16 model graph.
    detached = [[x.detach().requires_grad_(True) for x in row] for row in losses]
    objective = pair_objective(detached, zs, baseline, arm, config)
    flat = [x for row in detached for x in row]
    derivatives = torch.autograd.grad(objective, flat + zs, allow_unused=True)
    gradients = []
    for i, (row, z) in enumerate(zip(losses, zs)):
        direct = derivatives[len(flat) + i]
        total = torch.zeros_like(z) if direct is None else finite(direct).detach()
        for j, loss in enumerate(row):
            budget.check_time()
            if budget.backwards >= budget.config['backward_API_max'] or budget.gradient_paths >= budget.config['model_gradient_paths_max']:
                raise TimeoutError('Separated gradient budget exhausted')
            budget.backwards += 1
            budget.gradient_paths += 1
            gradient = torch.autograd.grad(loss, z, retain_graph=(j == 0))[0]
            total = total + finite(gradient).float() * finite(derivatives[2*i+j]).detach().float()
        gradients.append(finite(total))
    return objective.detach(), gradients


def fisher_cost(delta, a, b):
    return float((delta.double().square() * b.double()[:, None] * a.double()[None, :]).sum()) / 2


@contextmanager
def endpoint_native(model, layer, base, u, v, z, config):
    candidate = compose(base, u, z * math.sqrt(2 * config['trust_budget']), v)
    with native_weight(model, layer, candidate):
        yield


def solve_pair_comparison(runtime, layer, pair, factor, config, admission, progress, *, budget=None):
    """Return private U/V/Z plus anonymous scalar traces; never run or score generations.

    Caller owns exclusive launch reservation, source/F0 binding, independent process
    guard and storage/GPU admission. This function does not load models or data.
    """
    required = ('source_bound', 'input_bound', 'F0_bound', 'KV_smoke_pass', 'resource_guard_ready')
    if not all(admission.get(k) is True for k in required):
        raise PermissionError('Source/interface/resource admission incomplete')
    if config['version'] != 'PUREW_IND2_JOINT2_CORE_V1' or config['source_binding'] is None:
        raise PermissionError('Need concrete independently bound execution config')
    if not callable(progress):
        raise ValueError('Durable anonymous progress callback required')
    validate(runtime.model, layer)
    if layer.weight.dtype != torch.float16:
        raise ValueError('Native FP16 required')
    base = layer.weight.detach().clone()
    a, b = (factor[k].detach().to(device=base.device, dtype=torch.float32) for k in ('a', 'b'))
    if a.shape != (base.shape[1],) or b.shape != (base.shape[0],) or not bool((a > 0).all() and (b > 0).all()):
        raise ValueError('Invalid bound F0 factors')
    finite(a); finite(b)
    owns_budget = budget is None
    budget = Budget(config) if owns_budget else budget
    previous_restoring = budget.restoring
    modes = [(module, module.training) for module in runtime.model.modules()]
    parameter_flags = [(parameter, parameter.requires_grad) for parameter in runtime.model.parameters()]
    hook = runtime.model.register_forward_pre_hook(budget.before_forward) if owns_budget else None
    result = {'arms': {}, 'semantic_correctness': 'NOT_EVALUATED'}
    try:
        runtime.model.eval()
        with torch.no_grad():
            baseline = [float(nll(runtime.model, batch)) for batch in pair]
        with editable_fp32(runtime.model, layer, readonly=True):
            gradient = finite(budget.grad(nll(runtime.model, pair[0]), (layer.weight,), 1)[0].detach())
        u, v = source_basis(gradient, a, b, config)
        del gradient
        for arm in config['arms']:
            zs, optimizers = [], []
            for sign in config['init_Z_E11']:
                z = torch.zeros((config['rank'], config['rank']), device=base.device, dtype=torch.float32)
                z[0, 0] = sign
                z.requires_grad_(True)
                zs.append(z)
                optimizers.append(torch.optim.Adam([z], lr=config['lr'], betas=tuple(config['betas']), eps=config['eps'], weight_decay=config['weight_decay']))
            trace = []
            for step in range(config['steps']):
                losses = []
                for z in zs:
                    with coefficient_forward(runtime.model, layer, u, z * math.sqrt(2 * config['trust_budget']), v):
                        losses.append([nll(runtime.model, batch) for batch in pair])
                objective, gradients = separated_gradients(losses, zs, baseline, arm, config, budget)
                trace.append({'step': step, 'pre_update_losses': [[float(x.detach()) for x in values] for values in losses], 'objective': float(objective.detach())})
                for optimizer, z, gradient in zip(optimizers, zs, gradients):
                    optimizer.zero_grad(set_to_none=True)
                    z.grad = finite(gradient).detach()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    finite(z)
                progress({'arm': arm, 'step': step + 1, 'last_step': trace[-1], 'forwards': budget.forwards, 'backward_API': budget.backwards, 'model_gradient_paths': budget.gradient_paths})
                del objective, losses, gradients
            finals = []
            for z in zs:
                with endpoint_native(runtime.model, layer, base, u, v, z, config), torch.no_grad():
                    values = [float(nll(runtime.model, batch)) for batch in pair]
                    delta = layer.weight.detach().float() - base.float()
                    cost = fisher_cost(delta, a, b)
                    finals.append({'native_nll': values, 'fisher_cost': cost, 'changed': bool(torch.count_nonzero(delta)), 'numeric_feasible': values[0] <= config['source_NLL_target'] and abs(values[1] - baseline[1]) <= config['null_abs_budget'] and cost <= config['trust_budget']})
            first = compose(base, u, zs[0].detach() * math.sqrt(2 * config['trust_budget']), v).half()
            second = compose(base, u, zs[1].detach() * math.sqrt(2 * config['trust_budget']), v).half()
            distance = math.sqrt(fisher_cost(first.float() - second.float(), a, b) / config['trust_budget'])
            result['arms'][arm] = {'trace': trace, 'finals': finals, 'native_distinct': not torch.equal(first, second), 'native_distance': distance, 'numeric_pair_pass': all(x['numeric_feasible'] for x in finals) and not torch.equal(first, second) and distance >= config['diversity_min'], 'coefficients_private': [z.detach().clone() for z in zs]}
            del first, second, delta
        result.update(baseline=baseline, basis_private=(u, v))
    except BaseException as error:
        progress({'status': 'FAILED', 'exception_type': type(error).__name__, 'forwards': budget.forwards, 'backward_API': budget.backwards, 'model_gradient_paths': budget.gradient_paths})
        raise
    finally:
        budget.restoring = True
        with torch.no_grad():
            layer.weight.copy_(base)
        try:
            if layer.weight.dtype != base.dtype or not torch.equal(layer.weight, base):
                raise RuntimeError('Native restoration failed')
            if 'baseline' in locals():
                with torch.no_grad():
                    restored = [float(nll(runtime.model, batch)) for batch in pair]
                if any(abs(x - y) > 1e-5 for x, y in zip(restored, baseline)):
                    raise RuntimeError('Baseline NLL restoration failed')
        finally:
            if hook is not None:
                hook.remove()
            budget.restoring = previous_restoring
            for parameter, flag in parameter_flags:
                parameter.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode
    result['costs'] = {'forward': budget.forwards, 'backward_API': budget.backwards, 'model_gradient_paths': budget.gradient_paths, 'seconds': time.monotonic() - budget.started}
    return result
