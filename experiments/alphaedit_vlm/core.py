"""AlphaEdit single-request algebra and native multimodal latent optimization.

Adapted from jianghoucheng/AlphaEdit, commit b84624f44dfe8fc6cd9e41df916c44124a0c46dc.
See UPSTREAM_LICENSE. This campaign resets after each request: history is empty.
"""
import torch
import torch.nn.functional as F


def prefix_inputs(prepared):
    inputs, labels = prepared[:2]
    positions = (labels[0] != -100).nonzero().flatten()
    if not len(positions) or int(positions[0]) < 1:
        raise ValueError('missing expanded answer boundary')
    end = int(positions[0])
    result = dict(inputs)
    result['inputs_embeds'] = inputs['inputs_embeds'][:, :end]
    for name in ('attention_mask', 'position_ids'):
        if result.get(name) is not None:
            result[name] = result[name][:, :end]
    return result, end - 1


def answer_nll(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def capture(model, inputs, layer, position):
    """Input of down_proj is the key; block output is the latent target space."""
    found = {}
    block = model.get_submodule(f'model.layers.{layer}')
    def key_hook(_, args):
        found['key'] = args[0][0, position].detach().float().clone()
    def output_hook(_, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        found['hidden'] = hidden[0, position].detach().float().clone()
    hooks = [block.mlp.down_proj.register_forward_pre_hook(key_hook),
             block.register_forward_hook(output_hook)]
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return found['key'], found['hidden']


def single_update(key, residual, basis, l2):
    """Exact rank-one form of official solve(P k k^T + L2 I, P k r^T)^T.

    P = U U^T; U has orthonormal columns. There are no prior edits after a reset.
    This avoids constructing/solving a 14336-square system for a single request.
    """
    if l2 <= 0 or key.ndim != 1 or residual.ndim != 1 or basis.shape[0] != key.numel():
        raise ValueError('invalid single-request system')
    projected = basis @ (basis.T @ key)
    denominator = l2 + torch.dot(key, projected)
    update = torch.outer(residual, projected / denominator)
    if not torch.isfinite(update).all():
        raise FloatingPointError('nonfinite closed-form update')
    return update


def optimize_z(model, target, reference, layer, hp):
    """Optimize one FP32 latent delta; all model parameters stay frozen.

    Adaptations: native image+question input, last prompt position, one native
    context, and answer+EOS NLL. The reference QA replaces text '{} is a'.
    KL direction, Adam, norm penalty and clamp follow the official code.
    """
    target_prefix, position = prefix_inputs(target)
    reference_prefix, reference_position = prefix_inputs(reference)
    _, initial = capture(model, target_prefix, layer, position)
    with torch.no_grad():
        reference_base = model(**reference_prefix).logits[0, -1].float().log_softmax(-1)
    delta = torch.zeros_like(initial, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=hp['v_lr'])
    block = model.get_submodule(f'model.layers.{layer}')
    active_position = position
    def inject(_, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[0, active_position] = (hidden[0, active_position].float() + delta).to(hidden.dtype)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited
    hook = block.register_forward_hook(inject)
    trace = []
    try:
        for step in range(hp['v_num_grad_steps']):
            optimizer.zero_grad(set_to_none=True)
            active_position = position
            nll = answer_nll(model(**target[0]).logits, target[1])
            active_position = reference_position
            current_logp = model(**reference_prefix).logits[0, -1].float().log_softmax(-1)
            kl = hp['kl_factor'] * F.kl_div(reference_base[None], current_logp[None],
                                           log_target=True, reduction='batchmean')
            penalty = hp['v_weight_decay'] * delta.norm() / initial.norm().square()
            loss = nll + kl + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite latent objective')
            trace.append(dict(step=step, loss=float(loss.detach()), nll=float(nll.detach()),
                              kl=float(kl.detach()), norm_penalty=float(penalty.detach())))
            if float(loss.detach()) < .05 or step == hp['v_num_grad_steps'] - 1:
                break
            loss.backward()
            if delta.grad is None or not torch.isfinite(delta.grad).all():
                raise FloatingPointError('invalid latent gradient')
            trace[-1]['gradient_norm'] = float(delta.grad.norm())
            optimizer.step()
            with torch.no_grad():
                maximum = hp['clamp_norm_factor'] * initial.norm()
                if delta.norm() > maximum:
                    delta.mul_(maximum / delta.norm())
    finally:
        hook.remove()
    return (initial + delta.detach()), dict(trace=trace, initial_norm=float(initial.norm()),
                                            delta_norm=float(delta.detach().norm()))


def write_edit(model, prepared, layers, target_z, bases, hp):
    inputs, position = prefix_inputs(prepared)
    records = []
    for index, layer in enumerate(layers):
        key, _ = capture(model, inputs, layer, position)
        _, current_z = capture(model, inputs, layers[-1], position)
        residual = (target_z - current_z) / (len(layers) - index)
        basis = bases[layer].to(key.device)
        update = single_update(key, residual, basis, hp['L2'])
        weight = model.get_submodule(f'model.layers.{layer}.mlp.down_proj').weight
        old = weight.detach().float().clone()
        with torch.no_grad():
            weight.copy_((old + update).to(weight.dtype))
        actual = weight.detach().float() - old
        # This measures write-rounding effects relative to the admissible subspace.
        projected_actual = (actual @ basis) @ basis.T
        records.append(dict(layer=layer, requested_update_norm=float(update.norm()),
                            actual_update_norm=float(actual.norm()), residual_norm=float(residual.norm()),
                            rounding_error_norm=float((actual-update).norm()),
                            actual_outside_nullspace_norm=float((actual-projected_actual).norm())))
        del basis, update, old, actual, projected_actual
    return records
