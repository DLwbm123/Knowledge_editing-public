"""Image-conditioned MEMIT adapter; native formula and optimizer.

Explicit MM adaptation: original image, native+4 frozen FIT contexts, expanded
last prompt token, answer+EOS CE; an image-only generic prompt replaces the
text-entity KL template. No evaluation examples enter editing or statistics.
"""
import torch
import torch.nn.functional as F
from algebra import memit_key


def prefix(batch):
    kwargs = dict(batch.forward_kwargs())
    labels = kwargs.pop('labels')
    end = int(torch.where(labels[0] != -100)[0][0])
    assert end > 0
    for name in ('inputs_embeds', 'attention_mask', 'position_ids'):
        if kwargs.get(name) is not None:
            kwargs[name] = kwargs[name][:, :end]
    return kwargs, end - 1


def capture(runtime, batches, layers, final):
    """Official two context groups: native .5, mean(FIT4) .5."""
    keys = {layer: [] for layer in layers}
    native_z = None
    for index, batch in enumerate(batches):
        kwargs, position = prefix(batch)
        found = {}
        handles = []
        for layer in layers:
            def key_hook(_module, args, layer=layer):
                found[layer] = args[0][0, position].detach().float().clone()
            handles.append(runtime.get_module(f'model.layers.{layer}.mlp.down_proj').register_forward_pre_hook(key_hook))
        def z_hook(_module, args, output):
            found['z'] = output[0][0, position].detach().float().clone()
        handles.append(runtime.get_module(f'model.layers.{final}').register_forward_hook(z_hook))
        try:
            with torch.no_grad():
                runtime.model(**kwargs)
        finally:
            for handle in handles:
                handle.remove()
        assert set(found) == set(layers) | {'z'}
        for layer in layers:
            keys[layer].append(found[layer])
        if index == 0:
            native_z = found['z']
    assert len(batches) == 5
    keys = {layer: torch.stack((values[0],torch.stack(values[1:]).mean(0))).mean(0).cpu().double()
            for layer, values in keys.items()}
    return keys, native_z


def optimize_z(runtime, batches, kl_batch, final, hp, counts, budget):
    _, initial = capture(runtime, batches, [final], final)
    kl_kwargs, kl_position = kl_batch
    with torch.no_grad():
        base_kl = runtime.model(**kl_kwargs).logits[0, -1].float().log_softmax(-1)
    delta = torch.zeros_like(initial, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=hp['v_lr'])
    position = 0
    def inject(_module, args, output):
        hidden = output[0].clone()
        hidden[0, position] = hidden[0, position] + delta
        return (hidden, *output[1:])
    handle = runtime.get_module(f'model.layers.{final}').register_forward_hook(inject)
    trace = []
    try:
        for step in range(hp['v_num_grad_steps']):
            budget()
            optimizer.zero_grad(set_to_none=True)
            # Accumulate each context separately to bound activation memory.
            nll_value = 0.
            last = step == hp['v_num_grad_steps'] - 1
            for batch in batches:
                position = prefix(batch)[1]
                kwargs = dict(batch.forward_kwargs())
                kwargs.pop('labels')
                output = runtime.model(**kwargs)
                logits, labels = output.logits, batch.labels
                nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1), ignore_index=-100) / len(batches)
                assert torch.isfinite(nll)
                nll_value += float(nll.detach())
                if not last:
                    nll.backward()
                    counts['backwards'] += 1
                del output, logits, nll
            position = kl_position
            current = runtime.model(**kl_kwargs).logits[0, -1].float().log_softmax(-1)
            kl = hp['kl_factor'] * F.kl_div(base_kl[None], current[None], log_target=True, reduction='batchmean')
            penalty = hp['v_weight_decay'] * delta.norm() / initial.norm().square()
            regularizer = kl + penalty
            total = nll_value + float(regularizer.detach())
            assert torch.isfinite(regularizer) and torch.isfinite(torch.tensor(total))
            trace.append(dict(step=step, objective=total, NLL=nll_value,
                KL=float(kl.detach()), norm_penalty=float(penalty.detach())))
            # Native stopping occurs before optimizer.step. Earlier separate
            # backwards have no effect when the stopping condition is reached.
            if total < .05 or last:
                break
            regularizer.backward()
            counts['backwards'] += 1
            assert delta.grad is not None and torch.isfinite(delta.grad).all()
            trace[-1]['gradient_norm'] = float(delta.grad.norm())
            optimizer.step()
            counts['updates'] += 1
            with torch.no_grad():
                maximum = hp['clamp_norm_factor'] * initial.norm()
                if delta.norm() > maximum:
                    delta.mul_(maximum / delta.norm())
    finally:
        handle.remove()
    return initial + delta.detach(), dict(trace=trace, initial_norm=float(initial.norm()),
        delta_norm=float(delta.detach().norm()), context_objective='Mean CE over native+4FIT',
        key_context_groups='native .5 + FIT4 mean .5')


def edit(runtime, batches, kl_batch, layers, hp, statistics, counts, budget):
    target, audit = optimize_z(runtime, batches, kl_batch, layers[-1], hp, counts, budget)
    writes = []
    for index, layer in enumerate(layers):
        budget()
        keys, current = capture(runtime, batches, [layer], layers[-1])
        key = keys[layer]
        adjusted = memit_key(statistics[layer], key)
        residual = (target - current).double() / (len(layers) - index)
        weight = runtime.get_module(f'model.layers.{layer}.mlp.down_proj').weight
        assert weight.dtype == torch.float32 and not weight.requires_grad
        update = torch.outer(residual, adjusted.to(runtime.device)).float()
        assert update.shape == weight.shape and torch.isfinite(update).all()
        with torch.no_grad():
            weight.add_(update)
        writes.append(dict(layer=layer, key_norm=float(key.norm()),
            adjusted_key_norm=float(adjusted.norm()), residual_norm=float(residual.norm()),
            update_norm=float(update.norm())))
        del update
    counts['edits'] += 1
    return dict(latent=audit, writes=writes, method='MEMIT_MM')
