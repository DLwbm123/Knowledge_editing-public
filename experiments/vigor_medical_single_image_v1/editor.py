"""Single-image VIGOR, reconstructed from equations 9–14 and appendix A.5."""
from contextlib import contextmanager
from dataclasses import replace
import math

import torch
import torch.nn.functional as F

VERSION = 'vigor-medical-single-image-v1'
TARGET_NLL = .01
TRUST_BUDGET = 1e-3
MAX_STEPS = 5


def finite(tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError('Nonfinite numerical state')
    return tensor


def same_optional(a, b):
    return (a is None and b is None) or (
        isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))


def paired_batches(runtime, record):
    """Reuse target masking/expansion; intervene only on preprocessed pixels."""
    full = runtime.build_edit_batch(record)
    raw = runtime.adapter.prepare_inputs(record.image_path, record.question, record.target)
    if not torch.equal(raw['input_ids'], full.raw_input_ids):
        raise ValueError('Non-deterministic prompt preprocessing')
    images = raw['images']
    tensors = images if isinstance(images, list) else [images]
    if len(tensors) != 1 or tensors[0].ndim != 4 or tensors[0].shape[0] != 1:
        raise ValueError('This adapter supports exactly one image, without crop grids')
    shape = [list(t.shape) for t in tensors] if isinstance(images, list) else list(images.shape)
    if shape != full.image_tensor_shape:
        raise ValueError('Image shape changed during preprocessing')
    zeros = [torch.zeros_like(t) for t in tensors]
    null_images = zeros if isinstance(images, list) else zeros[0]
    kwargs = dict(raw_input_ids=full.raw_input_ids, attention_mask=raw['attention_mask'],
                  labels=full.raw_labels)
    # The runtime exposes no raw-pixel batch: verify the second preprocessing pass.
    original = runtime._expand_multimodal(**kwargs, images=images)
    null = runtime._expand_multimodal(**kwargs, images=null_images)
    reference = (full.inputs_embeds, full.attention_mask, full.position_ids, full.labels)
    if not all(same_optional(a, b) for a, b in zip(original, reference)):
        raise ValueError('Non-deterministic multimodal preprocessing')
    if null[0].shape != full.inputs_embeds.shape or null[0].dtype != full.inputs_embeds.dtype:
        raise ValueError('Visual-Null changed embedding structure')
    if not all(same_optional(a, b) for a, b in zip(null[1:], reference[1:])):
        raise ValueError('Visual-Null changed masks, positions or target labels')
    return full, replace(full, inputs_embeds=null[0], attention_mask=null[1],
                         position_ids=null[2], labels=null[3])


def nll(model, batch):
    labels = batch.labels[:, 1:]
    if not (labels != -100).any():
        raise ValueError('Empty target-token mask')
    logits = model(**batch.forward_kwargs()).logits[:, :-1].float()
    return finite(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                  ignore_index=-100, reduction='mean'))


@contextmanager
def editable_fp32(model, layer):
    """Temporary casts, frozen backbone, rollback on exception, native exit."""
    if not isinstance(layer, torch.nn.Linear) or layer.bias is not None:
        raise ValueError('Expected a bias-free FFN output Linear')
    if layer.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Unsupported native weight dtype')
    if layer._forward_hooks or layer._forward_pre_hooks:
        raise ValueError('Editable layer already has hooks')
    parameters = list(model.parameters())
    if any(p.grad is not None for p in parameters):
        raise ValueError('Cannot edit a model with pending gradients')
    flags = [p.requires_grad for p in parameters]
    modes = [(m, m.training) for m in model.modules()]
    native = layer.weight.detach().clone()
    handles = []
    input_dtype = []
    try:
        model.eval()
        for p in parameters:
            p.requires_grad_(False)
        layer.weight.data = native.float().clone()
        layer.weight.requires_grad_(True)
        def before(module, args):
            input_dtype.append(args[0].dtype)
            return (args[0].float(), *args[1:])
        def after(module, args, output):
            return output.to(input_dtype.pop())
        handles = [layer.register_forward_pre_hook(before), layer.register_forward_hook(after)]
        yield
        committed = finite(layer.weight.detach().to(native.dtype))
        layer.weight.data = committed
    except BaseException:
        layer.weight.data = native
        raise
    finally:
        for handle in handles:
            handle.remove()
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def damp(moment):
    finite(moment)
    if moment.ndim != 1 or (moment < 0).any():
        raise ValueError('Invalid Fisher moment')
    return (moment + 1e-3 * moment.mean()).clamp_min(1e-12)


def direction(gv, g0, a, b):
    """Diagonal Kronecker inverse in [output, input] matrix order."""
    if gv.shape != g0.shape or gv.shape != (b.numel(), a.numel()):
        raise ValueError('Fisher/gradient shape mismatch')
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError('Nonpositive damped Fisher factor')
    for value in (gv, g0, a, b):
        finite(value)
    iv = gv / b[:, None] / a[None, :]
    i0 = g0 / b[:, None] / a[None, :]
    vv = float((gv.double() * iv.double()).sum())
    zz = float((g0.double() * i0.double()).sum())
    vz = float((gv.double() * i0.double()).sum())
    if not all(math.isfinite(x) for x in (vv, zz, vz)):
        raise ValueError('Nonfinite Fisher geometry')
    if vv <= 0 or zz <= 0:
        return None, dict(status='DEGENERATE_GRADIENT', E=0., kvv=vv, k00=zz)
    tangent = finite(iv - (vz / zz) * i0)
    energy = float((tangent.double().square() * b.double()[:, None] * a.double()[None, :]).sum())
    descent = float((gv.double() * tangent.double()).sum())
    residual = float((g0.double() * tangent.double()).sum())
    if energy <= 1e-6 * vv or descent <= 0:
        return None, dict(status='DEGENERATE_RESIDUAL', E=energy, kvv=vv, k00=zz)
    if abs(residual) > 1e-4 * math.sqrt(zz * energy) or abs(descent-energy) > 1e-4 * energy:
        raise ValueError('Unstable FP32 residualization')
    return tangent, dict(status='OK', E=energy, kvv=vv, k00=zz,
                         kv0=vz, null_directional_derivative=residual)


def gradient_pair(runtime, layer, pair):
    values, gradients = [], []
    for batch in pair:
        loss = nll(runtime.model, batch)
        gradients.append(finite(torch.autograd.grad(loss, layer.weight)[0].detach()))
        values.append(float(loss.detach()))
    return values, gradients


def calibrate(runtime, records, candidate_layers, *, train_ids, excluded_ids, binding):
    """Explicit train-only calibration and mean-E fixed layer selection; no I/O."""
    records, candidate_layers = list(records), list(candidate_layers)
    ids = [r.record_id for r in records]
    if not binding or not records or len(ids) != len(set(ids)):
        raise ValueError('Need unique calibration records and a model/data binding')
    if not set(ids) <= set(train_ids) or set(ids) & set(excluded_ids):
        raise ValueError('Calibration split violation')
    if not candidate_layers or len(candidate_layers) != len(set(candidate_layers)):
        raise ValueError('Need distinct candidate FFN output layers')
    factors, profile = {}, {}
    for name in candidate_layers:
        layer = runtime.get_module(name)
        a = torch.zeros(layer.weight.shape[1], device=layer.weight.device, dtype=torch.float64)
        b = torch.zeros(layer.weight.shape[0], device=layer.weight.device, dtype=torch.float64)
        count = 0
        with editable_fp32(runtime.model, layer):
            for record in records:
                batch = runtime.build_edit_batch(record)
                captured = []
                handle = layer.register_forward_hook(
                    lambda m, args, out: captured.append((args[0], out)), prepend=True)
                try:
                    loss = nll(runtime.model, batch)
                finally:
                    handle.remove()
                if len(captured) != 1:
                    raise ValueError('Expected one FFN invocation per forward')
                x, output = captured[0]
                gradient = torch.autograd.grad(loss, output)[0]
                valid = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
                if batch.attention_mask is not None:
                    valid = batch.attention_mask.bool()
                if valid.shape != x.shape[:-1] or not valid.any():
                    raise ValueError('Invalid calibration attention mask')
                a += finite(x.detach()[valid].double()).square().sum(0)
                b += finite(gradient.detach()[valid].double()).square().sum(0)
                count += int(valid.sum())
            af, bf = damp((a/count).float()), damp((b/count).float())
            energies = []
            for record in records:
                _, gradients = gradient_pair(runtime, layer, paired_batches(runtime, record))
                _, geometry = direction(*gradients, af, bf)
                energies.append(geometry['E'] if geometry['status'] == 'OK' else 0.)
        factors[name] = dict(a=af.cpu(), b=bf.cpu(), positions=count)
        profile[name] = sum(energies) / len(energies)
    selected = max(candidate_layers, key=profile.get)
    if profile[selected] <= 0:
        raise ValueError('No nondegenerate training editability candidate')
    return dict(version=VERSION, binding=dict(binding), calibration_ids=ids,
                selected_layer=selected, mean_E=profile, factors=factors)


class Editor:
    """Direct native-W editor; caller retains existing generation/evaluation loop."""
    def __init__(self, runtime, calibration, *, binding):
        if calibration['version'] != VERSION or calibration['binding'] != binding:
            raise ValueError('Calibration binding mismatch')
        self.runtime = runtime
        self.binding = dict(binding)
        self.name = calibration['selected_layer']
        self.layer = runtime.get_module(self.name)
        factor = calibration['factors'][self.name]
        self.a = finite(factor['a'].to(self.layer.weight.device, torch.float32).clone())
        self.b = finite(factor['b'].to(self.layer.weight.device, torch.float32).clone())
        if self.a.shape != (self.layer.weight.shape[1],) or self.b.shape != (self.layer.weight.shape[0],):
            raise ValueError('Calibration dimensions mismatch')
        if (self.a <= 0).any() or (self.b <= 0).any():
            raise ValueError('Invalid Fisher factors')
        self.calibration_ids = set(calibration['calibration_ids'])

    def edit(self, record):
        if record.record_id in self.calibration_ids:
            raise ValueError('Edit request overlaps calibration')
        original = self.layer.weight.detach().clone()
        steps = []
        try:
            with editable_fp32(self.runtime.model, self.layer):
                pair = paired_batches(self.runtime, record)
                for step in range(MAX_STEPS):
                    losses, gradients = gradient_pair(self.runtime, self.layer, pair)
                    row = dict(step=step+1, visual_nll=losses[0], null_nll=losses[1])
                    steps.append(row)
                    if losses[0] <= TARGET_NLL:
                        row['status'] = 'TARGET_ALREADY_MET'
                        break
                    tangent, geometry = direction(*gradients, self.a, self.b)
                    row.update(geometry)
                    if tangent is None:
                        break
                    requested = losses[0] - TARGET_NLL
                    effective = min(requested, math.sqrt(2 * TRUST_BUDGET * geometry['E']))
                    delta = finite((-effective / geometry['E']) * tangent)
                    with torch.no_grad():
                        self.layer.weight.add_(delta)
                    finite(self.layer.weight)
                    row.update(requested_reduction=requested, effective_reduction=effective,
                               fisher_cost=effective**2/(2*geometry['E']), status='UPDATED')
                with torch.no_grad():
                    fp32_losses = [float(nll(self.runtime.model, batch)) for batch in pair]
                    native = finite(self.layer.weight.to(original.dtype))
                    rounding = float(torch.linalg.vector_norm(self.layer.weight-native.float()))
            # Evaluate the actual deployed dtype without temporary editing hooks.
            modes = [(m, m.training) for m in self.runtime.model.modules()]
            try:
                self.runtime.model.eval()
                with torch.no_grad():
                    native_losses = [float(nll(self.runtime.model, batch)) for batch in pair]
            finally:
                for module, mode in modes:
                    module.training = mode
        except BaseException:
            self.layer.weight.data = original
            raise
        return dict(version=VERSION, layer=self.name, steps=steps, fp32_final_nll=fp32_losses,
                    native_final_nll=native_losses, native_target_met=native_losses[0] <= TARGET_NLL,
                    native_dtype=str(original.dtype), native_rounding_l2=rounding)

    def state_dict(self):
        return dict(version=VERSION, binding=self.binding, layer=self.name,
                    a=self.a.cpu().clone(), b=self.b.cpu().clone(),
                    calibration_ids=sorted(self.calibration_ids),
                    weight=self.layer.weight.detach().cpu().clone())

    def load_state_dict(self, state):
        if (state['version'] != VERSION or state['binding'] != self.binding or state['layer'] != self.name
                or set(state['calibration_ids']) != self.calibration_ids
                or not torch.equal(state['a'], self.a.cpu()) or not torch.equal(state['b'], self.b.cpu())):
            raise ValueError('Editor state binding mismatch')
        weight = finite(state['weight'])
        if weight.shape != self.layer.weight.shape or weight.dtype != self.layer.weight.dtype:
            raise ValueError('Native weight mismatch')
        with torch.no_grad():
            self.layer.weight.copy_(weight.to(self.layer.weight.device))
