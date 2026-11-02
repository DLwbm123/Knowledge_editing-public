"""Check both original hooks and the midpoint on an admitted source forward."""
import torch
from operator_bridge import OperatorBridge


def check(runtime, r, task, rows, config, med_state, du_state, budget):
    if [row['audit_role'] for row in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
        raise ValueError('source roles differ')
    runtime.model.half()
    try:
        prepared = r.s.prepare(runtime, {'configuration_only': config}, task, rows, True)
    finally:
        r.old.fp32(runtime)
    batch = r.old.batch_on(runtime, prepared[0][1])
    med = r.p.expert(med_state, task['seed'], runtime.device).requires_grad_(False)
    du = r.p.expert(du_state, task['seed'], runtime.device).requires_grad_(False)
    layer = runtime.get_module('model.layers.21.mlp.down_proj')
    hook_class = r.q.scoped.replay.MedTraceLayerHook
    observed = []

    def verify(module, args, output):
        if observed:
            raise ValueError('unexpected repeated layer call')
        mask = torch.ones(args[0].shape[:-1], dtype=torch.bool, device=args[0].device)
        def apply(expert):
            hook = hook_class(module, expert)
            hook.set_request_routing(mask)
            return hook._forward_hook(module, args, output)
        for weight, expert in ((0.0, med), (1.0, du)):
            endpoint = OperatorBridge(med, du, weight).to(runtime.device)
            if not torch.equal(apply(endpoint), apply(expert)):
                raise ValueError('original expert and bridge endpoint differ')
        midpoint = apply(OperatorBridge(med, du, 0.5).to(runtime.device))
        a, b = med.residual(args[0]), du.residual(args[0])
        expected = output + (a * 0.5 + b * 0.5).to(output.dtype)
        if not torch.equal(midpoint, expected):
            raise ValueError('midpoint layer output differs')
        observed.append(True)

    handle = layer.register_forward_hook(verify)
    try:
        budget()
        with torch.no_grad():
            runtime.model(**batch.forward_kwargs())
    finally:
        handle.remove()
    if observed != [True]:
        raise ValueError('actual layer output not checked')
    return {'owner': task['order'], 'endpoint_checks': 2, 'midpoint_checks': 1}
