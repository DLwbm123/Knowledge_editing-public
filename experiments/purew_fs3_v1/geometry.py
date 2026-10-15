"""Token-mean Fisher probes; FD certificate versus finite full-support covariance."""
import math
import os
import torch
from curvature import Risk as FDRisk, dot


class SupportRisk(FDRisk):
    def restore(self, state, device):
        assert set(state)=={'B','v','rho','q','ev','erho','eq','compressions'}
        for key,value in state.items():setattr(self,key,value.to(device) if isinstance(value,torch.Tensor) else value)
        assert len(self.B)<=12 and self.rho==0 and self.q>=-1e-9

    def consolidate(self, rows):
        assert torch.isfinite(rows).all()
        device = self.v.device
        previous = self.B.cpu()
        self.B = torch.empty((0, self.v.numel()), dtype=torch.float64, device=device)
        stack = torch.cat((previous, rows.reshape(len(rows), -1).double().cpu()), 0)
        del previous
        assert len(stack) <= 12, 'Finite three-source support only; no automatic expansion'
        _, U = torch.linalg.eigh(stack @ stack.T)
        orthogonal_error = float((U.T @ U - torch.eye(len(U), dtype=torch.float64)).abs().max())
        assert orthogonal_error < 1e-12
        rotated = torch.empty_like(stack)
        for start in range(0, stack.shape[1], 262144):
            rotated[:, start:start+262144] = U.T @ stack[:, start:start+262144]
        del stack
        self.B = rotated.to(device)
        # Full-support rotation removes source-indexed rows without truncating covariance.
        return dict(tau=0., rows=len(rows), retained_rows=len(self.B),
                    orthogonal_error=orthogonal_error, compression='NONE_FINITE_SUPPORT_ROTATION')


def Risk(native):
    return SupportRisk(native) if os.environ['ARM'] == 'S' else FDRisk(native)


def fisher_noise(probability, noise):
    root = probability.sqrt()
    return root * noise - probability * (root * noise).sum(-1, keepdim=True)


def probe_rows(runtime, weight, full, removed, seed):
    from runtime import b, backward
    assert not removed
    gen = torch.Generator(device=runtime.device); gen.manual_seed(seed)
    rows = torch.zeros((4, weight.numel()), device=runtime.device, dtype=torch.float64)
    for j in range(4):
        for batch, coefficient in zip(full, (.5, .125, .125, .125, .125)):
            z, y = b.logits(runtime, batch)
            p = z.detach().softmax(-1)
            noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=gen)
            v = fisher_noise(p, noise)
            # Four independent rows scaled by sqrt(1/4), token mean and source weights.
            rows[j] += backward((z * v).sum() * math.sqrt(coefficient / len(y)) / 2, weight).flatten().double()
    return rows


def native_candidate(master, native, gradient, risk, eta=.01, epsilon=.01):
    delta, event = risk.solve(gradient, eta, epsilon)
    before = native.detach().float()
    event.update(proposal_norm=float(delta.norm()), risk_before=risk.q,
                 unconstrained_norm=float((eta * gradient).norm()),
                 unconstrained_risk=risk.value(-eta * gradient), candidate_diagnostics=[])
    for attempt in range(8):
        candidate_master = master + delta.reshape_as(master).float() * (.5 ** attempt)
        candidate_native = candidate_master.half()
        actual = candidate_native.float() - before
        q = risk.value(actual)
        proximal = float(dot(gradient, actual) + dot(actual, actual)/(2 * eta))
        changed = int(torch.count_nonzero(actual))
        assert math.isfinite(q) and math.isfinite(proximal)
        reasons = ([] if changed else ['ZERO_NATIVE']) + ([] if q <= epsilon else ['RISK']) + ([] if proximal <= 0 else ['PROXIMAL'])
        event['candidate_diagnostics'].append(dict(backtrack=attempt, factor=.5**attempt,
            changed_elements=changed, native_norm=float(actual.norm()), risk=q,
            proximal=proximal, rejection_reasons=reasons))
        if not reasons:
            risk.advance(actual)
            event.update(accepted=True, native_backtracks=attempt, actual_native_norm=float(actual.norm()),
                changed_elements=changed, risk_after=risk.q, proximal_objective=proximal,
                native_master_discrepancy=float((candidate_native.float()-candidate_master).norm()))
            return candidate_master, candidate_native, event
    event.update(accepted=False, reason='FINITE_NATIVE_GATE_REJECT', native_backtracks=8,
        actual_native_norm=0., changed_elements=0, risk_after=risk.q,
        native_master_discrepancy=float((before-master).norm()))
    return master, native.detach(), event
