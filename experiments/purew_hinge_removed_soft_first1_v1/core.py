"""R: certified five-affine fallback; legacy_core is the immutable J core."""
import math
import numpy as np
import torch
from legacy_core import Memory as LegacyMemory, Block, dot, TILE

ETA = .01
EPSILON = .01
KKT = 1e-7


def affine_fallback(memory, gradient, values, gradients):
    """An affine optimum feasible for every quadratic also solves the full QCQP."""
    if len(values) != 5 or len(gradients) != 5:
        raise ValueError('Exactly five current KL constraints required')
    g = gradient.detach().flatten().double().cpu()
    rows = [a.detach().flatten().double().cpu() for a in gradients]
    if g.numel() != memory.n or not torch.isfinite(g).all():
        raise ValueError('Invalid objective gradient')
    if any(a.numel() != memory.n or not torch.isfinite(a).all() for a in rows):
        raise ValueError('Invalid current KL gradient')
    if any(not math.isfinite(k) or k < -1e-12 or k > EPSILON for k in values):
        raise ValueError('Current point must satisfy the unchanged KL gate')
    h = EPSILON - np.asarray(values, dtype=np.float64)
    ag = np.array([dot(a, g) for a in rows])
    gram = np.array([[dot(a, b) for b in rows] for a in rows])
    if not np.isfinite(gram).all() or not np.isfinite(ag).all():
        raise ValueError('Nonfinite reduced products')
    scale = np.sqrt(np.diag(gram))
    audit = dict(status='NOT_CERTIFIED', active_sets_checked=0,
                 singular_sets_skipped=0, certificate_failures=0,
                 reduced_gram=gram.tolist(), reduced_Ag=ag.tolist(),
                 objective_gradient_norm=float(g.norm()), affine_slack_at_origin=h.tolist(),
                 actual_KL_checked=False, infeasibility_proved=False)
    for mask in range(32):
        audit['active_sets_checked'] += 1
        ix = np.array([j for j in range(5) if mask & (1 << j)], dtype=int)
        lam = np.zeros(5)
        if len(ix):
            if np.any(scale[ix] == 0):
                audit['singular_sets_skipped'] += 1
                continue
            # Scaling changes coordinates only; never regularize or truncate a direction.
            normalized = gram[np.ix_(ix, ix)] / np.outer(scale[ix], scale[ix])
            eigen = np.linalg.eigvalsh(normalized)
            cutoff = 64 * np.finfo(np.float64).eps * len(ix) * eigen[-1]
            if eigen[0] <= cutoff:
                audit['singular_sets_skipped'] += 1
                continue
            rhs = (-ag[ix] - h[ix] / ETA) / scale[ix]
            try:
                lam[ix] = np.linalg.solve(normalized, rhs) / scale[ix]
            except np.linalg.LinAlgError:
                audit['singular_sets_skipped'] += 1
                continue
        if not np.isfinite(lam).all() or np.any(lam < 0):
            continue
        slack = h + ETA * (ag + gram @ lam)
        projected = np.where(lam > 1e-8, slack, np.minimum(slack, 0))
        if np.max(np.abs(projected)) > KKT or np.min(slack) < -KKT:
            continue
        z = g.clone()
        for a, multiplier in zip(rows, lam):
            z.add_(a, alpha=float(multiplier))
        d = z * -ETA
        raw_norm = float(d.norm())
        raw_slope = dot(g, d)
        ad = np.array([dot(a, d) for a in rows])
        factor = 1.
        for value, gap in zip(ad, h):
            if value > gap:
                factor = min(factor, max(0., gap / value) * (1 - 32 * np.finfo(float).eps))
        d *= factor
        ad = np.array([dot(a, d) for a in rows])
        risks = {}
        for name, block in memory.blocks.items():
            bd = block.B @ d
            risks[name] = float(block.q + dot(block.v, d) + .5 * dot(bd, bd))
        slack = h - ad
        projected = np.where(lam > 1e-8, slack, np.minimum(slack, 0))
        kkt = float(np.max(np.abs(projected)))
        stationarity = float((d / ETA + z).abs().max())
        complementarity = float(np.max(np.abs(lam * slack)))
        primal = dot(g, d) + .5 * dot(d, d) / ETA
        dual = -.5 * ETA * dot(z, z) - float(lam @ h)
        gap = primal - dual
        record = dict(active_set=ix.tolist(), multipliers=lam.tolist(),
                      raw_proposal_norm=raw_norm, raw_g_dot_d=raw_slope,
                      affine_ray_retraction=factor, affine_slack=slack.tolist(),
                      quadratic_predicted_risks=risks, projected_KKT_residual=kkt,
                      stationarity_inf=stationarity, complementarity_inf=complementarity,
                      primal_objective=primal, dual_lower_bound=dual, primal_dual_gap=gap)
        # Strict feasibility is checked independently of the approximate KKT certificate.
        finite = all(math.isfinite(x) for x in [kkt, stationarity, complementarity, primal, dual, gap, *risks.values()])
        certified = (finite and np.all(slack >= 0) and all(r <= EPSILON for r in risks.values())
                     and kkt <= KKT and stationarity <= KKT and complementarity <= KKT
                     and -1e-10 <= gap <= KKT)
        if not certified:
            audit['certificate_failures'] += 1
            audit['last_failed_certificate'] = record
            continue
        if dot(g, d) >= 0 or float(d.norm()) == 0:
            audit.update(status='CERTIFIED_PROXY_NO_DESCENT', certificate=record)
            return torch.zeros_like(g), audit
        norm = float(d.norm())
        cap = min(1., .05 / norm)
        d *= cap
        audit.update(status='CERTIFIED_AFFINE_OPTIMUM_FULL_PROXY_FEASIBLE', certificate=record,
                     norm_cap_factor=cap, proposal_norm=float(d.norm()),
                     post_cap_g_dot_d=dot(g, d), certificate_applies_before_norm_cap=True)
        return d, audit
    return torch.zeros_like(g), audit


class Memory(LegacyMemory):
    def solve_current(self, gradient, values, gradients):
        proposal, event = super().solve_current(gradient, values, gradients)
        if event['solver_reason'] == 'PASS':
            return proposal, event
        legacy_event = dict(event)
        proposal, fallback = affine_fallback(self, gradient, values, gradients)
        event = dict(event, legacy_solver=legacy_event, fallback=fallback)
        if fallback['status'] == 'CERTIFIED_AFFINE_OPTIMUM_FULL_PROXY_FEASIBLE':
            certificate = fallback['certificate']
            event.update(solver_reason='PASS', solver_path='R_FINITE_ACTIVE_SET',
                         projected_KKT_residual=certificate['projected_KKT_residual'],
                         predicted_risk_stage='CERTIFICATE_BEFORE_NORM_CAP',
                         predicted_risks=dict(certificate['quadratic_predicted_risks']),
                         multipliers={name: 0. for name in self.blocks},
                         proposal_norm=float(proposal.norm()))
            for j, multiplier in enumerate(certificate['multipliers']):
                event['multipliers']['current_KL_' + str(j)] = multiplier
        else:
            event.update(solver_reason='FINITE_DUAL_REJECT', solver_path='R_FALLBACK_NOT_ACCEPTED', proposal_norm=0.)
        event['current_linear_predicted_KLs'] = [float(k + dot(a, proposal)) for k, a in zip(values, gradients)]
        return proposal, event
