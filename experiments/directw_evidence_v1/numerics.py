"""Exact matrix-free categorical GGN and small constraint-space QP.

Dense parameter geometry is allowed ONLY in the bounded hard-QP reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable
import torch
from torch import Tensor

Operator = Callable[[Tensor], Tensor]


def position_weights(mask: Tensor, sources: tuple[str, ...]) -> Tensor:
    """Equal source, then equal sample, then equal effective predictor position."""
    if mask.ndim != 2 or len(sources) != len(mask) or not mask.any(dim=1).all():
        raise ValueError("every protection sample must have effective positions")
    groups = sorted(set(sources))
    weights = torch.zeros_like(mask, dtype=torch.float64)
    for source in groups:
        indices = [i for i, s in enumerate(sources) if s == source]
        for i in indices:
            weights[i] = mask[i] / (len(groups) * len(indices) * mask[i].sum())
    return weights


def protection_kl(logits: Tensor, anchor: Tensor, weights: Tensor) -> Tensor:
    """Fixed detached teacher and fixed prefix; full-vocabulary KL(q_ref || p)."""
    if anchor.requires_grad or anchor.shape != logits.shape or weights.shape != logits.shape[:-1]:
        raise ValueError("teacher/position mismatch")
    if not torch.isfinite(anchor).all() or (anchor < 0).any() or not torch.allclose(
        anchor.sum(-1), torch.ones_like(anchor[..., 0]), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("invalid teacher distribution")
    return (weights.to(logits) * (anchor * (anchor.clamp_min(torch.finfo(anchor.dtype).tiny).log()
            - logits.log_softmax(-1))).sum(-1)).sum()


def protection_gradient(logits: Operator, point: Tensor, anchor: Tensor, weights: Tensor) -> Tensor:
    """Categorical KL pullback J^T w(p-q); identical fixed teacher gives zero.

    Uses the analytic probability residual, avoiding normalization roundoff in
    the log-softmax backward. This does not drop nonstationary protection terms.
    """
    value, pullback = torch.func.vjp(logits, point)
    protection_kl(value, anchor, weights)  # Validate the fixed teacher contract.
    residual = weights.to(value).unsqueeze(-1) * (value.softmax(-1) - anchor)
    return pullback(residual)[0].detach()


class FrozenGGN:
    """Freeze local point, probabilities, position weights and Torch CPU RNG.

    Closure must use cloned inputs/prefix/mask and an eval-mode functional model.
    The editor constructs a fresh instance at each accepted local iterate.
    """
    def __init__(self, logits: Operator, point: Tensor, weights: Tensor):
        self.logits = logits
        self.point = point.detach().clone()
        self.weights = weights.detach().clone()
        self.rng = torch.random.get_rng_state().clone()
        self.calls = 0
        self.probability = self._fixed(lambda: logits(self.point).softmax(-1).detach())
        if self.probability.shape[:-1] != weights.shape:
            raise ValueError("GGN predictor positions mismatch")

    def _fixed(self, call: Callable[[], Tensor]) -> Tensor:
        # Does not advance caller RNG; no CUDA RNG access in stage A.
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(self.rng)
            return call()

    def __call__(self, vector: Tensor) -> Tensor:
        if vector.shape != self.point.shape:
            raise ValueError("GGN vector shape mismatch")
        def compute() -> Tensor:
            _, jv = torch.func.jvp(self.logits, (self.point,), (vector,))
            p = self.probability
            fisher_jv = p * (jv - (p * jv).sum(-1, keepdim=True))
            _, pullback = torch.func.vjp(self.logits, self.point)
            return pullback(self.weights.to(p).unsqueeze(-1) * fisher_jv)[0].detach()
        self.calls += 1
        return self._fixed(compute)


def ggn_matvec(logits: Operator, weight: Tensor, vector: Tensor, weights: Tensor) -> Tensor:
    return FrozenGGN(logits, weight, weights)(vector)


@dataclass
class CGResult:
    value: Tensor
    status: str
    relative_residual: float
    iterations: int
    calls: int
    condition_surrogate: float
    recursive_relative_residuals: list[float] = field(default_factory=list)


def cg(operator: Operator, rhs: Tensor, *, rtol: float = 1e-8, max_iter: int = 32) -> CGResult:
    """Zero-start deterministic SPD solve; repeat probe rejects changing operators."""
    if rtol <= 0 or max_iter < 1 or not torch.isfinite(rhs).all():
        raise ValueError("invalid CG input")
    probe = torch.ones_like(rhs)
    first, second = operator(probe), operator(probe)
    calls = 2
    if not torch.equal(first, second):
        return CGResult(torch.zeros_like(rhs), "NONSTATIONARY", float("inf"), 0, calls, float("inf"))
    x, r, p = torch.zeros_like(rhs), rhs.clone(), rhs.clone()
    norm = float(rhs.norm())
    if norm == 0:
        return CGResult(x, "CONVERGED", 0., 0, calls, 1.)
    rr = (r * r).sum()
    rayleigh, residuals = [], []
    status, iterations = "NOT_CONVERGED", 0
    for iterations in range(1, max_iter + 1):
        ap = operator(p); calls += 1
        pap = (p * ap).sum()
        if not torch.isfinite(ap).all() or float(pap) <= 0:
            status = "NON_SPD"; break
        rayleigh.append(float(pap / (p * p).sum()))
        alpha = rr / pap
        x = x + alpha * p
        r = r - alpha * ap
        next_rr = (r * r).sum()
        residuals.append(float(next_rr.sqrt()) / norm)
        if float(next_rr.sqrt()) <= rtol * norm:
            status = "CONVERGED"; break
        p = r + (next_rr / rr) * p
        rr = next_rr
    actual = float((operator(x) - rhs).norm()) / norm; calls += 1
    if actual > rtol and status == "CONVERGED":
        status = "NOT_CONVERGED"
    condition = max(rayleigh) / min(rayleigh) if rayleigh else float("inf")
    return CGResult(x, status, actual, iterations, calls, condition, residuals)


@dataclass
class StepResult:
    direction: Tensor
    slack: Tensor
    alpha: Tensor
    status: str
    kkt: dict[str, float]
    cg: list[CGResult]
    dual_iterations: int
    active_constraints: list[int]


def solve_step(operator: Operator, constraints: Tensor, rhs: Tensor, preservation_grad: Tensor,
               *, nu: float = 10., max_active: int = 8, cg_rtol: float = 1e-8,
               cg_max_iter: int = 64, dual_tol: float = 1e-8,
               dual_max_iter: int = 10000,
               inverse_solutions: list[CGResult] | None = None) -> StepResult:
    """Soft QP via nonnegative dual coordinate descent, never a parameter inverse."""
    a, b, g = constraints, rhs, preservation_grad.reshape(-1)
    if nu <= 0 or a.ndim != 2 or a.shape != (len(b), g.numel()) or not 0 < len(b) <= max_active:
        raise ValueError("invalid slack QP/active constraint budget")
    if not all(torch.isfinite(t).all() for t in (a, b, g)):
        raise ValueError("nonfinite QP")
    vectors = [g, *a]
    if inverse_solutions is None:
        solves = [cg(operator, v, rtol=cg_rtol, max_iter=cg_max_iter) for v in vectors]
    else:
        if len(inverse_solutions) != len(vectors):
            raise ValueError("one inverse solution per preservation/constraint RHS required")
        solves = []
        for prior, v in zip(inverse_solutions, vectors):
            value = prior.value.reshape(-1)
            if value.shape != v.shape or value.dtype != v.dtype or value.device != v.device:
                raise ValueError("reused inverse shape/dtype/device mismatch")
            # Reuse never trusts an old convergence flag: verify the CURRENT Q.
            residual = operator(value) - v
            norm = float(v.norm())
            relative = float(residual.norm()) / norm if norm else (0. if torch.equal(residual, torch.zeros_like(v)) else float("inf"))
            okay = prior.status == "CONVERGED" and bool(torch.isfinite(residual).all()) and relative <= cg_rtol
            solves.append(CGResult(value, "CONVERGED" if okay else "NOT_CONVERGED", relative, 0, 1, prior.condition_surrogate))
    qg = solves[0].value
    qa = torch.stack([s.value for s in solves[1:]], dim=1)
    h = a @ qa + torch.eye(len(b), dtype=a.dtype, device=a.device) / nu
    target = b + a @ qg
    alpha = torch.zeros_like(b)
    solver_ok = all(s.status == "CONVERGED" for s in solves)
    stationary = torch.allclose(h, h.T, atol=dual_tol, rtol=dual_tol)
    if not stationary or (h.diag() <= 0).any():
        solver_ok = False
    iterations = 0
    if solver_ok:
        for iterations in range(1, dual_max_iter + 1):
            for j in range(len(b)):
                alpha[j] = (alpha[j] + (target[j] - h[j] @ alpha) / h[j, j]).clamp_min(0)
            derivative = h @ alpha - target
            projected = alpha - (alpha - derivative).clamp_min(0)
            if float(projected.abs().max()) <= dual_tol:
                break
    d = qa @ alpha - qg
    xi = alpha / nu
    gap = a @ d + xi - b
    stationarity = operator(d) + g - a.T @ alpha
    kkt = dict(primal=float((-gap).clamp_min(0).max()),
               stationarity=float(stationarity.norm()),
               complementarity=float((alpha * gap).abs().max()),
               slack_stationarity=float((nu * xi - alpha).abs().max()),
               dual_projected=float((alpha - (alpha - (h @ alpha - target)).clamp_min(0)).abs().max()))
    scale = max(1., float(g.norm()), float((a.T @ alpha).norm()))
    okay = solver_ok and max(kkt["primal"], kkt["complementarity"], kkt["dual_projected"]) <= 10 * dual_tol
    okay = okay and kkt["stationarity"] <= max(10 * dual_tol, 10 * cg_rtol * scale)
    return StepResult(d.reshape(preservation_grad.shape), xi, alpha,
                      "CONVERGED" if okay else "NOT_CONVERGED", kkt, solves, iterations,
                      [i for i, value in enumerate(alpha) if float(value) > dual_tol])


def hard_qp_reference(q: Tensor, a: Tensor, b: Tensor, g: Tensor,
                      tol: float = 1e-8) -> tuple[str, Tensor | None]:
    """Exhaustive active-set oracle for <=64 variables / <=10 constraints ONLY."""
    if q.shape != (g.numel(), g.numel()) or g.numel() > 64 or len(b) > 10:
        raise ValueError("hard reference is a bounded CPU oracle")
    if not torch.allclose(q, q.T) or float(torch.linalg.eigvalsh(q).min()) <= 0:
        raise ValueError("hard QP requires SPD")
    best, objective = None, float("inf")
    for size in range(min(len(b), len(g)) + 1):
        for subset in combinations(range(len(b)), size):
            active = a[list(subset)]
            kkt = torch.cat((torch.cat((q, -active.T), 1),
                            torch.cat((active, q.new_zeros(size, size)), 1)), 0)
            try:
                solution = torch.linalg.solve(kkt, torch.cat((-g, b[list(subset)])))
            except torch.linalg.LinAlgError:
                continue
            d, dual = solution[:len(g)], solution[len(g):]
            if (a @ d >= b - tol).all() and (dual >= -tol).all():
                value = float(.5 * d @ q @ d + g @ d)
                if value < objective:
                    best, objective = d, value
    return ("FEASIBLE", best) if best is not None else ("INFEASIBLE", None)
