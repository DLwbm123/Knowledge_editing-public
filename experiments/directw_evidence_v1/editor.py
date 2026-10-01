"""Single native-matrix editing with normal-forward acceptance and rollback.

No adapter, residual hook, optimizer restoration or legacy runtime imports.
Native template preparation remains a separately gated qualification task.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import copy
import hashlib
import json
import os
from pathlib import Path
import torch
from torch import Tensor, nn
from .numerics import FrozenGGN, position_weights, protection_kl, protection_gradient, solve_step

BRANCHES = ("W_FT", "W_EUCLIDEAN_QP", "W_KEY_QP", "W_FUNCTIONAL_QP", "W_EVIDENCE_QP")


def tensor_digest(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def mean_answer_logprob(logits: Tensor, labels: Tensor, *, eos_id: int,
                        include_eos: bool = False) -> Tensor:
    """Labels are at the MODEL-EXPANDED positions; score predictor t for label t+1.

    Prefix, image expansion and padding labels must be -100. Average each answer
    separately so answer length / padding never silently changes its weight.
    """
    if logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise ValueError("expanded logits/labels mismatch")
    target = labels[:, 1:]
    mask = target != -100
    if not include_eos:
        mask = mask & (target != eos_id)
    if not mask.any(dim=1).all():
        raise ValueError("empty content answer")
    if ((target[mask] < 0) | (target[mask] >= logits.shape[-1])).any():
        raise ValueError("invalid content token")
    safe = target.masked_fill(~mask, 0)
    score = logits[:, :-1].log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (score * mask).sum(-1) / mask.sum(-1)


@dataclass
class Constraint:
    name: str
    role: str
    score: Callable[[Tensor], Tensor]
    threshold: float
    cross_image: bool = False
    protection_group: str | None = None
    normal_score: Callable[[], Tensor] | None = None


def build_constraints(supports: list[tuple[str, Callable[[Tensor], Tensor], float, Callable[[], Tensor]]],
                      pairs: list[dict[str, Any]], branch: str) -> list[Constraint]:
    """All branches get BOTH endpoint supervision; only evidence adds cross-image."""
    if branch not in BRANCHES:
        raise ValueError("unknown branch")
    constraints = [Constraint(name, "EDIT_FIT", score, threshold, normal_score=normal) for name, score, threshold, normal in supports]
    for pair in pairs:
        if pair.get("verified") is not True or pair.get("scope") != "OUT_OF_SCOPE" or pair.get("negative_base_correct") is not True:
            raise ValueError("unqualified visual pair")
        if not pair.get("negative_protection_group"):
            raise ValueError("visual counterexample requires its own protection group")
        plus, minus = pair["plus_margin"], pair["minus_margin"]
        normal_plus, normal_minus = pair["normal_plus_margin"], pair["normal_minus_margin"]
        constraints.extend([Constraint(pair["id"] + ":plus", "VIS_PAIR_FIT", plus, pair["endpoint_margin"], normal_score=normal_plus),
                            Constraint(pair["id"] + ":minus", "VIS_PAIR_FIT", minus, pair["endpoint_margin"],
                                       protection_group=pair["negative_protection_group"], normal_score=normal_minus)])
        if branch == "W_EVIDENCE_QP":
            constraints.append(Constraint(pair["id"] + ":cross", "VIS_PAIR_FIT",
                                          lambda w, p=plus, n=minus: p(w) + n(w),
                                          pair["visual_margin"], True, normal_score=lambda p=normal_plus, n=normal_minus: p() + n()))
    return constraints


@dataclass
class ProtectionGroup:
    name: str
    logits: Callable[[Tensor], Tensor]
    anchor: Tensor
    mask: Tensor
    sources: tuple[str, ...]
    coefficient: float
    budget: float
    binding: dict[str, str]
    keys: Tensor | None = None
    normal_logits: Callable[[], Tensor] | None = None
    deployment_anchor: Tensor | None = None
    deployment_binding: dict[str, str] | None = None

    def __post_init__(self) -> None:
        required = {"model", "weight_version", "input", "prefix", "mask", "dtype", "backend", "config", "teacher_version"}
        if set(self.binding) != required or not all(self.binding.values()):
            raise ValueError("incomplete protection binding")
        if self.coefficient <= 0 or self.budget < 0 or self.anchor.requires_grad:
            raise ValueError("invalid protection anchor/budget")
        self.anchor = self.anchor.detach().clone()
        if self.deployment_anchor is not None:
            if self.deployment_anchor.requires_grad or self.deployment_binding is None or set(self.deployment_binding) != required or not all(self.deployment_binding.values()):
                raise ValueError("incomplete detached deployment teacher binding")
            self.deployment_anchor = self.deployment_anchor.detach().clone()
        self.mask = self.mask.detach().clone()
        self.weights = position_weights(self.mask, self.sources)
        if self.keys is not None:
            self.keys = self.keys.detach().clone()

    def loss(self, weight: Tensor) -> Tensor:
        return protection_kl(self.logits(weight), self.anchor, self.weights)


class MatrixRuntime:
    """Bind one unshared original 2D weight; freeze all other functional state."""
    def __init__(self, model: nn.Module, path: str):
        if model.training or any(m.training for m in model.modules()):
            raise ValueError("model must be eval mode")
        parameters = dict(model.named_parameters(remove_duplicate=False))
        if path not in parameters or parameters[path].ndim != 2 or not path.endswith(".weight"):
            raise ValueError("one original 2D matrix required")
        self.model, self.path, self.weight = model, path, parameters[path]
        buffers = dict(model.named_buffers(remove_duplicate=False))
        aliases = [name for name, p in {**parameters, **buffers}.items() if p.untyped_storage().data_ptr() == self.weight.untyped_storage().data_ptr()]
        if aliases != [path]:
            raise ValueError("shared editable weight requires separate review")
        self.base = self.weight.detach().clone()
        self.base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # state_dict omits non-persistent RoPE/position buffers; strict functional
        # replacement requires every named parameter AND every named buffer.
        self.buffer_base = {k: v.detach().cpu().clone() for k, v in buffers.items()}
        self.functional_state = {k: v.detach() for k, v in parameters.items() if k != path}
        self.functional_state.update({k: v.detach().clone() for k, v in buffers.items()})
        self.base_hooks = self.hooks()
        self.version = 0
        self.last_rounding: dict[str, float] = {}

    def logits(self, weight: Tensor, inputs: dict[str, Any]) -> Tensor:
        if "past_key_values" in inputs or inputs.get("use_cache", False):
            raise ValueError("editing forward cannot reuse KV cache")
        state = {**self.functional_state, self.path: weight}
        result = torch.func.functional_call(self.model, state, (), inputs, strict=True)
        return result.logits if hasattr(result, "logits") else result

    def normal_logits(self, inputs: dict[str, Any]) -> Tensor:
        if "past_key_values" in inputs or inputs.get("use_cache", False):
            raise ValueError("stale/native KV cache rejected")
        result = self.model(**inputs)
        return result.logits if hasattr(result, "logits") else result

    def bind_inputs(self, inputs: dict[str, Any], *, arithmetic_dtype: torch.dtype | None = None) -> Callable[[Tensor], Tensor]:
        """Freeze tensor inputs, template prefixes and masks for one solver batch."""
        if "past_key_values" in inputs or inputs.get("use_cache", False):
            raise ValueError("cannot bind an inference KV cache")
        fixed = copy.deepcopy(inputs)
        for key, value in fixed.items():
            if isinstance(value, Tensor):
                fixed[key] = value.detach().clone()
        if arithmetic_dtype is not None:
            if arithmetic_dtype != torch.float32:
                raise ValueError("temporary native arithmetic supports explicit FP32 only")
            # Promote the complete functional calculation, not just W followed by
            # a BF16 cast. The physical deployment model is never converted.
            state = {k: v.to(arithmetic_dtype) if v.is_floating_point() else v
                     for k, v in self.functional_state.items()}
            fixed = {k: v.to(arithmetic_dtype) if isinstance(v, Tensor) and v.is_floating_point() else v
                     for k, v in fixed.items()}
            def promoted(weight: Tensor) -> Tensor:
                if weight.dtype != arithmetic_dtype:
                    raise ValueError("solver coordinate and functional arithmetic must match")
                result = torch.func.functional_call(self.model, {**state, self.path: weight}, (), fixed, strict=True)
                return result.logits if hasattr(result, "logits") else result
            return promoted
        return lambda w: self.logits(w, fixed)

    def bind_normal_inputs(self, inputs: dict[str, Any]) -> Callable[[], Tensor]:
        """Explicit physical-model path; never substitutes a functional closure."""
        if "past_key_values" in inputs or inputs.get("use_cache", False):
            raise ValueError("cannot bind an inference KV cache")
        fixed = copy.deepcopy(inputs)
        return lambda: self.normal_logits(fixed)

    def capture_input(self, inputs: dict[str, Any]) -> Tensor:
        """Read-only hook at the actual matrix input, removed even on exception."""
        captured = []
        module = self.model.get_submodule(self.path.rsplit(".", 1)[0])
        hook = module.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                self.normal_logits(inputs)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("ambiguous reused editable module")
        return captured[0]

    def write(self, candidate: Tensor) -> None:
        if candidate.shape != self.weight.shape or not torch.isfinite(candidate).all():
            raise ValueError("invalid candidate weight")
        rounded = candidate.to(self.weight)
        if not torch.isfinite(rounded).all():
            raise ValueError("nonfinite deployment dtype")
        requested = candidate - self.weight.detach().to(candidate)
        actual = rounded - self.weight.detach()
        self.last_rounding = dict(requested_delta_norm=float(requested.norm()),
                                  actual_delta_norm=float(actual.norm()),
                                  zeroed_fraction=float(((actual == 0) & (requested != 0)).sum()) / max(1, int((requested != 0).sum())))
        with torch.no_grad():
            self.weight.copy_(rounded)
        self.version += 1

    def restore_snapshot(self, snapshot: dict[str, Tensor], buffers: dict[str, Tensor],
                         hooks: tuple[int, ...]) -> None:
        """Restore and verify persistent AND nonpersistent edit-start state."""
        current_buffers = dict(self.model.named_buffers(remove_duplicate=False))
        self.model.load_state_dict(snapshot, strict=True)
        if set(current_buffers) != set(buffers):
            raise RuntimeError("cannot restore changed buffer registry")
        with torch.no_grad():
            for key, value in current_buffers.items():
                value.copy_(buffers[key].to(value))
        self.audit(snapshot, hooks, buffer_snapshot=buffers)
        if not torch.equal(self.weight.detach().cpu(), snapshot[self.path]):
            raise RuntimeError("selected weight restoration failed")
        self.version += 1

    def reset_single(self) -> None:
        self.restore_snapshot(self.base_state, self.buffer_base, self.base_hooks)

    def audit(self, snapshot: dict[str, Tensor], hooks: tuple[int, ...], *,
              buffer_snapshot: dict[str, Tensor] | None = None) -> None:
        now = self.model.state_dict()
        if set(now) != set(snapshot) or self.hooks() != hooks:
            raise RuntimeError("module/state key/hook mutation")
        expected_buffers = self.buffer_base if buffer_snapshot is None else buffer_snapshot
        buffers = dict(self.model.named_buffers(remove_duplicate=False))
        if set(buffers) != set(expected_buffers) or any(
            v.shape != expected_buffers[k].shape or v.dtype != expected_buffers[k].dtype
            or not torch.equal(v.detach().cpu(), expected_buffers[k]) for k, v in buffers.items()
        ):
            raise RuntimeError("unauthorized buffer mutation, including non-persistent state")
        for key, value in now.items():
            if value.shape != snapshot[key].shape or value.dtype != snapshot[key].dtype:
                raise RuntimeError("shape/dtype mutation")
            if key != self.path and not torch.equal(value.detach().cpu(), snapshot[key]):
                raise RuntimeError("unauthorized parameter/buffer mutation")

    def hooks(self) -> tuple[int, ...]:
        return tuple(len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
                     for m in self.model.modules())


@dataclass
class EditConfig:
    tau: float
    nu: float
    trust_radius: float
    base_drift_limit: float
    edit_drift_limit: float
    max_steps: int = 20
    max_active: int = 8
    factors: tuple[float, ...] = (1., .5, .25, .125, .0625, .03125, .015625)
    cg_max_iter: int = 32
    cg_rtol: float = 1e-4
    tolerance: float = 1e-8
    ft_lr: float = .1
    arithmetic_dtype: torch.dtype | None = None
    max_ggn_calls: int | None = None
    constraint_value_mode: str = "functional"

    def __post_init__(self) -> None:
        if self.constraint_value_mode not in {"functional", "native_value"}:
            raise ValueError("unknown constraint value mode")
        if self.max_ggn_calls is not None and (type(self.max_ggn_calls) is not int or self.max_ggn_calls < 0):
            raise ValueError("invalid GGN call cap")
        if self.arithmetic_dtype not in (None, torch.float32):
            raise ValueError("explicit temporary FP32 arithmetic or native dtype required")
        if min(self.tau, self.nu, self.trust_radius, self.base_drift_limit, self.edit_drift_limit, self.ft_lr) <= 0:
            raise ValueError("positive budgets required")
        if min(self.max_steps, self.max_active, self.cg_max_iter) < 1 or not self.factors or any(not 0 < f <= 1 for f in self.factors):
            raise ValueError("finite positive iteration/line-search budgets required")


@dataclass
class EditResult:
    status: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    final_violations: dict[str, float] = field(default_factory=dict)
    accepted_steps: int = 0
    ggn_calls: int = 0
    rollback: bool = False
    error: str | None = None
    initial_protection: dict[str, float] = field(default_factory=dict)


def constraint_values(constraints: list[Constraint], weight: Tensor) -> Tensor:
    return torch.stack([c.score(weight).reshape(()) for c in constraints])


def edit_one(runtime: MatrixRuntime, constraints: list[Constraint], groups: list[ProtectionGroup],
             config: EditConfig, branch: str, *, attempt_log: Path | None = None) -> EditResult:
    """Rejected/final-unsatisfied edits restore their edit-start state.

    Solver closures and physical normal-forward callbacks are separate contracts.
    FP32 solver teachers never stand in for BF16 deployment protection teachers.
    """
    if branch not in BRANCHES or not constraints or not groups:
        raise ValueError("branch, legal constraints and protection groups required")
    if branch == "W_FT" and config.constraint_value_mode != "functional":
        raise ValueError("native value anchoring is a QP ablation only")
    if branch == "W_EVIDENCE_QP" and not any(c.cross_image for c in constraints):
        raise ValueError("BLOCKED_DATA: evidence branch requires verified pairs")
    if len({g.name for g in groups}) != len(groups) or any(
        c.protection_group and c.protection_group not in {g.name for g in groups} for c in constraints
    ):
        raise ValueError("missing/ambiguous endpoint protection group")
    if any(c.normal_score is None for c in constraints) or any(
        g.normal_logits is None or g.deployment_anchor is None or g.deployment_binding is None for g in groups
    ):
        raise ValueError("explicit normal-forward scores and bound deployment teachers required")
    if runtime.weight.dtype in (torch.bfloat16, torch.float16) and config.arithmetic_dtype != torch.float32:
        raise ValueError("low-precision deployment requires explicit FP32 solver arithmetic")
    snapshot = {k: v.detach().cpu().clone() for k, v in runtime.model.state_dict().items()}
    buffer_snapshot = {k: v.detach().cpu().clone() for k, v in runtime.model.named_buffers(remove_duplicate=False)}
    hooks = runtime.hooks()
    start = runtime.weight.detach().clone()
    coordinate_start = start.to(config.arithmetic_dtype or start.dtype)
    thresholds = coordinate_start.new_tensor([c.threshold for c in constraints])
    result = EditResult("NOT_SATISFIED")
    # Fixed role-stratified rotation, independent of scores/test performance.
    ordered = sorted(range(len(constraints)), key=lambda i: (constraints[i].role, i))

    def journal(event: dict[str, Any]) -> None:
        if attempt_log is not None:
            # Caller supplies a private path inside this run's RUN_ROOT.
            with attempt_log.open("a") as stream:
                stream.write(json.dumps(event,default=str)+"\n")
                stream.flush();os.fsync(stream.fileno())

    def evaluate() -> tuple[Tensor, dict[str, float]]:
        with torch.no_grad():
            values = torch.stack([c.normal_score().reshape(()) for c in constraints]).to(thresholds)
            losses = {g.name: float(protection_kl(g.normal_logits().to(g.deployment_anchor),
                                                g.deployment_anchor, g.weights)) for g in groups}
        runtime.audit(snapshot, hooks, buffer_snapshot=buffer_snapshot)
        return values, losses

    def restore() -> None:
        runtime.restore_snapshot(snapshot, buffer_snapshot, hooks)
        result.rollback = True

    try:
        initial, initial_losses = evaluate()
        deployed_values = initial
        result.initial_protection = initial_losses
        journal(dict(event="EDIT_START",protection=initial_losses,weight_version=runtime.version))
        runtime.audit(snapshot, hooks, buffer_snapshot=buffer_snapshot)
        if any(not torch.isfinite(torch.tensor(v)) or v > g.budget + 1e-12
               for g in groups for v in [initial_losses[g.name]]):
            result.status = "START_BUDGET_VIOLATION"
            journal(dict(event="TERMINAL",status=result.status,protection=initial_losses))
            return result
        for step in range(config.max_steps):
            if bool(torch.isfinite(deployed_values).all()) and float((thresholds - deployed_values).clamp_min(0).max()) <= config.tolerance:
                result.status = "ACCEPTED"; break
            w = runtime.weight.detach().to(coordinate_start).clone().requires_grad_(True)
            full = constraint_values(constraints, w)
            preservation = sum(g.coefficient * g.loss(w) for g in groups)
            gpres = (sum(g.coefficient * protection_gradient(g.logits, w.detach(), g.anchor, g.weights) for g in groups)
                     if config.arithmetic_dtype == torch.float32
                     else torch.autograd.grad(preservation, w, retain_graph=True)[0].detach())
            active = [ordered[(step * config.max_active + j) % len(ordered)]
                      for j in range(min(config.max_active, len(ordered)))]
            a = torch.stack([torch.autograd.grad(full[i], w, retain_graph=True)[0].flatten() for i in active]).detach()
            # Optional ablation: match affine values to deployment, retain FP32 Jacobian.
            values_for_qp = deployed_values if config.constraint_value_mode == "native_value" else full.detach()
            b = (thresholds - values_for_qp)[active]
            journal(dict(event="LINEARIZATION",step=step,mode=config.constraint_value_mode,
                         functional_scores=full.detach().cpu().tolist(),native_scores=deployed_values.cpu().tolist(),
                         rhs=b.detach().cpu().tolist()))
            frozen = [FrozenGGN(g.logits, w, g.weights) for g in groups]
            def operator(v: Tensor) -> Tensor:
                shaped = v.reshape_as(w)
                out = config.tau * shaped
                for g, curvature in zip(groups, frozen):
                    if branch in ("W_FUNCTIONAL_QP", "W_EVIDENCE_QP"):
                        if config.max_ggn_calls is not None and result.ggn_calls >= config.max_ggn_calls:
                            raise TimeoutError("GGN call safety ceiling reached")
                        result.ggn_calls += 1
                        out = out + g.coefficient * curvature(shaped)
                    elif branch == "W_KEY_QP":
                        if g.keys is None or g.keys.shape[:-1] != g.mask.shape or g.keys.shape[-1] != w.shape[-1]:
                            raise ValueError("actual aligned matrix inputs required for key geometry")
                        keys = g.keys.reshape(-1, w.shape[-1]).to(w)
                        out = out + g.coefficient * ((shaped @ keys.T) * g.weights.to(w).flatten()) @ keys
                return out.reshape_as(v)
            qp = None
            if branch == "W_FT":
                objective = .5 * (thresholds - full).clamp_min(0).square().sum() + preservation + .5 * config.tau * (w - coordinate_start).square().sum()
                direction = -config.ft_lr * torch.autograd.grad(objective, w)[0].detach()
            else:
                qp = solve_step(operator, a, b, gpres, nu=config.nu, max_active=config.max_active,
                                cg_rtol=config.cg_rtol, cg_max_iter=config.cg_max_iter)
                direction = qp.direction
                if qp.status != "CONVERGED":
                    result.attempts.append(dict(step=step, status=qp.status, kkt=qp.kkt))
                    result.status = "SOLVER_NOT_CONVERGED"; break
            norm = float(direction.norm())
            clipped = norm > config.trust_radius
            if clipped:
                direction = direction * (config.trust_radius / norm)
            before = float((thresholds - deployed_values).clamp_min(0).square().sum())
            accepted = False
            for factor in config.factors:
                attempt = dict(step=step,factor=factor,accepted=False,status="STARTED")
                result.attempts.append(attempt)
                journal(dict(event="ATTEMPT",**attempt))
                runtime.write(w.detach() + factor * direction)
                runtime.audit(snapshot, hooks, buffer_snapshot=buffer_snapshot)
                values, losses = evaluate()
                merit = float((thresholds - values).clamp_min(0).square().sum())
                finite = bool(torch.isfinite(values).all()) and all(torch.isfinite(torch.tensor(v)) for v in losses.values())
                budget_ok = all(losses[g.name] <= g.budget + 1e-12 for g in groups)
                edit_drift = float((runtime.weight.to(coordinate_start) - coordinate_start).norm())
                base_drift = float((runtime.weight.to(coordinate_start) - runtime.base.to(coordinate_start)).norm())
                budget_ok = budget_ok and edit_drift <= config.edit_drift_limit and base_drift <= config.base_drift_limit
                # Merit has squared-score units; constraint tolerance has score units.
                merit_roundoff = 16 * torch.finfo(values.dtype).eps * max(before, torch.finfo(values.dtype).tiny)
                accepted = finite and budget_ok and merit < before - merit_roundoff
                attempt.update(dict(step=step, factor=factor, accepted=accepted, merit=merit,
                    status="ACCEPTED" if accepted else "REJECTED",
                    protection=losses, clipped=clipped, edit_drift=edit_drift, base_drift=base_drift,
                    rounding=copy.copy(runtime.last_rounding), actual_scores=values.tolist(),
                    qp=None if qp is None else dict(status=qp.status, slack=qp.slack.tolist(), kkt=qp.kkt,
                        cg=[dict(status=s.status, residual=s.relative_residual, iterations=s.iterations,
                                 calls=s.calls, condition_surrogate=s.condition_surrogate) for s in qp.cg],
                        matvecs=sum(f.calls for f in frozen), active=active)))
                journal(dict(event="ATTEMPT_RESULT",**attempt))
                if accepted:
                    result.accepted_steps += 1; deployed_values = values; break
                runtime.write(w.detach())
            if not accepted:
                result.status = "BACKTRACK_REJECTED"; break
        values, final_losses = evaluate()
        result.final_violations = {c.name: float((thresholds[i] - values[i]).clamp_min(0)) for i, c in enumerate(constraints)}
        if bool(torch.isfinite(values).all()) and max(result.final_violations.values()) <= config.tolerance and all(
            torch.isfinite(torch.tensor(final_losses[g.name])) and final_losses[g.name] <= g.budget + 1e-12 for g in groups
        ):
            result.status = "ACCEPTED"
        elif result.status == "ACCEPTED":
            result.status = "NOT_SATISFIED"
        if result.status != "ACCEPTED":
            restore()
        runtime.audit(snapshot, hooks, buffer_snapshot=buffer_snapshot)
        journal(dict(event="TERMINAL",status=result.status,rollback=result.rollback,violations=result.final_violations))
    except BaseException as exc:
        result.error = type(exc).__name__ + ": " + str(exc)
        try:
            restore()
            result.status = "EXCEPTION_ROLLED_BACK"
        except BaseException as restoration_error:
            result.rollback = False
            result.status = "ROLLBACK_FAILED"
            result.error += "; restoration: " + str(restoration_error)
        if result.attempts and result.attempts[-1].get("status") == "STARTED":
            result.attempts[-1].update(status=result.status,error=result.error)
        journal(dict(event="TERMINAL",status=result.status,rollback=result.rollback,error=result.error))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    return result


def geometry(keys_plus: Tensor, keys_minus: Tensor, grad_plus: Tensor, grad_minus: Tensor) -> dict[str, float]:
    """Caller must supply same predictor-position alignment at actual matrix input."""
    if keys_plus.shape != keys_minus.shape:
        raise ValueError("fixed token alignment required")
    p, n = keys_plus.flatten(), keys_minus.flatten()
    cosine = lambda a, b: float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()).clamp_min(1e-30))
    return dict(distance=float((p - n).norm()), plus_norm=float(p.norm()), minus_norm=float(n.norm()),
                cosine=cosine(p, n), gradient_cosine=cosine(grad_plus, grad_minus))
