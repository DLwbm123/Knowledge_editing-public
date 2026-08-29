"""Pure contracts for the minimal LiveEdit-Med Router-R2 experiment.

The deployable R2 execution change is intentionally narrow: compute the
unchanged Router-R1 route, keep its top-1 expert and its pre-pruning final
coefficient, and execute no other residual.  Rejection is a separate decision
and the NO_EDIT path installs no residual at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import math
import numpy as np
import torch
import torch.nn.functional as F

from methods.liveedit_med.source_ops import BaseRoutePlan, RoutePlan


PROTOCOL = "LIVEEDIT_MED_ROUTER_R2_MINIMAL_CAUSAL_RESCUE_V1"
ROLES = ("native", "textual", "visual", "paired")
EXECUTION_MODES = (
    "R1_MULTI_RESIDUAL",
    "TOP1_ORIGINAL_COEFFICIENT",
    "TOP1_RENORMALIZED_DIAGNOSTIC",
    "TOP1_FULL_STRENGTH_DIAGNOSTIC",
    "NO_EDIT",
)
EMPTY_RUNNER_UP_SENTINEL = -1.0e9


def kplus1_label(*, repository_size: int, target_repository_index: int | None,
                 no_edit: bool) -> int:
    """Return the fixed K+1 CE label; index K is the explicit NO_EDIT class."""
    if repository_size <= 0:
        raise ValueError("ROUTER_R2_INVALID_REPOSITORY_SIZE")
    if no_edit:
        if target_repository_index is not None:
            raise ValueError("ROUTER_R2_NO_EDIT_TARGET_COLLISION")
        return repository_size
    if target_repository_index is None or not 0 <= target_repository_index < repository_size:
        raise ValueError("ROUTER_R2_INVALID_POSITIVE_TARGET")
    return int(target_repository_index)


def kplus1_text_logits(input_text_key: torch.Tensor, expert_text_keys: torch.Tensor,
                       no_edit_text_key: torch.Tensor) -> torch.Tensor:
    """Score K experts and NO_EDIT through the unchanged text scorer family."""
    from methods.liveedit_med.source_ops import SIM_SCALE
    keys = torch.cat([expert_text_keys, no_edit_text_key], dim=0)
    return torch.einsum("ned,med->nme", input_text_key, keys).mean(2) * SIM_SCALE


def kplus1_cross_entropy(logits: torch.Tensor, label: int) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("ROUTER_R2_KPLUS1_LOGIT_SHAPE")
    target = torch.tensor([int(label)], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def kplus1_decision(expert_logits: torch.Tensor, no_edit_logit: torch.Tensor,
                    hard_candidate_mask: torch.Tensor, *, tau_null: float,
                    tau_gap: float) -> dict[str, Any]:
    """Apply the preregistered T1 null/gap rule without changing the hard gate."""
    if expert_logits.ndim != 2 or expert_logits.shape[0] != 1:
        raise ValueError("ROUTER_R2_KPLUS1_EXPERT_LOGIT_SHAPE")
    if no_edit_logit.numel() != 1 or len(hard_candidate_mask) != expert_logits.shape[1]:
        raise ValueError("ROUTER_R2_KPLUS1_DECISION_SHAPE")
    top = int(torch.argmax(expert_logits[0]).item())
    z1 = float(expert_logits[0, top].item())
    if expert_logits.shape[1] == 1:
        z2 = EMPTY_RUNNER_UP_SENTINEL
    else:
        other = torch.cat([expert_logits[0, :top], expert_logits[0, top + 1:]])
        z2 = float(torch.max(other).item())
    null_margin = z1 - float(no_edit_logit.item())
    gap = z1 - z2
    hard_admitted = bool(hard_candidate_mask[top])
    expert_is_overall_top = null_margin >= 0.0
    accepted = (hard_admitted and expert_is_overall_top
                and null_margin >= tau_null and gap >= tau_gap)
    return {"accepted": accepted, "selected_repository_index": top,
            "hard_gate_admitted": hard_admitted, "z1": z1, "z2": z2,
            "null_logit": float(no_edit_logit.item()),
            "null_margin": null_margin, "expert_gap": gap,
            "expert_is_overall_top": expert_is_overall_top,
            "reason": (None if accepted else
                       "HARD_GATE_NO_EDIT" if not hard_admitted else
                       "NO_EDIT_TOP_CLASS" if not expert_is_overall_top else
                       "NULL_MARGIN_NO_EDIT" if null_margin < tau_null else
                       "EXPERT_GAP_NO_EDIT")}


@dataclass(frozen=True)
class R2ExecutionPlan:
    mode: str
    accepted: bool
    candidate_mask: torch.Tensor
    execution_weights: torch.Tensor
    selected_repository_index: int | None
    selected_candidate_position: int | None
    selected_original_coefficient: float | None
    s1: float | None
    s2: float | None
    gap: float | None
    candidate_count: int
    rejection_reason: str | None = None


def t1_execution_plan(plan: BaseRoutePlan | RoutePlan, expert_logits: torch.Tensor,
                      no_edit_logit: torch.Tensor, *, tau_null: float,
                      tau_gap: float) -> tuple[R2ExecutionPlan, dict[str, Any]]:
    """Build the explicit-NO_EDIT T1 single-residual execution plan.

    Expert identity is selected in the unchanged text-score space over the
    complete repository.  The unchanged visual hard gate must admit that same
    identity.  If accepted, the selected expert keeps its original R1
    coefficient from before execution pruning.
    """
    if plan.candidate_mask is None:
        raise ValueError("ROUTER_R2_CANDIDATE_MASK_MISSING")
    decision = kplus1_decision(
        expert_logits, no_edit_logit, plan.candidate_mask,
        tau_null=tau_null, tau_gap=tau_gap)
    selected = int(decision["selected_repository_index"])
    empty = torch.zeros_like(plan.candidate_mask, dtype=torch.bool)
    empty_weights = torch.empty(1, 0, device=empty.device, dtype=torch.float32)
    coefficient = None
    local = None
    if decision["hard_gate_admitted"]:
        if not isinstance(plan, RoutePlan):
            raise RuntimeError("ROUTER_R2_T1_HARD_GATE_PLAN_MISMATCH")
        candidate_positions = torch.where(plan.candidate_mask)[0]
        matches = torch.where(candidate_positions == selected)[0]
        if len(matches) != 1:
            raise RuntimeError("ROUTER_R2_T1_SELECTED_CANDIDATE_MISMATCH")
        local = int(matches[0].item())
        coefficient = float(plan.final_weights[0, local].item())
    if not decision["accepted"]:
        result = R2ExecutionPlan(
            "TOP1_ORIGINAL_COEFFICIENT", False, empty, empty_weights,
            selected, local, coefficient, float(decision["z1"]),
            float(decision["z2"]), float(decision["expert_gap"]),
            int(plan.candidate_mask.sum().item()), str(decision["reason"]),
        )
        return result, decision
    assert isinstance(plan, RoutePlan) and local is not None and coefficient is not None
    mask = empty.clone(); mask[selected] = True
    weights = plan.final_weights[:, local:local + 1].detach().clone()
    result = R2ExecutionPlan(
        "TOP1_ORIGINAL_COEFFICIENT", True, mask, weights,
        selected, local, coefficient, float(decision["z1"]),
        float(decision["z2"]), float(decision["expert_gap"]),
        int(plan.candidate_mask.sum().item()), None,
    )
    return result, decision


def _empty_mask(plan: BaseRoutePlan | RoutePlan) -> torch.Tensor:
    if plan.candidate_mask is None:
        raise ValueError("ROUTER_R2_CANDIDATE_MASK_MISSING")
    return torch.zeros_like(plan.candidate_mask, dtype=torch.bool)


def top1_route_statistics(plan: BaseRoutePlan | RoutePlan) -> dict[str, Any]:
    """Return the unchanged R1 top-1 identity and pre-pruning score statistics."""
    if isinstance(plan, BaseRoutePlan) or int(plan.candidate_mask.sum()) == 0:
        return {
            "candidate_count": 0,
            "selected_candidate_position": None,
            "selected_repository_index": None,
            "selected_original_coefficient": None,
            "s1": None,
            "s2": None,
            "gap": None,
        }
    candidate_positions = torch.where(plan.candidate_mask)[0]
    local = int(torch.argmax(plan.final_weights[0]).item())
    repository_index = int(candidate_positions[local].item())
    raw = plan.text_scores[0]
    s1 = float(raw[local].item())
    if len(raw) == 1:
        s2 = EMPTY_RUNNER_UP_SENTINEL
    else:
        runner = torch.cat([raw[:local], raw[local + 1:]])
        s2 = float(torch.max(runner).item())
    return {
        "candidate_count": int(plan.candidate_mask.sum().item()),
        "selected_candidate_position": local,
        "selected_repository_index": repository_index,
        "selected_original_coefficient": float(plan.final_weights[0, local].item()),
        "s1": s1,
        "s2": s2,
        "gap": s1 - s2,
    }


def threshold_acceptance(stats: Mapping[str, Any], tau_abs: float,
                         tau_gap: float) -> tuple[bool, str | None]:
    if int(stats["candidate_count"]) == 0:
        return False, "EMPTY_CANDIDATE_NO_EDIT"
    if float(stats["s1"]) < float(tau_abs):
        return False, "ABSOLUTE_THRESHOLD_NO_EDIT"
    if float(stats["gap"]) < float(tau_gap):
        return False, "RUNNER_UP_GAP_NO_EDIT"
    return True, None


def execution_plan(plan: BaseRoutePlan | RoutePlan, mode: str, *,
                   tau_abs: float | None = None,
                   tau_gap: float | None = None) -> R2ExecutionPlan:
    if mode not in EXECUTION_MODES:
        raise ValueError(f"ROUTER_R2_UNKNOWN_EXECUTION_MODE:{mode}")
    stats = top1_route_statistics(plan)
    empty = _empty_mask(plan)
    device = empty.device
    empty_weights = torch.empty(1, 0, device=device, dtype=torch.float32)

    if mode == "NO_EDIT":
        return R2ExecutionPlan(mode, False, empty, empty_weights, None, None, None,
                               stats["s1"], stats["s2"], stats["gap"],
                               int(stats["candidate_count"]), "EXPLICIT_NO_EDIT")
    if int(stats["candidate_count"]) == 0:
        return R2ExecutionPlan(mode, False, empty, empty_weights, None, None, None,
                               None, None, None, 0, "EMPTY_CANDIDATE_NO_EDIT")
    if mode == "R1_MULTI_RESIDUAL":
        assert isinstance(plan, RoutePlan)
        return R2ExecutionPlan(
            mode, True, plan.candidate_mask.detach().clone(), plan.final_weights.detach().clone(),
            int(stats["selected_repository_index"]), int(stats["selected_candidate_position"]),
            float(stats["selected_original_coefficient"]), float(stats["s1"]),
            float(stats["s2"]), float(stats["gap"]), int(stats["candidate_count"]), None,
        )

    accepted = True
    rejection_reason = None
    if tau_abs is not None or tau_gap is not None:
        if tau_abs is None or tau_gap is None:
            raise ValueError("ROUTER_R2_PARTIAL_THRESHOLD_CONFIGURATION")
        accepted, rejection_reason = threshold_acceptance(stats, tau_abs, tau_gap)
    if not accepted:
        return R2ExecutionPlan(
            mode, False, empty, empty_weights,
            int(stats["selected_repository_index"]), int(stats["selected_candidate_position"]),
            float(stats["selected_original_coefficient"]), float(stats["s1"]),
            float(stats["s2"]), float(stats["gap"]), int(stats["candidate_count"]),
            rejection_reason,
        )

    assert isinstance(plan, RoutePlan)
    mask = empty.clone()
    mask[int(stats["selected_repository_index"])] = True
    local = int(stats["selected_candidate_position"])
    if mode == "TOP1_ORIGINAL_COEFFICIENT":
        weights = plan.final_weights[:, local:local + 1].detach().clone()
    elif mode == "TOP1_RENORMALIZED_DIAGNOSTIC":
        weights = plan.absolute_weights[:, local:local + 1].detach().clone()
    elif mode == "TOP1_FULL_STRENGTH_DIAGNOSTIC":
        weights = torch.ones_like(plan.final_weights[:, local:local + 1])
    else:  # pragma: no cover - protected by mode validation above
        raise AssertionError(mode)
    return R2ExecutionPlan(
        mode, True, mask, weights,
        int(stats["selected_repository_index"]), local,
        float(stats["selected_original_coefficient"]), float(stats["s1"]),
        float(stats["s2"]), float(stats["gap"]), int(stats["candidate_count"]), None,
    )


def _next_above(value: float) -> float:
    return float(np.nextafter(np.float64(value), np.float64(math.inf)))


def empirical_thresholds(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = sorted({float(row[key]) for row in rows
                     if int(row["candidate_count"]) > 0 and row[key] is not None}, reverse=True)
    if not values:
        return [0.0]
    return [_next_above(values[0]), *values]


def apply_thresholds(rows: Sequence[Mapping[str, Any]], tau_abs: float,
                     tau_gap: float) -> dict[str, Any]:
    positives = [row for row in rows if row["kind"] == "positive"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    role_totals = {role: sum(row["role"] == role for row in positives) for role in ROLES}
    forced = {role: sum(row["role"] == role and bool(row["forced_success"])
                        for row in positives) for role in ROLES}
    routed = {role: 0 for role in ROLES}
    positive_rejections = 0
    target_top1 = 0
    target_hard = 0
    accepted_wrong = 0
    for row in positives:
        accept = (int(row["candidate_count"]) > 0
                  and float(row["s1"]) >= tau_abs and float(row["gap"]) >= tau_gap)
        success = bool(row["top1_success"] if accept else row["base_success"])
        routed[row["role"]] += int(success)
        positive_rejections += int(not accept)
        target_hard += int(bool(row["target_hard_recalled"]))
        target_top1 += int(bool(row["target_selected_top1"]))
        accepted_wrong += int(accept and not bool(row["target_selected_top1"]))

    exact_s0 = false_activation = clinical_failures = contaminations = 0
    no_edit = 0
    for row in negatives:
        accept = (int(row["candidate_count"]) > 0
                  and float(row["s1"]) >= tau_abs and float(row["gap"]) >= tau_gap)
        false_activation += int(accept)
        no_edit += int(not accept)
        exact_s0 += int(bool(row["top1_exact_s0"]) if accept else True)
        clinical_failures += int(accept and not bool(row["top1_clinical_passed"]))
        contaminations += int(row["top1_target_contamination_count"] if accept else 0)
        accepted_wrong += int(accept)

    floors = {
        "native": math.ceil(0.90 * forced["native"]),
        "textual": math.ceil(0.75 * forced["textual"]),
        "visual": math.ceil(0.75 * forced["visual"]),
        "paired": math.ceil(0.75 * forced["paired"]),
    }
    effectiveness_eligible = all(routed[role] >= floors[role] for role in ROLES)
    safety_eligible = clinical_failures == 0 and contaminations == 0
    rates = [routed[role] / role_totals[role] if role_totals[role] else 0.0 for role in ROLES]
    return {
        "thresholds": {"tau_abs": float(tau_abs), "tau_gap": float(tau_gap)},
        "positive_count": len(positives), "negative_count": len(negatives),
        "role_totals": role_totals, "forced_on": forced, "effectiveness_floors": floors,
        "routed": routed, "routed_total": sum(routed.values()),
        "role_macro_success": sum(rates) / len(rates),
        "positive_false_rejection": positive_rejections,
        "target_hard_recall": target_hard, "target_top1": target_top1,
        "negative_exact_s0": exact_s0, "false_activation": false_activation,
        "no_edit_count": no_edit, "clinical_canonical_failures": clinical_failures,
        "target_contamination": contaminations,
        "accepted_wrong_expert": accepted_wrong,
        "effectiveness_eligible": effectiveness_eligible,
        "safety_eligible": safety_eligible,
        "joint_eligible": effectiveness_eligible and safety_eligible,
    }


def select_dual_thresholds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit empirical dual thresholds on calibration_fit only.

    The implementation intentionally evaluates the exact empirical breakpoint
    grid.  This is deterministic calibration, not a free-form hyperparameter
    sweep.  Safety is the frozen R1 rule: zero clinical/canonical failures and
    zero target contamination.
    """
    abs_values = empirical_thresholds(rows, "s1")
    gap_values = empirical_thresholds(rows, "gap")
    candidates = [row for row in rows if int(row["candidate_count"]) > 0]
    if not candidates:
        raise RuntimeError("ROUTER_R2_STOP__FROZEN_SAFETY_POLICY_NOT_FOUND")

    # Gap thresholds are descending.  Sort candidate rows by gap descending and
    # map each empirical threshold to the inclusive prefix it accepts.  For one
    # fixed tau_abs all lexicographic metrics then come from short cumulative
    # sums instead of repeatedly rescanning every input for every threshold pair.
    order = np.argsort(-np.asarray([float(row["gap"]) for row in candidates]), kind="stable")
    ordered = [candidates[int(index)] for index in order]
    ordered_gaps = np.asarray([float(row["gap"]) for row in ordered], dtype=np.float64)
    prefix_lengths = np.asarray([
        int(np.searchsorted(-ordered_gaps, -float(value), side="right"))
        for value in gap_values
    ], dtype=np.int64)

    positives = [row for row in rows if row["kind"] == "positive"]
    role_totals = {role: sum(row["role"] == role for row in positives) for role in ROLES}
    base_success = {role: sum(row["role"] == role and bool(row["base_success"])
                              for row in positives) for role in ROLES}

    def cumulative(values: Sequence[int]) -> np.ndarray:
        raw = np.asarray(values, dtype=np.int64)
        return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(raw)])[prefix_lengths]

    best_key = None
    best_thresholds = None
    safe_count = 0
    for tau_abs in abs_values:  # descending: stricter absolute ties win
        active = [float(row["s1"]) >= tau_abs for row in ordered]
        role_success = {}
        for role in ROLES:
            delta = [
                ((int(bool(row["top1_success"])) - int(bool(row["base_success"])))
                 if flag and row["kind"] == "positive" and row["role"] == role else 0)
                for flag, row in zip(active, ordered)
            ]
            role_success[role] = base_success[role] + cumulative(delta)
        false_activation = cumulative([
            int(flag and row["kind"] == "negative") for flag, row in zip(active, ordered)
        ])
        clinical = cumulative([
            int(flag and row["kind"] == "negative" and not bool(row["top1_clinical_passed"]))
            for flag, row in zip(active, ordered)
        ])
        contamination = cumulative([
            (int(row["top1_target_contamination_count"])
             if flag and row["kind"] == "negative" else 0)
            for flag, row in zip(active, ordered)
        ])
        safe = np.flatnonzero((clinical == 0) & (contamination == 0))
        safe_count += int(len(safe))
        for gap_index in safe:
            successes = [int(role_success[role][gap_index]) for role in ROLES]
            macro = sum(success / role_totals[role]
                        for success, role in zip(successes, ROLES)) / len(ROLES)
            total = sum(successes)
            # Threshold lists are descending and loops retain the first complete
            # tie, implementing the required stricter-pair tie break.
            key = (macro, total, -int(false_activation[gap_index]),
                   -int(clinical[gap_index]), -int(contamination[gap_index]))
            if best_key is None or key > best_key:
                best_key = key
                best_thresholds = (float(tau_abs), float(gap_values[int(gap_index)]))
    if best_thresholds is None:
        raise RuntimeError("ROUTER_R2_STOP__FROZEN_SAFETY_POLICY_NOT_FOUND")
    selected = apply_thresholds(rows, *best_thresholds)
    return {
        "selected": selected,
        "candidate_grid": {"tau_abs_count": len(abs_values),
                           "tau_gap_count": len(gap_values),
                           "pair_count": len(abs_values) * len(gap_values),
                           "safety_eligible_pair_count": safe_count},
        "selection_order": [
            "frozen_safety_eligible", "role_macro_positive_success",
            "total_positive_success", "fewer_false_activations",
            "fewer_clinical_canonical_failures", "fewer_target_contaminations",
            "larger_tau_abs", "larger_tau_gap",
        ],
    }
