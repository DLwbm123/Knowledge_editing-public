from __future__ import annotations

import torch

from methods.liveedit_med.router_r2 import (
    EMPTY_RUNNER_UP_SENTINEL,
    apply_thresholds,
    execution_plan,
    kplus1_cross_entropy,
    kplus1_decision,
    kplus1_label,
    kplus1_text_logits,
    select_dual_thresholds,
    t1_execution_plan,
)
from methods.liveedit_med.source_ops import BaseRoutePlan, RoutePlan


def route() -> RoutePlan:
    mask = torch.tensor([True, False, True])
    raw = torch.tensor([[1.0, 2.0]])
    relative = torch.softmax(raw, dim=1)
    absolute = torch.sigmoid(raw)
    return RoutePlan(mask, torch.tensor([[2.0, 0.0, 3.0]]), torch.tensor([[1.0]]),
                     raw, relative, absolute, relative * absolute)


def test_top1_original_coefficient_prunes_without_renormalizing():
    original = route()
    result = execution_plan(original, "TOP1_ORIGINAL_COEFFICIENT")
    assert result.selected_repository_index == 2
    assert result.candidate_mask.tolist() == [False, False, True]
    assert torch.equal(result.execution_weights, original.final_weights[:, 1:2])
    assert result.selected_original_coefficient == float(original.final_weights[0, 1])


def test_r1_multi_and_diagnostic_modes_are_distinct():
    original = route()
    multi = execution_plan(original, "R1_MULTI_RESIDUAL")
    kept = execution_plan(original, "TOP1_ORIGINAL_COEFFICIENT")
    renorm = execution_plan(original, "TOP1_RENORMALIZED_DIAGNOSTIC")
    full = execution_plan(original, "TOP1_FULL_STRENGTH_DIAGNOSTIC")
    assert torch.equal(multi.candidate_mask, original.candidate_mask)
    assert torch.equal(multi.execution_weights, original.final_weights)
    assert not torch.equal(kept.execution_weights, renorm.execution_weights)
    assert torch.equal(renorm.execution_weights, original.absolute_weights[:, 1:2])
    assert full.execution_weights.item() == 1.0


def test_no_edit_and_empty_candidate_execute_no_residual():
    original = route()
    explicit = execution_plan(original, "NO_EDIT")
    empty = BaseRoutePlan(candidate_mask=torch.tensor([False, False, False]))
    implicit = execution_plan(empty, "TOP1_ORIGINAL_COEFFICIENT")
    for value in (explicit, implicit):
        assert not value.accepted
        assert not value.candidate_mask.any()
        assert value.execution_weights.numel() == 0


def test_single_candidate_uses_documented_runner_up_sentinel():
    original = route()
    single = RoutePlan(torch.tensor([False, True, False]), original.visual_scores,
                       original.sentinel_score, torch.tensor([[1.5]]),
                       torch.ones(1, 1), torch.sigmoid(torch.tensor([[1.5]])),
                       torch.sigmoid(torch.tensor([[1.5]])))
    result = execution_plan(single, "TOP1_ORIGINAL_COEFFICIENT")
    assert result.s2 == EMPTY_RUNNER_UP_SENTINEL
    assert result.gap == result.s1 - EMPTY_RUNNER_UP_SENTINEL


def test_dual_threshold_rejection_installs_no_execution():
    result = execution_plan(route(), "TOP1_ORIGINAL_COEFFICIENT", tau_abs=3.0, tau_gap=0.0)
    assert not result.accepted
    assert result.rejection_reason == "ABSOLUTE_THRESHOLD_NO_EDIT"
    assert result.execution_weights.numel() == 0


def calibration_rows():
    rows = []
    for role in ("native", "textual", "visual", "paired"):
        rows.append({"kind": "positive", "role": role, "candidate_count": 1,
                     "s1": 2.0, "gap": 2.0, "forced_success": True,
                     "top1_success": True, "base_success": False,
                     "target_hard_recalled": True, "target_selected_top1": True})
    rows.append({"kind": "negative", "role": None, "candidate_count": 1,
                 "s1": 0.0, "gap": 0.1, "top1_exact_s0": False,
                 "top1_clinical_passed": False, "top1_target_contamination_count": 1})
    return rows


def test_threshold_fit_uses_frozen_safety_then_effectiveness_order():
    result = select_dual_thresholds(calibration_rows())
    selected = result["selected"]
    assert selected["safety_eligible"]
    assert selected["routed_total"] == 4
    assert selected["false_activation"] == 0
    assert selected["thresholds"]["tau_abs"] > 0.0


def test_apply_thresholds_no_edit_negative_is_exact_base():
    metrics = apply_thresholds(calibration_rows(), tau_abs=1.0, tau_gap=1.0)
    assert metrics["negative_exact_s0"] == 1
    assert metrics["clinical_canonical_failures"] == 0
    assert metrics["target_contamination"] == 0


def test_kplus1_positive_and_negative_label_mapping():
    assert kplus1_label(repository_size=32, target_repository_index=0, no_edit=False) == 0
    assert kplus1_label(repository_size=32, target_repository_index=None, no_edit=True) == 32


def test_kplus1_negative_no_edit_training_path():
    input_key = torch.ones(1, 4, 2)
    experts = torch.stack([torch.ones(4, 2), -torch.ones(4, 2)])
    null = torch.full((1, 4, 2), 2.0, requires_grad=True)
    logits = kplus1_text_logits(input_key, experts, null)
    loss = kplus1_cross_entropy(logits, kplus1_label(
        repository_size=2, target_repository_index=None, no_edit=True))
    loss.backward()
    assert torch.isfinite(loss)
    assert null.grad is not None
    assert int(torch.argmax(logits, dim=1).item()) == 2


def test_kplus1_positive_target_expert_training_path():
    logits = torch.tensor([[3.0, 0.0, -1.0]], requires_grad=True)
    loss = kplus1_cross_entropy(logits, kplus1_label(
        repository_size=2, target_repository_index=0, no_edit=False))
    loss.backward()
    assert logits.grad is not None and logits.grad[0, 0] < 0


def test_kplus1_decision_keeps_unchanged_hard_gate():
    experts = torch.tensor([[3.0, 1.0]])
    accepted = kplus1_decision(experts, torch.tensor([[0.0]]),
                               torch.tensor([True, False]), tau_null=1.0, tau_gap=1.0)
    rejected = kplus1_decision(experts, torch.tensor([[0.0]]),
                               torch.tensor([False, True]), tau_null=1.0, tau_gap=1.0)
    assert accepted["accepted"]
    assert not rejected["accepted"] and rejected["reason"] == "HARD_GATE_NO_EDIT"


def test_kplus1_no_edit_must_not_be_overall_top_class():
    decision = kplus1_decision(torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0]]),
                               torch.tensor([True, False]), tau_null=-10.0, tau_gap=0.0)
    assert not decision["accepted"]
    assert not decision["expert_is_overall_top"]
    assert decision["reason"] == "NO_EDIT_TOP_CLASS"


def test_t1_execution_preserves_selected_original_coefficient():
    original = route()
    plan, decision = t1_execution_plan(
        original, torch.tensor([[0.0, 1.0, 3.0]]), torch.tensor([[-1.0]]),
        tau_null=0.0, tau_gap=0.0)
    assert decision["accepted"] and plan.accepted
    assert plan.selected_repository_index == 2
    assert plan.candidate_mask.tolist() == [False, False, True]
    assert torch.equal(plan.execution_weights, original.final_weights[:, 1:2])
    assert plan.selected_original_coefficient == float(original.final_weights[0, 1])


def test_t1_rejects_overall_top_expert_outside_unchanged_hard_gate():
    original = route()
    plan, decision = t1_execution_plan(
        original, torch.tensor([[0.0, 4.0, 3.0]]), torch.tensor([[-1.0]]),
        tau_null=0.0, tau_gap=0.0)
    assert not decision["hard_gate_admitted"]
    assert not plan.accepted and plan.execution_weights.numel() == 0
