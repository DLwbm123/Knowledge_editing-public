from __future__ import annotations

import torch

from methods.liveedit_med.expert_execution_policy import (
    POLICY_NAME,
    SparseTop1Plan,
    apply_sparse_top1_residual,
    sparse_top1_audit,
    sparse_top1_plan,
)
from methods.liveedit_med.source_ops import BaseRoutePlan, RoutePlan, apply_low_rank_expert_residual


def route(mask, weights):
    count = sum(mask)
    raw = torch.arange(count, dtype=torch.float32).reshape(1, count)
    return RoutePlan(
        candidate_mask=torch.tensor(mask),
        visual_scores=torch.zeros(1, len(mask)),
        sentinel_score=torch.zeros(1, 1),
        text_scores=raw,
        relative_weights=torch.softmax(raw, 1),
        absolute_weights=torch.sigmoid(raw),
        final_weights=torch.tensor([weights], dtype=torch.float32),
    )


def test_empty_candidates_preserve_exact_base_plan():
    base = BaseRoutePlan(candidate_mask=torch.tensor([False, False]))
    assert sparse_top1_plan(base, ["a", "b"]) is base
    assert sparse_top1_audit(base)["executed_expert_count"] == 0


def test_multiple_candidates_choose_max_source_final_weight():
    selected = sparse_top1_plan(route([True, False, True], [0.2, 0.7]), ["a", "b", "c"])
    assert isinstance(selected, SparseTop1Plan)
    assert selected.selected_expert_id == "c"
    assert selected.selected_repository_index == 2
    assert selected.selected_candidate_position == 1
    assert selected.selected_final_weight.item() == torch.tensor(0.7).item()


def test_tie_break_prefers_lower_stable_expert_id():
    selected = sparse_top1_plan(route([True, True], [0.5, 0.5]), ["z", "a"])
    assert isinstance(selected, SparseTop1Plan)
    assert selected.selected_expert_id == "a"


def test_selected_weight_is_bit_exact_view_and_not_renormalized():
    source = route([True, True], [0.125, 0.375])
    selected = sparse_top1_plan(source, ["a", "b"])
    assert isinstance(selected, SparseTop1Plan)
    assert torch.equal(selected.selected_final_weight, source.final_weights[:, 1:2])
    audit = sparse_top1_audit(selected)
    assert audit["policy_name"] == POLICY_NAME
    assert audit["weight_preserved_from_source_plan"]
    assert not audit["renormalized"]


def test_one_candidate_matches_original_fusion_exactly():
    torch.manual_seed(1)
    hidden = torch.randn(1, 3, 4)
    cs, rs = torch.randn(1, 2, 4), torch.randn(1, 2, 4)
    norm = torch.nn.LayerNorm(4)
    source = route([True], [0.4])
    selected = sparse_top1_plan(source, ["a"])
    actual = apply_sparse_top1_residual(hidden, selected, cs, rs, norm)
    expected = apply_low_rank_expert_residual(hidden.float(), cs, rs, source.final_weights, norm).to(hidden.dtype)
    assert torch.equal(actual, expected)


def test_distractor_has_no_residual_contribution():
    hidden = torch.ones(1, 2, 4)
    cs = torch.stack([torch.ones(2, 4), torch.full((2, 4), 999.0)])
    rs = torch.stack([torch.ones(2, 4), torch.full((2, 4), 999.0)])
    norm = torch.nn.LayerNorm(4)
    source = route([True, True], [0.8, 0.2])
    selected = sparse_top1_plan(source, ["a", "b"])
    actual = apply_sparse_top1_residual(hidden, selected, cs, rs, norm)
    expected = apply_low_rank_expert_residual(hidden.float(), cs[:1], rs[:1], source.final_weights[:, :1], norm)
    assert torch.equal(actual, expected)


def test_policy_api_has_no_target_identity_argument():
    assert "target" not in sparse_top1_plan.__annotations__

