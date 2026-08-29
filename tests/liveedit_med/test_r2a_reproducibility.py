import torch

from methods.liveedit_med.expert_execution_policy import sparse_top1_plan
from methods.liveedit_med.source_ops import RoutePlan


def test_top1_selection_replays_deterministically():
    plan = RoutePlan(
        candidate_mask=torch.tensor([True, True, False]),
        visual_scores=torch.zeros(1, 3),
        sentinel_score=torch.zeros(1, 1),
        text_scores=torch.zeros(1, 2),
        relative_weights=torch.tensor([[0.5, 0.5]]),
        absolute_weights=torch.tensor([[0.5, 0.5]]),
        final_weights=torch.tensor([[0.25, 0.25]]),
    )
    first = sparse_top1_plan(plan, ["b", "a", "c"])
    second = sparse_top1_plan(plan, ["b", "a", "c"])
    assert first.selected_expert_id == second.selected_expert_id == "a"
    assert torch.equal(first.selected_final_weight, second.selected_final_weight)
