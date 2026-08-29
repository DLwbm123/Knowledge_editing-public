"""Frozen execution policies layered on top of an unchanged LiveEdit route.

The R2A policy in this module never computes candidates or routing scores.  It
only chooses one member of an existing :class:`RoutePlan` and preserves that
member's original final coefficient exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .source_ops import BaseRoutePlan, RoutePlan, apply_low_rank_expert_residual


PROTOCOL = "LIVEEDIT_MED_ROUTER_R2A_SPARSE_TOP1_ORIGINAL_GAIN_V1"
POLICY_NAME = "SPARSE_TOP1_ORIGINAL_GAIN"
POLICY_VERSION = 1


@dataclass(frozen=True)
class SparseTop1Plan:
    """Execution-only view of an unchanged source ``RoutePlan``."""

    source_plan: RoutePlan
    selected_candidate_position: int
    selected_repository_index: int
    selected_expert_id: str
    selected_final_weight: torch.Tensor
    execution_mask: torch.Tensor


def _candidate_repository_indices(plan: RoutePlan) -> list[int]:
    return [index for index, selected in enumerate(plan.candidate_mask.tolist()) if selected]


def sparse_top1_plan(
    plan: BaseRoutePlan | RoutePlan,
    expert_ids: Sequence[str],
) -> BaseRoutePlan | SparseTop1Plan:
    """Select max original final weight, breaking exact ties by stable ID.

    ``expert_ids`` are repository identities only.  No target identity is
    accepted by this API, preventing target-aware deployable selection.
    """

    if isinstance(plan, BaseRoutePlan):
        return plan
    repository_indices = _candidate_repository_indices(plan)
    if len(repository_indices) != int(plan.final_weights.shape[1]):
        raise RuntimeError("R2A_TOP1_INVALID_ROUTE_SHAPE")
    if len(expert_ids) != int(plan.candidate_mask.numel()):
        raise RuntimeError("R2A_TOP1_EXPERT_ID_COUNT_MISMATCH")
    if not repository_indices:
        raise RuntimeError("R2A_TOP1_EMPTY_ROUTED_PLAN")

    candidate_rows = [
        (position, repository_index, str(expert_ids[repository_index]))
        for position, repository_index in enumerate(repository_indices)
    ]
    selected_position, selected_repository_index, selected_id = min(
        candidate_rows,
        key=lambda row: (-float(plan.final_weights[0, row[0]].detach().cpu().item()), row[2]),
    )
    execution_mask = torch.zeros_like(plan.candidate_mask, dtype=torch.bool)
    execution_mask[selected_repository_index] = True
    selected_weight = plan.final_weights[:, selected_position:selected_position + 1]
    return SparseTop1Plan(
        source_plan=plan,
        selected_candidate_position=selected_position,
        selected_repository_index=selected_repository_index,
        selected_expert_id=selected_id,
        selected_final_weight=selected_weight,
        execution_mask=execution_mask,
    )


def apply_sparse_top1_residual(
    hidden: torch.Tensor,
    execution: BaseRoutePlan | SparseTop1Plan,
    moe_cs: torch.Tensor,
    moe_rs: torch.Tensor,
    instant_norm: torch.nn.Module,
) -> torch.Tensor:
    """Apply exactly one expert with its source-plan coefficient."""

    if isinstance(execution, BaseRoutePlan):
        return torch.zeros_like(hidden)
    index = execution.selected_repository_index
    return apply_low_rank_expert_residual(
        hidden.float(),
        moe_cs[index:index + 1],
        moe_rs[index:index + 1],
        execution.selected_final_weight,
        instant_norm,
    ).to(hidden.dtype)


def sparse_top1_audit(execution: BaseRoutePlan | SparseTop1Plan) -> dict[str, Any]:
    if isinstance(execution, BaseRoutePlan):
        return {
            "policy_name": POLICY_NAME,
            "kind": "base",
            "reason": execution.reason,
            "selected_expert_id": None,
            "selected_repository_index": None,
            "selected_candidate_position": None,
            "selected_final_weight": None,
            "executed_expert_count": 0,
        }
    return {
        "policy_name": POLICY_NAME,
        "kind": "routed",
        "selected_expert_id": execution.selected_expert_id,
        "selected_repository_index": execution.selected_repository_index,
        "selected_candidate_position": execution.selected_candidate_position,
        "selected_final_weight": float(execution.selected_final_weight.detach().cpu().item()),
        "executed_expert_count": 1,
        "weight_preserved_from_source_plan": bool(torch.equal(
            execution.selected_final_weight,
            execution.source_plan.final_weights[
                :, execution.selected_candidate_position:execution.selected_candidate_position + 1
            ],
        )),
        "renormalized": False,
    }

