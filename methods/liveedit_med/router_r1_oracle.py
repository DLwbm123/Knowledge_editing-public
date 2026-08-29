"""Pure contracts for validation-only Router-R1 oracle diagnosis.

This module contains no model, data, held-out, record-953, or blind-set
loading.  The functions define the immutable O0--O4 interventions and the
pre-registered mechanism attribution calculations.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROTOCOL = "LIVEEDIT_MED_ROUTER_R1_VALIDATION_ORACLE_DIAGNOSIS_V1"
ORACLES = ("O0", "O1", "O2", "O3", "O4")
ROLES = ("native", "textual", "visual", "paired")


def zero_safe_cosine(left: torch.Tensor, right: torch.Tensor) -> tuple[float, bool]:
    """Return a finite cosine and whether either input was norm-degenerate."""
    x, y = left.detach().float().reshape(-1), right.detach().float().reshape(-1)
    xn, yn = x.norm(), y.norm()
    degenerate = bool(xn.item() == 0.0 or yn.item() == 0.0)
    if degenerate:
        return 0.0, True
    return float(torch.dot(x, y).div(xn * yn).item()), False


def tensor_norm_rms(value: torch.Tensor) -> dict[str, float]:
    tensor = value.detach().float()
    return {"norm": float(tensor.norm().item()),
            "rms": float(tensor.square().mean().sqrt().item())}


def ensure_target_candidate(mask: torch.Tensor, target_index: int) -> torch.Tensor:
    """O1 candidate intervention: preserve candidates and insert target only."""
    result = mask.detach().clone().bool()
    if result.ndim != 1 or not 0 <= target_index < len(result):
        raise ValueError("ROUTER_R1_ORACLE_INVALID_TARGET_INDEX")
    result[target_index] = True
    return result


def original_text_weights(raw_scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Exact frozen sigmoid x softmax equation, without post-normalization."""
    if raw_scores.ndim != 2 or raw_scores.shape[0] != 1 or raw_scores.shape[1] < 1:
        raise ValueError("ROUTER_R1_ORACLE_INVALID_TEXT_SCORE_SHAPE")
    relative = torch.softmax(raw_scores, dim=1)
    absolute = torch.sigmoid(raw_scores)
    return {"raw": raw_scores, "sigmoid": absolute, "softmax": relative,
            "final": absolute * relative}


def oracle_coefficients(target_final_o1: torch.Tensor,
                        target_sigmoid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return O2--O4 target-only execution coefficients."""
    final = target_final_o1.reshape(1, 1)
    sigmoid = target_sigmoid.reshape(1, 1)
    one = torch.ones_like(final)
    return {"O2": final, "O3": sigmoid, "O4": one}


def fhtr_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the pre-registered F/H/T/R cross-table and conditional rates."""
    counts = Counter()
    for row in rows:
        f, h, t, r = (bool(row[name]) for name in ("F", "H", "T", "R"))
        counts["total"] += 1
        counts["F"] += f; counts["H"] += h; counts["T"] += t; counts["R"] += r
        counts["F_H"] += f and h
        counts["F_H_T"] += f and h and t
        counts["F_H_R"] += f and h and r
        counts["F_H_T_R"] += f and h and t and r
        counts["F_R"] += f and r

    def ratio(numerator: str, denominator: str) -> float | None:
        return (counts[numerator] / counts[denominator]) if counts[denominator] else None

    return {
        "counts": {name: int(counts[name]) for name in (
            "total", "F", "H", "T", "R", "F_H", "F_H_T", "F_H_R", "F_H_T_R")},
        "conditional_retention": {
            "P_R_given_F": ratio("F_R", "F"),
            "P_R_given_F_H": ratio("F_H_R", "F_H"),
            "P_R_given_F_H_T": ratio("F_H_T_R", "F_H_T"),
        },
    }


def sequential_success_gains(successes: Mapping[str, int]) -> dict[str, int]:
    if tuple(successes) != ORACLES:
        raise ValueError("ROUTER_R1_ORACLE_ORDER_MISMATCH")
    return {
        "gain_hard_recall": int(successes["O1"] - successes["O0"]),
        "gain_remove_distractors": int(successes["O2"] - successes["O1"]),
        "gain_remove_softmax": int(successes["O3"] - successes["O2"]),
        "gain_full_strength": int(successes["O4"] - successes["O3"]),
    }


def mechanism_label(successes: Mapping[str, int], *, o4_parity: bool,
                    forced_on_is_dominant_limit: bool = False) -> str:
    if not o4_parity:
        return "ORACLE_DIAGNOSIS_INVALID_ENGINEERING_RUN"
    if forced_on_is_dominant_limit:
        return "FROZEN_EXPERT_UPPER_BOUND_LIMIT"
    gains = sequential_success_gains(successes)
    names = {
        "gain_hard_recall": "PRIMARY_VISUAL_HARD_RECALL_BOTTLENECK",
        "gain_remove_distractors": "PRIMARY_DISTRACTOR_RESIDUAL_INTERFERENCE",
        "gain_remove_softmax": "PRIMARY_RELATIVE_SOFTMAX_DILUTION",
        "gain_full_strength": "PRIMARY_SIGMOID_UNDERSCALING",
    }
    total = int(successes["O4"] - successes["O0"])
    if total > 0:
        for key, value in gains.items():
            if value >= 4 and value / total >= 0.5:
                return names[key]
    if sum(value >= 3 for value in gains.values()) >= 2:
        return "MIXED_POST_RECALL_ROUTING_BOTTLENECK"
    # The fixed rule has no single-mechanism winner; use the mixed post-recall
    # label instead of inventing a new threshold after seeing validation.
    return "MIXED_POST_RECALL_ROUTING_BOTTLENECK"


def bootstrap_gain_intervals(family_rows: Sequence[Mapping[str, Any]], *,
                             replicates: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Family bootstrap CIs for sequential success-count gains (descriptive)."""
    if not family_rows:
        raise ValueError("ROUTER_R1_ORACLE_EMPTY_BOOTSTRAP")
    matrix = np.asarray([[int(row[name]) for name in ORACLES] for row in family_rows], dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in (
        "gain_hard_recall", "gain_remove_distractors", "gain_remove_softmax", "gain_full_strength")}
    for _ in range(replicates):
        sample = matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0)
        gains = sequential_success_gains(dict(zip(ORACLES, sample.tolist())))
        for name, value in gains.items():
            values[name].append(value)
    return {name: {"lower_2p5": float(np.percentile(rows, 2.5)),
                   "median": float(np.percentile(rows, 50)),
                   "upper_97p5": float(np.percentile(rows, 97.5))}
            for name, rows in values.items()}
