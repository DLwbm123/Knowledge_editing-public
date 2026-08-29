from __future__ import annotations

from methods.liveedit_med.top1_execution_audit import capture_ratio, select_checkpoint, validation_eligible


def row(step, *, hard=152, visual=14, paired=14, native=18, textual=13, weight=0.2):
    return {
        "step": step,
        "generation_parity_failures": 0,
        "repository_sizes": {"32": {
            "routed": {"native": native, "textual": textual, "visual": visual, "paired": paired},
            "locality_exact_preservation": 64,
            "target_contaminations": 0,
            "clinical_canonical_failures": 0,
            "target_expert_top1_count": 50,
        }},
        "safety": {
            "hard_negative_exact_s0": hard,
            "target_contaminations": 0,
            "clinical_canonical_failures": 0,
            "mean_selected_final_weight": weight,
            "mean_selected_residual_norm": 1.0,
        },
    }


def test_validation_floor_is_frozen_and_conjunctive():
    assert validation_eligible(row(80))
    assert not validation_eligible(row(80, visual=13))
    assert not validation_eligible(row(80, hard=151))


def test_lexicographic_selection_and_earlier_tie_break():
    rows = [row(step) for step in (80, 160, 240, 320, 400, 480, 560, 640)]
    rows[-1]["safety"]["hard_negative_exact_s0"] = 153
    assert select_checkpoint(rows)["step"] == 640
    rows[-1]["safety"]["hard_negative_exact_s0"] = 152
    assert select_checkpoint(rows)["step"] == 80


def test_no_eligible_checkpoint_returns_none():
    rows = [row(step, hard=151) for step in (80, 160, 240, 320, 400, 480, 560, 640)]
    assert select_checkpoint(rows) is None


def test_capture_ratio_handles_zero_denominator():
    assert capture_ratio(10, 10, 11)["oracle_capture_ratio"] is None
    assert capture_ratio(10, 14, 12)["oracle_capture_ratio"] == 0.5

