from __future__ import annotations

import torch
from pathlib import Path

from methods.liveedit_med.router_r1_oracle import (
    ORACLES,
    bootstrap_gain_intervals,
    ensure_target_candidate,
    fhtr_counts,
    mechanism_label,
    oracle_coefficients,
    original_text_weights,
    sequential_success_gains,
    tensor_norm_rms,
    zero_safe_cosine,
)


ROOT = Path(__file__).resolve().parents[2]


def test_o1_inserts_only_target_and_preserves_candidates():
    source = torch.tensor([False, True, False, True])
    result = ensure_target_candidate(source, 2)
    assert result.tolist() == [False, True, True, True]
    assert source.tolist() == [False, True, False, True]


def test_frozen_text_equation_has_no_post_normalization():
    raw = torch.tensor([[0.2, -0.4, 1.1]])
    values = original_text_weights(raw)
    assert torch.equal(values["final"], values["sigmoid"] * values["softmax"])
    assert not torch.isclose(values["final"].sum(), torch.tensor(1.0))


def test_o2_o3_o4_coefficients_are_exact():
    values = oracle_coefficients(torch.tensor(0.25), torch.tensor(0.7))
    assert values["O2"].item() == 0.25
    assert torch.isclose(values["O3"], torch.tensor([[0.7]])).all()
    assert values["O4"].item() == 1.0


def test_cross_table_uses_forced_on_conditioning():
    rows = [
        {"F": True, "H": True, "T": True, "R": True},
        {"F": True, "H": True, "T": False, "R": False},
        {"F": True, "H": False, "T": False, "R": False},
        {"F": False, "H": True, "T": True, "R": True},
    ]
    result = fhtr_counts(rows)
    assert result["counts"] == {"total": 4, "F": 3, "H": 3, "T": 2, "R": 2,
                                 "F_H": 2, "F_H_T": 1, "F_H_R": 1, "F_H_T_R": 1}
    assert result["conditional_retention"]["P_R_given_F"] == 1 / 3
    assert result["conditional_retention"]["P_R_given_F_H"] == 1 / 2
    assert result["conditional_retention"]["P_R_given_F_H_T"] == 1.0


def test_zero_safe_cosine_and_rms():
    cosine, degenerate = zero_safe_cosine(torch.zeros(3), torch.ones(3))
    assert cosine == 0.0 and degenerate
    cosine, degenerate = zero_safe_cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    assert cosine == 1.0 and not degenerate
    assert tensor_norm_rms(torch.tensor([3.0, 4.0]))["norm"] == 5.0


def test_sequential_gains_and_primary_rule():
    values = dict(zip(ORACLES, (10, 12, 18, 19, 20)))
    assert sequential_success_gains(values)["gain_remove_distractors"] == 6
    assert mechanism_label(values, o4_parity=True) == "PRIMARY_DISTRACTOR_RESIDUAL_INTERFERENCE"
    assert mechanism_label(values, o4_parity=False) == "ORACLE_DIAGNOSIS_INVALID_ENGINEERING_RUN"


def test_bootstrap_is_deterministic():
    rows = [{name: (index + offset) % 2 for offset, name in enumerate(ORACLES)}
            for index in range(8)]
    first = bootstrap_gain_intervals(rows, replicates=100, seed=42)
    second = bootstrap_gain_intervals(rows, replicates=100, seed=42)
    assert first == second


def test_oracle_worker_has_frozen_o0_and_o4_hard_stops():
    text = (ROOT / "scripts/liveedit_med/evaluate_router_r1_oracle_ladder.py").read_text()
    assert "ORACLE_BASELINE_ROUTED_PARITY_FAILURE" in text
    assert "ORACLE_TARGET_ONLY_FORCED_ON_PARITY_FAILURE" in text
    assert '"O4": one' in (ROOT / "methods/liveedit_med/router_r1_oracle.py").read_text()


def test_oracle_does_not_add_target_to_prompt_or_enter_selection():
    text = (ROOT / "scripts/liveedit_med/evaluate_router_r1_oracle_ladder.py").read_text()
    assert '"target_added_to_prompt": False' in text
    assert '"not_for_selection": True' in text
    assert "select_eqkey_clean" not in text


def test_oracle_scope_has_no_heldout_record953_or_blind_loader():
    text = (ROOT / "scripts/liveedit_med/evaluate_router_r1_oracle_ladder.py").read_text()
    assert "purged_heldout" not in text
    assert "run_router_r1_record953" not in text
    assert "blind_set_builder" not in text


def test_safety_closure_requires_identical_160_inputs_and_null_selection():
    text = (ROOT / "scripts/liveedit_med/finalize_router_r1_safety_closure.py").read_text()
    assert "identities != safety_inputs or len(identities) != 160" in text
    assert '"selected_checkpoint": None' in text
    assert '"safety_cannot_override_effectiveness_eligibility": True' in text


def test_source_validation_loss_is_marked_invariant_only():
    text = (ROOT / "scripts/liveedit_med/finalize_router_r1_safety_closure.py").read_text()
    assert '"router_sensitive": False' in text
    assert '"invariant_diagnostic_only": True' in text


def test_cross_table_uses_all_four_booleans_and_nonoverwriting_output():
    text = (ROOT / "scripts/liveedit_med/build_router_r1_oracle_cross_table.py").read_text()
    assert '"F": bool(' in text and '"H": hard' in text
    assert '"T": top' in text and '"R": bool(' in text
    assert 'with path.open("x")' in text


def test_gradient_diagnostic_has_no_optimizer_and_resets_gradients():
    text = (ROOT / "scripts/liveedit_med/run_router_r1_gradient_feature_diagnostics.py").read_text()
    assert "torch.optim" not in text
    assert "optimizer.step" not in text
    assert "modules.zero_grad(set_to_none=True)" in text
    assert "ROUTER_R1_GRADIENT_NOT_RESET" in text


def test_gradient_and_feature_diagnostic_are_train_only_and_hash_guarded():
    text = (ROOT / "scripts/liveedit_med/run_router_r1_gradient_feature_diagnostics.py").read_text()
    assert '"train_only":True' in text
    assert '"validation_used":False' in text
    assert "before!=after" in text
    assert "ROUTER_R1_GRADIENT_BASE_OR_BANK_MUTATION" in text


def test_feature_score_decomposition_is_explicit():
    text = (ROOT / "scripts/liveedit_med/run_router_r1_gradient_feature_diagnostics.py").read_text()
    assert '"reconstructed_dot"' in text and '"absolute_error"' in text
    assert "routing_features_normalized_or_temperature_changed" in text


def test_finalizer_keeps_all_permissions_closed_and_does_not_implement_r2():
    text = (ROOT / "scripts/liveedit_med/finalize_router_r1_oracle_diagnosis.py").read_text()
    for token in ('"router_r2_implemented":False', '"heldout_permitted":False',
                  '"record953_permitted":False', '"sealed_blind_permitted":False',
                  '"stage2_permitted":False'):
        assert token in text
