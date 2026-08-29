from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_e0_entrypoint_has_no_evaluation_data_argument():
    source = (ROOT / "scripts/liveedit_med/evaluate_router_r2_e0.py").read_text()
    for forbidden in ("--validation", "--heldout", "--record-953", "--blind", "--stage-2"):
        assert forbidden not in source


def test_calibration_lock_requires_frozen_threshold_artifact():
    source = (ROOT / "scripts/liveedit_med/select_router_r2_e0.py").read_text()
    assert "thresholds_frozen_before_calibration_lock_read" in source
    assert "ROUTER_R2_LOCK_READ_BEFORE_THRESHOLD_FREEZE" in source


def test_validation_is_not_a_calibration_partition():
    source = (ROOT / "scripts/liveedit_med/evaluate_router_r2_e0.py").read_text()
    assert 'choices=("calibration_fit", "calibration_lock")' in source


def test_t1_training_entrypoint_has_no_evaluation_data_argument():
    source = (ROOT / "scripts/liveedit_med/train_router_r2_t1.py").read_text()
    for forbidden in ("--validation", "--heldout", "--record-953", "--blind", "--stage-2"):
        assert forbidden not in source


def test_t1_training_uses_fit_partition_and_excludes_lock():
    source = (ROOT / "scripts/liveedit_med/train_router_r2_t1.py").read_text()
    assert 'split["calibration_fit"]' in source
    assert 'training_partition": "calibration_fit_only"' in source
    assert 'calibration_lock_loaded_for_training": False' in source


def test_t1_calibration_entrypoint_has_no_evaluation_data_argument():
    source = (ROOT / "scripts/liveedit_med/evaluate_router_r2_t1.py").read_text()
    for forbidden in ("--validation", "--heldout", "--record-953", "--blind", "--stage-2"):
        assert forbidden not in source
    assert 'choices=("calibration_fit", "calibration_lock")' in source


def test_t1_selector_is_calibration_only():
    source = (ROOT / "scripts/liveedit_med/select_router_r2_t1.py").read_text()
    for forbidden in ("--validation", "--heldout", "--record-953", "--blind", "--stage-2"):
        assert forbidden not in source
    assert "ROUTER_R2_T1_NO_ELIGIBLE_CALIBRATION_CHECKPOINT" in source


def test_preregistered_split_has_zero_internal_overlap():
    # The runtime manifest is independently audited on the server.  This test
    # fixes the local schema-level invariant without embedding any sample ID.
    prep = (ROOT / "scripts/liveedit_med/prepare_router_r2_minimal_causal_rescue.py").read_text()
    assert "set(fit_ids) & set(lock_ids)" in prep
    assert "ROUTER_R2_CALIBRATION_SPLIT_INVALID" in prep


def test_frozen_hash_constants_are_source_pinned():
    source = (ROOT / "scripts/liveedit_med/prepare_router_r2_minimal_causal_rescue.py").read_text()
    assert "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db" in source
    assert "d1d5ce232ad2aeb7a29c4c5586a6af5dcdee064321cfc94c3393d576b1bc2249" in source
    assert "d8b7032a563e32f22fd51eb65d92bbb0177c913d19c5c1e6ce6ad73d0e5ca75d" in source
