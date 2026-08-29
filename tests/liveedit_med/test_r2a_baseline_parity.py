from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_r1_source_path_is_not_modified_by_policy_module():
    source = (ROOT / "methods/liveedit_med/source_ops.py").read_text()
    assert "SPARSE_TOP1_ORIGINAL_GAIN" not in source
    assert "torch.einsum(\"lmr,mrd,m->ld\", value, moe_rs, weights[0])" in source


def test_policy_does_not_compute_or_change_routing_scores():
    source = (ROOT / "methods/liveedit_med/expert_execution_policy.py").read_text()
    for forbidden in ("extract_vision", "extract_query", "torch.sigmoid", "torch.softmax", "sentinel_score >"):
        assert forbidden not in source


def test_policy_contains_no_optimizer_or_training_step():
    source = (ROOT / "methods/liveedit_med/expert_execution_policy.py").read_text()
    assert "torch.optim" not in source
    assert "optimizer.step" not in source

