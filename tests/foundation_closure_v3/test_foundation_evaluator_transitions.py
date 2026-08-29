import pytest

from m3bench_foundation.evaluator import evaluate_tasks, harmonic_mean


JUDGE = "m3bench-gpt56sol-v1"


def _edit(task, edit_id, probe_flags):
    return {
        "task": task,
        "edit_id": edit_id,
        "anchor": {"record_id_or_derived_probe_id": edit_id + ":anchor", "pre_is_correct": False, "judge_version": JUDGE},
        "probes": [
            {"record_id_or_derived_probe_id": f"{edit_id}:p{index}", "pre_is_correct": flag, "judge_version": JUDGE}
            for index, flag in enumerate(probe_flags)
        ],
    }


def _post(edit, anchor=True, probes=()):
    rows = [{"record_id_or_derived_probe_id": edit["anchor"]["record_id_or_derived_probe_id"], "is_correct": anchor}]
    rows.extend(
        {"record_id_or_derived_probe_id": probe["record_id_or_derived_probe_id"], "is_correct": value}
        for probe, value in zip(edit["probes"], probes)
    )
    return rows


def test_t0_wrong_to_correct():
    edit = _edit("T0", "t0", [])
    assert evaluate_tasks([edit], _post(edit))["tasks"]["T0"]["score"] == 1.0


@pytest.mark.parametrize("task", ["T1L", "T2L", "T3L", "T4L"])
def test_locality_counts_only_originally_correct(task):
    edit = _edit(task, task, [True, False])
    result = evaluate_tasks([edit], _post(edit, probes=[False, True]))["tasks"][task]
    assert result["eligible_probes"] == 1
    assert result["score"] == 0.0


@pytest.mark.parametrize("task", ["T1G", "T2G", "T3G", "T4G"])
def test_generality_counts_only_originally_wrong(task):
    edit = _edit(task, task, [False, True])
    result = evaluate_tasks([edit], _post(edit, probes=[True, False]))["tasks"][task]
    assert result["eligible_probes"] == 1
    assert result["score"] == 1.0


def test_excluded_na_fails_closed_before_denominator():
    edit = _edit("T1L", "excluded", [None])
    with pytest.raises(ValueError, match="unjudged pre/post probe"):
        evaluate_tasks([edit], _post(edit, probes=[True]))


def test_empty_denominator_is_null():
    edit = _edit("T1L", "null", [False])
    assert evaluate_tasks([edit], _post(edit, probes=[True]))["tasks"]["T1L"]["score"] is None


def test_metric_range_and_harmonic_zero():
    edit = _edit("T4G", "range", [False, False])
    score = evaluate_tasks([edit], _post(edit, probes=[True, False]))["tasks"]["T4G"]["score"]
    assert 0.0 <= score <= 1.0
    assert harmonic_mean([1.0, 0.0]) == 0.0


def test_duplicate_prediction_fails_closed():
    edit = _edit("T0", "duplicate", [])
    post = _post(edit)
    with pytest.raises(ValueError, match="duplicate post-edit prediction"):
        evaluate_tasks([edit], post + post)

