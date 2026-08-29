import pytest

from m3bench_repro.evaluation.metrics import ProbeOutcome, generality, harmonic_mean, locality, macro_average


def test_locality_excludes_previously_wrong_probes() -> None:
    probes = [ProbeOutcome(True, True), ProbeOutcome(True, False), ProbeOutcome(False, False)]
    assert locality(probes) == 0.5


def test_generality_excludes_previously_correct_probes() -> None:
    probes = [ProbeOutcome(False, True), ProbeOutcome(False, False), ProbeOutcome(True, False)]
    assert generality(probes) == 0.5


def test_macro_is_not_probe_micro_average() -> None:
    per_edit = [locality([ProbeOutcome(True, True)]), locality([ProbeOutcome(True, False)] * 9)]
    assert macro_average(per_edit) == 0.5
    assert locality([ProbeOutcome(True, True)] + [ProbeOutcome(True, False)] * 9) == pytest.approx(0.1)


def test_harmonic_mean_penalizes_zero() -> None:
    assert harmonic_mean([1.0, 0.5]) == pytest.approx(2 / 3)
    assert harmonic_mean([1.0, 0.0]) == 0.0
