from m3bench_repro.evaluation.evaluation_eligibility import require_evaluation_eligible
import pytest
def test_missing_gold_is_not_evaluation_eligible():
 with pytest.raises(ValueError): require_evaluation_eligible({"evaluation_eligible":False,"gold_answer":""})
