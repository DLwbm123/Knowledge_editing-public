from m3bench_repro.evaluation.evaluation_eligibility import require_task_eligible
import pytest
def test_null_is_not_false_for_task_construction():
 with pytest.raises(ValueError): require_task_eligible({"evaluation_eligible":True,"is_correct":None})
