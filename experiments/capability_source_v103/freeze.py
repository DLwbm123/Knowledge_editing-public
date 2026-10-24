"""Finalize the cohort only after complete source-only adjudication."""
import json
import os
from pathlib import Path
import time


def main():
    root = Path(os.environ['TASK_ROOT'])
    out = root / '.private/capability_source_v103_20261024'
    window = json.loads((out / 'WINDOW.json').read_text())
    assert time.time() < window['hard_end'], 'Original source window expired'
    assert not (out / 'MODEL_COHORT_FROZEN.json').exists(), 'Already frozen'
    pool = json.loads((out / 'REVIEW_POOL_FROZEN.json').read_text())
    review = json.loads((out / 'SOURCE_REVIEW_FROZEN.json').read_text())
    decisions = review['cases']
    assert len(pool) <= 48 and len(decisions) == len(pool)
    assert set(decisions) == {r['case'] for r in pool}
    assert review['model_answers_seen'] == 0
    assert len({r['patient'] for r in pool}) == len(pool)
    counts, chosen = {}, []
    for label in ['nodule', 'infiltrates']:
        for polarity in ['positive', 'negative']:
            eligible = [r for r in pool if r['label'] == label and r['polarity'] == polarity and decisions[r['case']]['admit']]
            counts[f'{label}_{polarity}'] = len(eligible)
            chosen.extend(dict(r, panel='D' if i < 4 else 'C') for i, r in enumerate(eligible[:8]))
    passed = all(n >= 8 for n in counts.values())
    if passed:
        assert len(chosen) == 32
        assert all(sum(r['panel'] == p for r in chosen) == 16 for p in ['D', 'C'])
        (out / 'MODEL_COHORT_FROZEN.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=2))
    result = dict(status='SOURCE_TEXT_ADMISSION_PASS' if passed else 'SOURCE_ADMISSION_FAIL_STOP_SCREENING',
        reviewed_patients=len(pool), source_admitted=sum(counts.values()), source_vetoes=len(pool)-sum(counts.values()),
        eligible_by_stratum=counts, frozen_model_patients=len(chosen) if passed else 0,
        excluded_prior_patients=92, patient_overlap=0, pool_expansions=0,
        model_calls=0, gpu_seconds=0, parameter_updates=0, image_extractions=0,
        image_materialization_and_native_input_checks='PENDING_BEFORE_MODEL_USE',
        clinical_independent_review=False, medgemma_access='BLOCKED_PREVIOUS_HTTP403_NOT_REPOLLED',
        source_phase_elapsed_seconds=time.time()-window['original_start'], original_window_preserved=True,
        remote_bytes=0)
    (out / 'COMPLETE.json').write_text(json.dumps(result, indent=2))
    result['local_private_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    assert result['local_private_bytes'] < window['local_bytes_limit']
    (root / 'reports/capability_source_v103_20261024/RESULTS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
