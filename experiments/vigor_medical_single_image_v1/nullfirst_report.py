"""Offline decomposition of already recorded null directional derivatives."""
import csv
import json
import math
from pathlib import Path


def summarize(source, output):
    source, output = Path(source), Path(output)
    result = json.loads((source/'RESULT.json').read_text())
    audit = json.loads((source/'PROTOCOL_AUDIT.json').read_text())
    assert result['status'] == 'COMPLETE' and result['probe_count'] == 4
    assert audit['status'] == 'PASS' and result['all_base_restored']
    rows = []
    for state in result['arms']:
        assert set(state['probes']) == {'F0', 'F1'}
        for metric in ('F1', 'F0'):
            probe = state['probes'][metric]
            p = probe['proposal']
            assert probe['endpoint_restored'] and probe['matching_passed']
            total = p['fp32_after'][1] - p['fp32_before'][1]
            linear = p['fp32_update']['null_linear_change']
            native = p['native_after'][1] - p['native_before'][1]
            rows.append(dict(endpoint=state['arm'], metric=metric,
                fp32_total=total, fp32_linear=linear, fp32_residual=total-linear,
                native_total=native, precision_native_minus_fp32=native-total,
                fp32_update_norm=p['quantization']['fp32_delta_l2'],
                null_gradient_norm=None, null_gradient_update_cosine=None,
                original_native_accepted=probe['verdict']['accepted']))
    assert len(rows) == 4
    index = {(r['endpoint'], r['metric']): r for r in rows}
    fields = ['fp32_total', 'fp32_linear', 'fp32_residual', 'native_total', 'precision_native_minus_fp32']
    contrasts = {m: {k: index['D', m][k]-index['A', m][k] for k in fields} for m in ('F1','F0')}
    interaction = {k: contrasts['F0'][k]-contrasts['F1'][k] for k in fields}
    for row in rows + list(contrasts.values()) + [interaction]:
        assert all(math.isfinite(row[k]) for k in fields)
        assert abs(row['fp32_total']-row['fp32_linear']-row['fp32_residual']) < 1e-12
        assert abs(row['native_total']-row['fp32_total']-row['precision_native_minus_fp32']) < 1e-12
        row['linear_residual_opposite_sign'] = row['fp32_linear']*row['fp32_residual'] < 0
        row['linear_abs_fraction_of_total'] = abs(row['fp32_linear']/row['fp32_total']) if row['fp32_total'] else None
        row['residual_abs_exceeds_linear'] = abs(row['fp32_residual']) > abs(row['fp32_linear'])
    output.mkdir(exist_ok=True, parents=True)
    summary = dict(status='COMPLETE_OFFLINE', fixed_existing_records=4, endpoint_denominator=2,
        new_candidates=0, new_model_calls=0, new_gradient_calls=0, gpu_seconds=0,
        existing_source_status='COMPLETE', existing_source_audit='PASS',
        missing=['Euclidean_null_gradient_norm','Euclidean_null_gradient_update_cosine'],
        private_weights_or_vectors_recomputed=False, decomposition_identities_checked=14)
    for name, value in [('READOUTS.json', rows), ('CONTRASTS.json', contrasts), ('INTERACTION.json', interaction), ('SUMMARY.json', summary)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    with (output/'READOUTS.tsv').open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader();w.writerows(rows)
    return summary, rows, contrasts, interaction
