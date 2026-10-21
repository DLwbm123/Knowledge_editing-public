"""Scalar-only, bounded posthoc analysis; never changes DQ qualification."""
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, median


def summary(values):
    values = sorted(values)
    return dict(n=len(values), minimum=values[0], median=median(values),
                p90_nearest_rank=values[math.ceil(.9*len(values))-1], maximum=values[-1])


def pearson(x, y):
    a = [v-mean(x) for v in x]; b = [v-mean(y) for v in y]
    den = math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
    return sum(u*v for u, v in zip(a, b))/den if den else None


def analyze(result, rankcorr):
    rows = result['measurements']
    expected = [(d+str(i), s, side) for d in 'RGB' for i in range(1, 5)
                for s in [.25, .5, 1.] for side in [-1, 1]]
    assert [(r['direction'], r['scale'], r['sign']) for r in rows] == expected
    assert all(len(r['record_KL']) == 32 and all(math.isfinite(v) and v >= 0
               for v in r['record_KL']) for r in rows)
    classes = {}; points = []; asymmetry = []; exponents = []
    for category in 'RGB':
        work = [r for r in rows if r['direction'][0] == category and r['scale'] == 1]
        classes[category] = {}
        for side in ['C', 'H']:
            classes[category][side] = dict(
                spearman={k: rankcorr([r[k] for r in work], [r[side+'_KL'] for r in work])
                          for k in ['audit', 'fit', 'F0', 'norm_squared', 'misaligned_audit']},
                audit_actual_ratio=summary([r['audit']/r[side+'_KL'] for r in work]),
                over_factor4=sum(r['audit'] > 4*r[side+'_KL'] for r in work),
                under_factor4=sum(4*r['audit'] < r[side+'_KL'] for r in work))
    nominal_norm = 17.58567032970479
    for row in rows:
        tails = {}
        for side, vals in [('C', row['record_KL'][:24]), ('H', row['record_KL'][24:])]:
            tails[side] = dict(**summary(vals), mean=mean(vals),
                              top4_KL_mass_fraction=sum(sorted(vals)[-4:])/sum(vals))
        points.append(dict(point=row['point'], direction=row['direction'], scale=row['scale'],
                           sign=row['sign'], native_norm=row['native_norm'],
                           native_nominal_norm_relative_error=row['native_norm']/(nominal_norm*row['scale'])-1,
                           C_audit_ratio=row['audit']/row['C_KL'],
                           H_audit_ratio=row['audit']/row['H_KL'], tails=tails))
    lookup = {(r['direction'], r['scale'], r['sign']): r for r in rows}
    for direction in [d+str(i) for d in 'RGB' for i in range(1, 5)]:
        for scale in [.25, .5, 1.]:
            minus, plus = [lookup[direction, scale, side] for side in [-1, 1]]
            asymmetry.append(dict(direction=direction, scale=scale,
                C_positive_negative_ratio=plus['C_KL']/minus['C_KL'],
                H_positive_negative_ratio=plus['H_KL']/minus['H_KL']))
        for side in [-1, 1]:
            lo, hi = lookup[direction, .25, side], lookup[direction, 1., side]
            exponents.append(dict(direction=direction, sign=side,
                **{s+'_nominal_exponent': math.log(hi[s+'_KL']/lo[s+'_KL'])/math.log(4) for s in ['C', 'H']},
                **{s+'_native_norm_exponent': math.log(hi[s+'_KL']/lo[s+'_KL'])/
                   math.log(hi['native_norm']/lo['native_norm']) for s in ['C', 'H']}))
    centered = {}
    work = [r for r in rows if r['scale'] == 1]
    for side in ['C', 'H']:
        x, y = [], []
        for category in 'RGB':
            rr = [r for r in work if r['direction'][0] == category]
            xx, yy = [math.log(r['audit']) for r in rr], [math.log(r[side+'_KL']) for r in rr]
            x.extend(v-mean(xx) for v in xx); y.extend(v-mean(yy) for v in yy)
        centered[side] = pearson(x, y)
    errors = [abs(math.log2(p['C_audit_ratio'])) for p in points]
    return dict(scope='posthoc_not_new_qualification', points=len(rows), record_measurements=32*len(rows),
        work_by_class=classes, signed_scale_curves=asymmetry, endpoint_scale_exponents=exponents,
        point_native_and_tails=points, class_centered_work_log_Pearson=centered,
        native_norm_deviation=summary([p['native_nominal_norm_relative_error'] for p in points]),
        native_norm_deviation_error_Spearman=rankcorr(
            [abs(p['native_nominal_norm_relative_error']) for p in points], errors),
        rounding_vector_error='NA_not_persisted', derivative_nonlinearity_error_attribution='UNRESOLVED',
        model_calls=0, judge_calls=0, DQ_qualification_changed=False)


def main():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('scalar_rank', Path(__file__).with_name('distQ_summary.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.corr([1, 1, 2, 3], [2, 2, 4, 6]) == 1
    assert module.corr([1, 2, 3], [3, 2, 1]) == -1
    assert summary(list(range(1, 9)))['p90_nearest_rank'] == 8
    assert pearson([1, 2, 3], [2, 4, 6]) == 1
    value = analyze(json.loads((root/'reports/vigor_distributionQ_v1_20261020/completion/RESULT.json').read_text()), module.corr)
    out = root/'reports/vigor_distributionBridge_review_v1_20261020'
    (out/'DQ_DECOMPOSITION.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: v for k, v in value.items() if k not in ['signed_scale_curves', 'point_native_and_tails', 'endpoint_scale_exponents']}, indent=2))


if __name__ == '__main__':
    main()
