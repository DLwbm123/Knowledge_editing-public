"""Read fixed scalar training receipts; never inspect examples or model state."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path('/data/bmw/Knowledge_editing/outputs/med150-v1-20261027/run/training')


def window(curve, first, last):
    steps = curve[first-1:last]
    native = [row['source_NLL'][0] for row in steps]
    by_fit = [[row['source_NLL'][1] for row in steps if (row['step']-1)%4 == j] for j in range(4)]
    assert len(native) == 20 and all(len(x) == 5 for x in by_fit)
    fit = [statistics.mean(x) for x in by_fit]
    return {'native_mean': statistics.mean(native), 'fit_means': fit,
            'fit_mean': statistics.mean(fit), 'fit_range': max(fit)-min(fit),
            'max_fit_index': fit.index(max(fit))}


def main():
    records=[]
    for index, owner in enumerate((147,148,149,150), 1):
        d=json.loads((ROOT/f'{owner}.json').read_text()); curve=d['curve']
        assert d['steps']==160 and len(curve)==160
        for i,row in enumerate(curve,1):
            assert row['step']==i and len(row['source_NLL'])==2
            assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in row['source_NLL'])
        a=window(curve,1,20); b=window(curve,141,160)
        signal=b['fit_range']>=.05 and a['max_fit_index']==b['max_fit_index']
        records.append({'anonymous_edit': index, 'first':a, 'last':b,
                        'native_mean_change':b['native_mean']-a['native_mean'],
                        'fit_mean_change':b['fit_mean']-a['fit_mean'], 'screen':signal})
    print(json.dumps({'status':'COMPLETE','records':records,'screen_edits':sum(r['screen'] for r in records),
                      'screen_triggered':sum(r['screen'] for r in records)>=2,
                      'model_calls':0,'generation':0,'judge':0,'train_updates':0}))


if __name__ == '__main__': main()
