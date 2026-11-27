"""Relate T2L projection to its paired change in answer-prefix KL."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np


def ranks(x):
    x = np.asarray(x)
    return 1 + (x[None, :] < x[:, None]).sum(1) + ((x[None, :] == x[:, None]).sum(1) - 1) / 2


def spearman(x, y):
    a = ranks(x)
    b = ranks(y)
    a -= a.mean()
    b -= b.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / scale) if scale else float('nan')


def association(panel, x, y):
    a = np.array([x(p) for p in panel])
    b = np.array([y(p) for p in panel])
    groups = np.array([p['image_group'] for p in panel])
    keys = np.unique(groups)
    indices = [np.flatnonzero(groups==i) for i in keys]
    estimates = []
    rng = np.random.default_rng(20261127)
    for _ in range(4000):
        sample = np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        estimates.append(spearman(a[sample],b[sample]))
    values = [v for v in estimates if np.isfinite(v)]
    return dict(rho=spearman(a,b), ci95=np.quantile(values,[.025,.975]).tolist(),
                events=len(panel), image_groups=len(keys), bootstrap=len(values))


def summary(values):
    return dict(n=len(values), median=float(np.median(values)),
                iqr=np.quantile(values,[.25,.75]).tolist())


def main(root):
    data=[json.loads(line) for line in (root/'metrics.jsonl').open()]
    owner_group={int(k):v for k,v in json.loads((root/'image_groups.private.json').read_text()).items()}
    fit=defaultdict(list)
    for row in data:
        if row['role']=='FIT':
            fit[row['selected']].append(row)
    assert len(fit)==150 and all(len(v)==4 for v in fit.values())
    events=[]
    for row in data:
        if row['role']!='T2L':
            continue
        row['image_group']=owner_group[row['owner']]
        for arm in ('MED','MDU'):
            row[arm+'_ratio']=row['arms'][arm]['projection']/np.median(
                [v['arms'][arm]['projection'] for v in fit[row['selected']]])
        row['delta_kl']=row['arms']['MDU']['kl']-row['arms']['MED']['kl']
        row['delta_ratio']=row['MDU_ratio']-row['MED_ratio']
        row['delta_residual']=row['arms']['MDU']['residual']-row['arms']['MED']['residual']
        events.append(row)
    assert len(events)==54 and len({r['image_group'] for r in events})==29
    selected=[r for r in events if r['routed_to_owner']]
    high=[r for r in events if r['MDU_ratio']>=.5]
    low=[r for r in events if r['MDU_ratio']<.5]
    assert len(selected)==52 and len(high)==8

    fields=('MDU_ratio','MED_ratio','delta_ratio','delta_residual')
    metrics={k:association(events,lambda r:r[k],lambda r:r['delta_kl']) for k in fields}
    metrics['mdu_ratio_owner_route']=association(selected,lambda r:r['MDU_ratio'],lambda r:r['delta_kl'])
    by_image=defaultdict(list)
    for row in events:
        by_image[row['image_group']].append(row)
    collapsed=[{'image_group':g,'projection':np.median([r['MDU_ratio'] for r in rows]),
                'delta':np.median([r['delta_kl'] for r in rows])} for g,rows in by_image.items()]
    metrics['image_medians']=association(collapsed,lambda r:r['projection'],lambda r:r['delta'])
    for name,panel in (('at_least_0_5',high),('below_0_5',low)):
        metrics[name]={'kl_change':summary([r['delta_kl'] for r in panel]),
                       'core_errors':sum(not r['status']['MDU150']['core'] for r in panel),
                       'image_groups':len({r['image_group'] for r in panel})}
    print(json.dumps(metrics,allow_nan=False,indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
