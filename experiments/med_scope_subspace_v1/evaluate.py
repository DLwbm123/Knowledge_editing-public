"""Scope diagnostics, paired image inference and routing metrics."""
from bisect import bisect_left
from collections import defaultdict
import math
import random
import statistics

import torch


def boot_interval(groups, estimate, replicates=2000, seed=20261127):
    rng = random.Random(seed)
    scopes = list(groups.values())
    if len(scopes) < 2:
        return None
    samples = []
    for _ in range(replicates):
        data = [row for group in rng.choices(scopes, k=len(scopes)) for row in group]
        value = estimate(data)
        if value is not None and math.isfinite(value):
            samples.append(value)
    if len(samples) < .9 * replicates:
        return None
    tails = statistics.quantiles(samples,n=40,method='inclusive')
    return [tails[0],tails[-1]]


def scope_metrics(rows):
    pos = [r['score'] for r in rows if r['role']=='hold_pos']
    ind = [r['score'] for r in rows if r['role']=='hold_ind']
    vis = [r['score'] for r in rows if r['role']=='hold_vis']
    def auc(neg):
        if not pos or not neg:
            return None
        neg.sort()
        return sum(bisect_left(neg,p) for p in pos)/(len(pos)*len(neg))
    neg = ind + vis
    def delta(x):
        return statistics.median(pos)-statistics.median(x) if pos and x else None
    return dict(delta=delta(neg),delta_ind=delta(ind),delta_vis=delta(vis),auc_ind=auc(ind),auc_vis=auc(vis))


def s0_report(scored, replicates=2000):
    rows = []
    for key, arms in sorted(scored.items()):
        arm = arms['K-TT']
        rows.append(dict(id=key,role=arm['role'],image=arm['image'],
                         by_arm={k:v['score'] for k,v in arms.items()}))
    if not rows:
        raise ValueError('Missing held-out scope evidence')
    candidates = sorted(rows[0]['by_arm'])
    if any(set(r['by_arm']) != set(candidates) for r in rows):
        raise ValueError('Incomplete S0 paired support')
    clusters = defaultdict(list)
    for row in rows:
        clusters[row['image']].append(row)
    required = {'hold_pos','hold_ind','hold_vis'}
    if any(not required <= {r['role'] for r in v} for v in clusters.values()):
        raise ValueError('S0 requires positive, independent negative and visual negative support on every evaluation image')
    def metrics(data, arm):
        return scope_metrics([dict(role=r['role'],score=r['by_arm'][arm]) for r in data])
    readouts = {arm:metrics(rows,arm) for arm in candidates}
    comparisons = {}
    for arm in candidates:
        for anchor in ('K-P4','K-U'):
            if arm == anchor:
                continue
            for metric in ('delta','delta_ind','delta_vis','auc_ind','auc_vis'):
                def change(data):
                    a,b = metrics(data,arm)[metric],metrics(data,anchor)[metric]
                    return None if a is None or b is None else a-b
                comparisons[f'{arm} vs {anchor} {metric}'] = dict(
                    delta=change(rows),ci95=boot_interval(clusters,change,replicates))
    p4,u,tt = [readouts[k] for k in ('K-P4','K-U','K-TT')]
    supported = all(v is not None for arm in (p4,u,tt) for v in arm.values())
    enter = supported and tt['delta'] >= p4['delta'] - .02 and all(
        tt[k] >= p4[k] for k in ('auc_ind','auc_vis'))
    return dict(arms=readouts,comparisons=comparisons,events=len(rows),images=len(clusters),
                decision='S1_ADMITTED' if enter else
                'INITIALIZATION_ONLY' if supported else 'INSUFFICIENT_SUPPORT')


def d0(rows, image_groups, replicates=2000):
    import numpy as np
    roles = ('paraphrase','distractor_insertion','numeric_perturbation','negation_uncertainty')
    events = []
    nonpositive = defaultdict(int)
    unavailable = defaultdict(int)
    for r in rows:
        if r['role']!='T2G' or r['variant'] not in roles or not r.get('routed_to_owner'):
            continue
        base = r['arms']['Base']['nll']
        g = base-r['arms']['MDU']['nll']
        residual = r['arms']['MDU']['residual']
        if g <= 0:
            nonpositive[r['variant']] += 1
        if not all(math.isfinite(v) for v in (base,g,residual)) or base <= 0 or residual <= 0:
            unavailable[r['variant']] += 1
            continue
        events.append((image_groups[str(r['owner'])],r['variant'],base,g,residual))

    def analyze(panel, log_gain):
        if not panel:
            return dict(events=0,images=0,delta=None,ci95=None)
        lr = [math.log(x[4]) for x in panel]
        center = statistics.median(lr)
        matrix = []
        response = []
        groups = defaultdict(list)
        for j,(im,kind,base,g,r) in enumerate(panel):
            indicators = [float(kind==k) for k in roles[1:]]
            x = lr[j]-center
            matrix.append([1.,x,*indicators,*[a*x for a in indicators],math.log(base)])
            response.append(math.log(g) if log_gain else g)
            groups[im].append(j)
        X = np.asarray(matrix)
        y = np.asarray(response)
        ind = 2+roles[1:].index('negation_uncertainty')
        def estimate(ix):
            coef,_,rank,_ = np.linalg.lstsq(X[ix],y[ix],rcond=None)
            return float(coef[ind]) if rank == X.shape[1] else None
        return dict(events=len(panel),images=len(groups),delta=estimate(list(range(len(panel)))),
                    ci95=boot_interval(groups,estimate,replicates),median_log_residual=center)
    primary = analyze([r for r in events if r[3]>0],True)
    linear = analyze(events,False)
    ci = primary['ci95']
    decision = ('WRITE_OUT_MISMATCH' if ci is not None and ci[1]<0 else
                'WRITE_OUT_MISMATCH_NOT_ESTABLISHED' if ci is not None else 'INSUFFICIENT_SUPPORT')
    return dict(primary=primary,linear_sensitivity=linear,gain_nonpositive=dict(nonpositive),
                invalid=dict(unavailable),decision=decision)


def routing_report(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['route'],r['role'])].append(r)
    official = {'official_ind','official_paraphrase','official_numeric','official_distractor'}
    for route in {route for route,_ in groups}:
        if all((route,role) in groups for role in official):
            groups[(route,'official_t2g')] = [row for role in sorted(official) for row in groups[(route,role)]]
    out = {}
    for (route,role), items in groups.items():
        decided = [r['core'] for r in items if r.get('core') is not None]
        images = defaultdict(list)
        for r in items:
            images[r['image']].append(r)
        def rate(events):
            return sum(r.get('selected') is not None for r in events)/len(events)
        out.setdefault(route,{})[role] = dict(events=len(items),images=len(images),
            activated=sum(r.get('selected') is not None for r in items),
            activated_ci95=boot_interval(images,rate),
            image_any_activation=sum(any(r.get('selected') is not None for r in v) for v in images.values()),
            correct_expert=sum(
                r.get('owner') is not None and r.get('owner') == r.get('selected') for r in items),
            core_correct=sum(decided),core_unknown=len(items)-len(decided),
            target_exact=sum(bool(r.get('exact')) for r in items),
            exact_support=sum(r.get('exact') is not None for r in items),
            medians={key:statistics.median(values) for key in
                ('first_nll','answer_nll','reference_kl','target_path_kl','target_path_nll','base_kl')
                if (values := [r['metric'][key] for r in items if key in r.get('metric',{})])})
    compared = {}
    for role in sorted({role for _,role in groups}):
        arms = sorted(a for a,k in groups if k==role)
        comparisons = [('P2','P4'),('MDU','P4'),('P4','V-TT'),('P4','V-TT-ft')]
        comparisons += [('R0',k) for k in arms if k!='R0']
        for anchor,arm in comparisons:
            if anchor not in arms or arm not in arms:
                continue
            a = {r['id']:r for r in groups[(anchor,role)]}
            b = {r['id']:r for r in groups[(arm,role)]}
            if set(a)!=set(b):
                raise ValueError('Paired evaluation support differs')
            paired = defaultdict(list)
            for key in a:
                if a[key]['image']!=b[key]['image']:
                    raise ValueError('Paired image identity differs')
                paired[a[key]['image']].append((a[key],b[key]))
            pair_rows = [x for v in paired.values() for x in v]
            known = [(l['core'],r['core']) for l,r in pair_rows if l.get('core') is not None and r.get('core') is not None]
            effects = {}
            field_pairs = [('core',lambda r:r.get('core')),('exact',lambda r:r.get('exact'))]
            field_pairs += [(metric,lambda r,k=metric:r.get('metric',{}).get(k)) for metric in
                            ('answer_nll','reference_kl','target_path_kl','target_path_nll','base_kl')]
            for metric,get in field_pairs:
                def est(data):
                    vals = [(get(l),get(r)) for l,r in data if get(l) is not None and get(r) is not None]
                    if not vals:
                        return None
                    left,right = zip(*vals)
                    average = statistics.mean if metric in {'core','exact'} else statistics.median
                    return average(right)-average(left)
                effects[metric] = dict(delta=est(pair_rows),ci95=boot_interval(paired,est))
            compared[f'{arm} vs {anchor} – {role}'] = dict(events=len(a),images=len(paired),
                gain=sum(not l and r for l,r in known),loss=sum(l and not r for l,r in known),
                unknown=len(a)-len(known),effects=effects)
    return dict(groups=out,paired=compared)


def fit_scale(a, fits):
    from .scope_solver import row_basis
    basis = row_basis(a)
    denominator = torch.quantile((fits.to(basis) @ basis.T).norm(dim=-1),.5)
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError('FIT projection denominator is zero or nonfinite')
    return basis, denominator
