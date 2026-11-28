"""CPU-only follow-up on the locked S0 solution and held feature shards."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import unicodedata

import torch

from .evaluate import boot_interval, fit_scale, scope_metrics
from .experiment import bound_solution
from .features import load_manifest, read_features
from .scope_solver import tt_svd


def summaries(values):
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values), positive=sum(x>0 for x in values), ordered=sorted(values))


def question_words(text):
    return tuple(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def grams(words, n):
    return {words[i:i+n] for i in range(len(words)-n+1)}


def template_audit(official, candidates):
    pool = []
    for item in candidates:
        words = question_words(item['question'])
        pool.append((item, words, grams(words,3), grams(words,5)))

    def counts(owner_only):
        exact = shared5 = jaccard = 0
        prefixes, strong_families, five_families = Counter(), Counter(), Counter()
        for item in official:
            words = question_words(item['question'])
            three,five = grams(words,3),grams(words,5)
            events = []
            for row,other,t3,t5 in pool:
                if owner_only and row['owner']!=item['owner']:
                    continue
                overlap = len(three&t3)/len(three|t3) if three|t3 else 0.
                prefix = next((i for i,(a,b) in enumerate(zip(words,other)) if a!=b), min(len(words),len(other)))
                events.append((words==other, bool(five&t5), overlap>=.50, prefix, row.get('family',row['role'])))
            exact += any(x[0] for x in events)
            shared5 += any(x[1] for x in events)
            jaccard += any(x[2] for x in events)
            prefixes[max((x[3] for x in events),default=0)] += 1
            strong_families.update({x[4] for x in events if x[2]})
            five_families.update({x[4] for x in events if x[1]})
        return dict(exact=exact, shared_five=shared5, jaccard_3_ge_half=jaccard,
                    longest_prefix_counts=dict(sorted(prefixes.items())),
                    strong_overlap_families=dict(strong_families), shared_five_families=dict(five_families))

    return dict(candidates=len(candidates), official_events=len(official),
                candidate_families=dict(Counter(r.get('family',r['role']) for r in candidates)),
                same_owner=counts(True), entire_panel=counts(False))


def main():
    torch.set_num_threads(4)
    plan = load_manifest('panel_run.json')
    archive = json.loads(Path('s0_results.private.json').read_text())
    solutions = bound_solution('solution.pt', 'panel_run.json')
    panel = {row['id']:row for row in plan['rows']}
    data = read_features(['held0.pt','held1.pt'])
    calibration = set(plan['calibration_owners'])
    main_roles = {'hold_pos','hold_ind','hold_vis'}
    assert len(solutions)==30 and len(calibration)==15
    for event_id, feature in data.items():
        row = panel[event_id]
        assert (feature['owner'],feature['role'],feature['image']) == (
            row.get('owner'),row['role'],row['source_sha256'])

    ranks = (8,16,32,64)
    arms = defaultdict(dict)
    for owner, entry in solutions.items():
        a = entry['K-U']['A']
        right = tt_svd(a,bond=max(ranks))
        flattened = a.reshape(-1,128)
        arms['K-U'][owner] = fit_scale(a,entry['fits'])
        arms['K-P4'][owner] = fit_scale(entry['K-P4']['A'],entry['fits'])
        for rank in ranks:
            basis = right[:rank]
            approx = ((flattened @ basis.T) @ basis).reshape_as(a)
            arms[f'TT-SVD-{rank}'][owner] = fit_scale(approx,entry['fits'])

    def score(feature, transform):
        basis,den = transform
        return float((feature[0].to(basis) @ basis.T).norm()/den)

    def metric(rows, arm, field=None):
        result = scope_metrics([dict(role=r['role'],score=r['scores'][arm]) for r in rows])
        return result if field is None else result[field]

    clustered, by_owner = defaultdict(list), defaultdict(list)
    official = defaultdict(list)
    for _, feature in sorted(data.items()):
        owner,role,vec = feature['owner'],feature['role'],feature['feature']
        if role in main_roles and owner not in calibration:
            row = dict(role=role,scores={k:score(vec,v[owner]) for k,v in arms.items()})
            clustered[feature['image']].append(row)
            by_owner[owner].append(row)
        if role=='official_ind':
            for k,v in arms.items():
                official[k].append(score(vec,v[owner]))
    active = [r for group in clustered.values() for r in group]
    assert len(active)==archive['events']==60 and len(clustered)==archive['images']==15
    assert len(official['K-U'])==29
    for arm in ('K-U','K-P4'):
        for key,value in metric(active,arm).items():
            assert abs(value-archive['arms'][arm][key])<1e-10
        assert abs(statistics.median(official[arm])-archive['official_ind'][arm])<1e-10

    worst, rank_scan = {}, {}
    for arm in ('K-U','K-P4'):
        margins = defaultdict(list)
        for rows in by_owner.values():
            pos = min(r['scores'][arm] for r in rows if r['role']=='hold_pos')
            ind = max(r['scores'][arm] for r in rows if r['role']=='hold_ind')
            vis = max(r['scores'][arm] for r in rows if r['role']=='hold_vis')
            for kind,negative in (('independent',ind),('visual',vis),('combined', max(ind,vis))):
                margins[kind].append(pos-negative)
        worst[arm] = {k:summaries(v) for k,v in margins.items()}
    for arm in arms:
        value = metric(active,arm)
        value['official_ind_median'] = statistics.median(official[arm])
        value['paired_vs_ku'] = {}
        for field in ('auc_ind','auc_vis'):
            def change(rows):
                return metric(rows,arm,field)-metric(rows,'K-U',field)
            value['paired_vs_ku'][field] = dict(delta=change(active),ci95=boot_interval(clustered,change))
        rank_scan[arm] = value

    extras = {}
    for role in ('t2l','ood'):
        values = []
        for v in data.values():
            if v['role']!=role:
                continue
            transforms = arms['K-U']
            candidates = transforms.values() if role=='ood' else [transforms[v['owner']]]
            values.append(max(score(v['feature'], tr) for tr in candidates))
        extras[role] = dict(in_manifest=sum(r['role']==role for r in panel.values()),
                            frozen_features=len(values),
                            read_in=summaries(values) if values else None)

    official_rows = [r for r in panel.values() if r['role']=='official_ind']
    train = [r for r in panel.values() if r['role'] in {'native','fit','train_eq'}]
    val = [r for r in panel.values() if r['role']=='val_pos']
    output = dict(images=len(clustered), events=len(active), official_ind_events=len(official_rows),
                  worst_margin=worst, rank_scan=rank_scan, extra_feature_support=extras,
                  overlap=dict(train=template_audit(official_rows,train), validation=template_audit(official_rows,val)))
    print(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
