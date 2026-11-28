"""Bind machine-reviewed expansion data; preserve all unadmitted rows and reasons."""
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

from experiments.med_mdu_v2_admission_v1.freeze_data import SEED, norm
from .expand_panel import blind_csv, load, write
from .features import load_manifest


def split_independent(rows, frozen):
    assignments = {}
    pools = defaultdict(list)
    for r in frozen:
        if r['role'] in {'train_ind','cal_ind','hold_ind'}:
            k = r['source_sha256'],norm(r['question'])
            assignments[k] = 'train' if r['role']=='train_ind' else 'holdout'
    for r in rows:
        group, question = r['image_sha256'],norm(r['question'])
        if r['role'] in {'train','holdout'}:
            if assignments.setdefault((group,question),r['role'])!=r['role']:
                raise ValueError('An earlier negative role would change')
        pools[group].append(r)
    for group, items in sorted(pools.items()):
        options = defaultdict(set)
        by_owner = defaultdict(list)
        for r in items:
            q = norm(r['question'])
            options[q].add(r['owner'])
            by_owner[r['owner']].append(q)
        keys = sorted(options)
        random.Random(SEED).shuffle(keys)
        for q in keys:
            if (group,q) in assignments:
                continue
            def missing(role):
                return sum(not any(assignments.get((group,x))==role for x in by_owner[o])
                           for o in options[q])
            assignments[group,q] = 'train' if missing('train')>=missing('holdout') else 'holdout'
    return [(r,assignments[r['image_sha256'],norm(r['question'])]) for r in rows]


def main(root, s0_path, transforms_path):
    draft = load(root/'expansion.private.json')
    labels = load(root/'llm_review/results.private.json')
    plan = load_manifest(s0_path)
    transforms = load(transforms_path)
    assert set(transforms)==set(draft['sources'])
    images = {int(o):v['source_rows'][0]['image_sha256'] for o,v in draft['sources'].items()}
    if set(images.values()) & set(draft['protected_image_sha256']):
        raise ValueError('Protected image reached expansion')
    for kind, pack in labels.items():
        expected = {r['opaque_query_id'] for r in load(root/f'{kind}_input.private.json')}
        if (set(pack['decisions'])|set(pack['unknown']))!=expected or pack['decisions'].keys() & set(pack['unknown']):
            raise ValueError('Review labels do not match the locked query roster')
    by_view = defaultdict(list)
    for v in draft['views']:
        if labels['view']['decisions'].get(v['review_id'])=='accept':
            by_view[v['owner']].append(v)
    independent = [r for r in draft['negatives'] if labels['negative']['decisions'].get(r['review_id'])=='independent']
    by_neg = defaultdict(lambda:defaultdict(list))
    for r,role in split_independent(independent,plan['rows']):
        by_neg[r['owner']][role].append(r)
    neg_review = load(root/'negative_input.private.json')
    for row in neg_review:
        row['stratum'] = labels['negative']['decisions'].get(row['opaque_query_id'],'unknown')
    final_review = blind_csv(root/'negatives_by_model_label_human_review.private.csv',neg_review,SEED)

    known_cal = {r['source_sha256'] for r in plan['rows'] if r['role'].startswith('cal_')}
    known_eval = {r['source_sha256'] for r in plan['rows'] if r['role'] in {'hold_ind','hold_vis','ood'}}
    checks = {}
    for o in images:
        views = {(r['role'],r['family']) for r in by_view[o]}
        required = {('train',a) for a in ('declarative','modal','context')}
        required |= {('validation',a) for a in ('declarative','modal','context')}
        required |= {('holdout_modal','modal'),('holdout_context','context')}
        checks[o] = sorted([f'{role}/{family}' for role,family in required-views]
                           + [f'negative/{r}' for r in ('train','holdout') if not by_neg[o][r]])
    ready = [o for o in sorted(images) if not checks[o]]
    new_cal_pool = sorted({images[o] for o in ready}-known_cal-known_eval)
    random.Random(SEED).shuffle(new_cal_pool)
    cal_images = known_cal | set(new_cal_pool[:max(0,19-len(known_cal))])
    calibration_owners = set(plan['calibration_owners'])
    mixed_schedules = dict(plan['schedules'])
    schedules = {}
    rows = list(plan['rows'])
    train_families = set(plan['train_families'])
    for o in ready:
        source = draft['sources'][str(o)]
        old = source['source_rows']
        base = old[0]
        frame = dict(owner=o,dataset=source['dataset'],source_sha256=images[o],image_sha256=images[o],
                     image_path=base['image_path'],original_image_path=base.get('original_image_path',base['image_path']))
        is_cal = images[o] in cal_images
        if is_cal:
            calibration_owners.add(o)

        def add(tag,role,q,y,**extra):
            rid = f'{o}-{tag}'
            rows.append(dict(frame,id=rid,role=role,question=q,answer=y,**extra))
            return rid

        native,*fits = [add(f'source-{i}','fit' if i else 'native',item['question'],item['reference'])
                        for i,item in enumerate(old)]
        view_ids = {}
        for i,v in enumerate(by_view[o]):
            role = {'train':'train_eq','validation':'val_pos'}.get(v['role'],'hold_pos')
            view_ids[v['role'],v['family']] = add(f'expanded-{i}',role,v['question'],v['reference'],
                                                   family=v['template'])
            if role=='train_eq':
                train_families.add(v['template'])
        views = [view_ids.get(('train','ordinary'),fits[0])] + [
            view_ids['train',k] for k in ('declarative','modal','context')]
        neg = {}
        for role in ('train','holdout'):
            candidates = sorted(by_neg[o][role],key=lambda r:r['query_id'])
            if role=='train':
                candidates = candidates[:2]
            kind = 'train_ind' if role=='train' else 'cal_ind' if is_cal else 'hold_ind'
            neg[role] = [add(r['query_id'],kind,r['question'],r['gold_answer'],
                             base_tokens=r['base_continuation']['raw_token_ids'],relation='independent')
                         for r in candidates]
        visual = []
        for tag in ('black','strong_noise','strong_blur','validation_noise'):
            transformed = transforms[str(o)][tag]
            kind = 'cal_vis' if is_cal else 'hold_vis'
            if tag in {'black','strong_noise'}:
                kind = 'train_vis'
            elif tag=='validation_noise':
                kind = 'val_neg'
            # Preserve both source and transformed identities.
            rid = add('visual-'+tag,kind,base['question'],base['reference'])
            rows[-1].update(transformed)
            if kind=='train_vis':
                visual.append(rid)
        schedules[str(o)] = [dict(native=native,views=views,negatives=neg['train']) for _ in range(160)]
        mixed_schedules[str(o)] = [dict(native=native,views=views,
                                         negatives=[neg['train'][0],visual[i%2]]) for i in range(160)]

    development = set(plan.get('development_image_sha256',[])) | set(images.values())
    initial_bg = sum(r['role']=='bg' for r in rows)
    rows = [r for r in rows if r['role'] not in {'bg','ood'} or r['source_sha256'] not in development]
    out = dict(plan,rows=rows,schedules=mixed_schedules,independent_only_schedules=schedules,
               calibration_owners=sorted(calibration_owners),train_families=sorted(train_families),
               development_image_sha256=sorted(development),expansion_status='MACHINE_REVIEWED_HUMAN_PENDING')
    path = root/'extended_candidate_manifest.private.json'
    write(path,out)
    load_manifest(path)
    labels_count = {k:dict(Counter(pack['decisions'].values())) for k,pack in labels.items()}
    result = dict(ready_edits=len(ready),blocked_edits=len(images)-len(ready),
                  missing_requirements=dict(Counter(reason for misses in checks.values() for reason in misses)),
                  label_count=labels_count,unknown={k:len(v['unknown']) for k,v in labels.items()},
                  provisional_calibration_images=len(cal_images),human_reviewed=0,
                  negative_sample=final_review,background_removed=initial_bg-sum(r['role']=='bg' for r in rows),
                  independent_only_schedules=len(schedules),mixed_schedules=len(ready))
    write(root/'expansion_verified_summary.json',result)
    print(result)


if __name__=='__main__':
    main(*(Path(x) for x in sys.argv[1:]))
