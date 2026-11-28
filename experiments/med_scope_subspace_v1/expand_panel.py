"""Prepare the 120-editor extension without reading final or calibration metrics."""
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sys

from experiments.med_mdu_v2_admission_v1.freeze_data import SEED, norm, review_view


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)


def signature(question, native):
    q, source = norm(question), norm(native)
    frame = re.sub(r'(?<!\w)'+re.escape(source)+r'(?!\w)', ' slot ', q) if source else q
    w = frame.split()
    return {tuple(w[i:i+3]) for i in range(len(w)-2)}


def collision(candidate, native, official):
    words = signature(candidate,native)
    for other in official:
        target = signature(other,native)
        if norm(candidate)==norm(other) or (words and target and len(words&target)/len(words|target)>=.50):
            return True
    return False


def key(kind, private_id):
    return sha256(json.dumps([kind,private_id],ensure_ascii=False).encode()).hexdigest()


def blind_csv(path, rows, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row['stratum']].append(row)
    sampled = []
    for stratum,items in sorted(groups.items()):
        sampled.extend(rng.sample(items,min(30,len(items))))
    rng.shuffle(sampled)
    with path.open('x',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('id','stratum','source_question','source_reference','candidate','human_verdict'))
        for r in sampled:
            writer.writerow((r['opaque_query_id'],r['stratum'],r['question'],r['gold_answer'],
                             r['raw_base_answer'],''))
    return {s:min(30,len(v)) for s,v in sorted(groups.items())}


def main(root, prior, s0, historical):
    original = load(prior/'inventory.private.json')
    protector = set(load(s0)['protected_image_sha256'])
    old = load(prior/'stage1_panel.private.json')
    old_owners = {e['owner'] for e in old['entries']}
    s0_cal = set(load(s0)['calibration_owners'])
    roles = load(historical)['roles']
    official_ids = {q for by_role in roles.values() for role in ('T2G','T2L') for q in by_role[role]}
    frozen = defaultdict(list)
    for row in original['frozen']:
        if row['query'] in official_ids:
            frozen[row['sha']].append(row['question'])
    sources = {int(k):v for k,v in original['sources'].items()}
    rest = set(sources)-old_owners
    assert len(sources)==150 and len(rest)==120 and len(old_owners)==30
    images = {o:sources[o]['source_rows'][0]['image_sha256'] for o in sources}
    excluded = sorted(o for o in rest if images[o] in protector)
    allowed = rest-set(excluded)
    lock = load(prior/'template_lock.json')
    bases = load(prior/'base_continuations.private.json')
    previous = load(prior/'relation_labels.private.json')
    payloads = dict(view=[],negative=[])
    usable, flags = [], Counter()
    for v in load(prior/'view_candidates.private.json'):
        o = v['owner']
        if o not in allowed:
            continue
        review_view(v,sources,lock['templates'],lock['plain_rules'])
        native = sources[o]['source_rows'][0]
        if collision(v['question'],native['question'],frozen[images[o]]):
            flags[v['family'],v['role']] += 1
            continue
        opaque = key('view',[o,v['template']])
        payloads['view'].append(dict(opaque_query_id=opaque,question=native['question'],
                                  gold_answer=native['reference'],raw_base_answer=v['question'],
                                  stratum=v['family']+' / '+v['role']))
        usable.append(dict(v,review_id=opaque))
    view_support = defaultdict(list)
    for row in usable:
        view_support[row['owner']].append(row)

    negatives = []
    for row in previous:
        o = row['owner']
        if o not in allowed:
            continue
        assert row['image_sha256']==images[o]
        base = bases[row['query_id']]
        native = sources[o]['source_rows'][0]
        record = dict(other_question=row['question'],other_reference=row['gold_answer'],
                      base_continuation=base['model_answer_raw'])
        opaque = key('negative',[o,row['query_id']])
        payloads['negative'].append(dict(opaque_query_id=opaque,question=native['question'],
                                  gold_answer=native['reference'],raw_base_answer=json.dumps(record,ensure_ascii=False),
                                  stratum=row['label']))
        negatives.append(dict(row,review_id=opaque,base_continuation=base))
    for kind, rows in payloads.items():
        assert len({r['opaque_query_id'] for r in rows})==len(rows)
        write(root/f'{kind}_input.private.json',rows)
    view_sample = blind_csv(root/'views_human_review.private.csv',payloads['view'],SEED)
    neg_sample = blind_csv(root/'negatives_human_review.private.csv',payloads['negative'],SEED)

    old_cal = {images[e['owner']] for e in old['entries'] if e['owner'] in s0_cal}
    old_eval = {images[e['owner']] for e in old['entries'] if e['owner'] not in s0_cal}
    if old_cal & old_eval:
        raise ValueError('Earlier pilot calibration and evaluation share images')
    supported = defaultdict(set)
    for r in negatives:
        if r['label']=='independent':
            supported[r['owner']].add(r['role'])
    pool = sorted({images[o] for o,roles in supported.items() if {'train','holdout'}<=roles} - old_cal - old_eval)
    random.Random(SEED).shuffle(pool)
    new_cal = set(pool[:max(0,19-len(old_cal))])
    write(root/'expansion.private.json',dict(status='DRAFT_REVIEW_PENDING',sources={o:sources[o] for o in sorted(allowed)},
        views=usable,negatives=negatives,protected_image_sha256=sorted(protector),
        calibration_images=sorted(old_cal|new_cal),prior_evaluation_images=sorted(old_eval)))
    aggregate = dict(requested=120,protected_exclusions=len(excluded),unique_images=len({images[o] for o in allowed}),
        official_n_gram_index=sum(len(x) for x in frozen.values()),candidates={k:len(v) for k,v in payloads.items()},
        views_n_gram_excluded={f'{a}/{b}':n for (a,b),n in sorted(flags.items())},
        view_sample=view_sample,negative_sample=neg_sample,prior_independent_roles={
            'train':sum('train' in p for p in supported.values()),
            'both':sum({'train','holdout'}<=p for p in supported.values())},
        provisional_calibration_groups=len(old_cal|new_cal),human_reviewed=0)
    write(root/'prepared_summary.json',aggregate)
    print(json.dumps(aggregate,ensure_ascii=False))


if __name__=='__main__':
    main(*(Path(v) for v in sys.argv[1:]))
