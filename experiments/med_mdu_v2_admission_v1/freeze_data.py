"""Prepare audited, private MDU-v2 input manifests without loading a model."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import csv
import json
import random
import re
import sys
import unicodedata


SEED = 20261127
TECHNIQUE_SOURCE = {
    'was this a contrast ct or a non contrast ct',
    'what is the imaging modality',
    'what is better identified on a pa cxr vs a lateral cxr',
    'in what position was the patient in when this image was taken',
    'what type of imaging modality is used to acquire the above image',
    'what can be typically visualized in this plane',
    'what imaging modality is seen here',
}
FAMILIES = ('ordinary', 'declarative', 'modal', 'context')
MODALITY = r'(?:ct(?: scan)?|an ct scan|mri(?: scan| image)?|an mri(?: scan| image)?|x ray)'
TECHNIQUE = {
    'modality': re.compile(r'^(?:is (?:this|the image)(?:(?: a| an)? '+MODALITY+r'| image taken via (?:mri|x ray))|'
      r'is this (?:a ct or an mri|an mri or a ct scan)|'
      r'was a ct or mri used to take the above image|'
      r'(?:what (?:is the )?)?(?:imaging|image) modality(?: is)?(?: used to (?:take this image|acquire this picture)| was used)?|'
      r'what modality (?:is used to take this image|is shown in this image|is used)|'
      r'what type of image modality is used|which type of modality is shown about this image mri ct or x ray)$'),
    'plane': re.compile(
      r'^(?:is (?:this|this image in|the image above oriented in) '
      r'(?:(?:a|the) )?(?:(?:axial|transverse|coronal|sagittal|saggital)(?: brain)? (?:plane|section))|'
      r'(?:in )?what (?:image )?plane (?:is (?:this(?: image scanned)?|the ct scan|the above taken|the image above captured|this image in)|'
      r'was (?:this image taken(?: in)?|used))|'
      r'what is the (?:scanning plane of this image|plane)|what scanning plane does this image belong to|'
      r'(?:the|this) image is taken in what plane|which plane is the image scanned(?: transverse plane or coronal plane)?|'
      r'what plane is the above image acquired in)$'),
    'sequence': re.compile(
      r'^(?:what is the mr weighting in this image|which kind of weighting is this image generated with t1 or t2|'
      r'what type of mri is (?:this|shown in this image|the above image)|'
      r'what is the (?:mri modality depicted in this image|modality of this mri)|mri imaging modality used|'
      r'is this (?:an mri image with t[12] weighted|a t[12] weighted mri(?: image)?|'
      r'a t1 weighted(?: or t2 weighted mri image| t2 weighted or flair image)))$'),
}
ACQUISITION_REFERENCE = {
    'modality': {'ct', 'mri', 'mr', 'x ray', 'xray', 'yes', 'no'},
    'plane': {'axial', 'axial plane', 'transverse', 'transverse plane', 'coronal',
              'coronal plane', 'sagittal', 'sagittal plane', 'yes', 'no'},
    'sequence': {'t1', 't2', 'dwi', 'flair', 'yes', 'no'},
}
PRESENCE = re.compile(r'^(?:can (brain edema|brain tumor) be observed in the picture|'
                      r'is there evidence of (?:a )?(brain edema|brain tumor)|'
                      r'does this image show (brain edema|brain non enhancing tumor|brain enhancing tumor|brain tumor))$')
LIST_SCOPE = {'what diseases are included in the picture', 'what abnormal findings are present in this image'}
BROAD_UNSPECIFIED = {'what is abnormal in this image', 'describe the pathology', 'pathology seen in this image',
                     'what pathology is seen in this image', 'what is the pathology'}


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)


@lru_cache(maxsize=None)
def norm(s):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', s).casefold()))


def scope(question):
    q = norm(question)
    if q in LIST_SCOPE:
        return 'whole_image_findings'
    if q in TECHNIQUE_SOURCE or q in BROAD_UNSPECIFIED or not q:
        return 'other_or_ambiguous'
    return 'specific_finding_or_property'


def classify_pair(source, candidate, base):
    q, ref = norm(candidate['question']), norm(candidate['gold_answer'])
    nq = norm(source['question'])
    attrs = [k for k, pattern in TECHNIQUE.items() if pattern.fullmatch(q)]
    if attrs:
        assert len(attrs) == 1
        attr = attrs[0]
        if nq not in TECHNIQUE_SOURCE and ref in ACQUISITION_REFERENCE[attr]:
            return 'independent', f'acquisition_{attr}', attr, None
        return 'unresolved', 'acquisition_edit_or_reference_not_clear', None, None

    match = PRESENCE.fullmatch(q)
    if match and (nq == 'what diseases are included in the picture' or
                  nq.startswith('what disease is are shown')):
        concept = next(x for x in match.groups() if x)
        named = {norm(x) for x in re.split(r'[,;，、]', source['reference'])}
        if concept == 'brain tumor':
            compatible = bool(named & {'brain tumor','brain enhancing tumor','brain non enhancing tumor'})
        else:
            compatible = concept in named
        if compatible and ref == 'yes':
            answer = norm(base['model_answer_raw'])
            polar = answer.split()[:1]
            hedged = re.search(r'\b(?:however|unclear|cannot|difficult|uncertain|unable|unsure|maybe)\b', answer)
            if not hedged and polar == ['yes'] and not re.search(r'\b(?:no|not)\b', answer[3:]):
                return 'related_aligned', 'entailed_presence_and_explicit_base_affirmation', None, 'yes'
            if not hedged and polar == ['no'] and not re.search(r'\byes\b', answer[2:]):
                return 'related_conflict', 'entailed_presence_and_explicit_base_negation', None, 'yes'
            return 'unresolved', 'entailed_presence_with_ambiguous_base_polarity', None, 'yes'
    return 'unresolved', 'no_reliable_independence_or_direct_entailment_rule', None, None


def review_view(row, sources, templates, plain):
    native = sources[row['owner']]['source_rows'][0]
    q = native['question'].strip()
    assert row['reference'] == native['reference'] and row['image_group'] == native['image_sha256']
    if row['family'] == 'ordinary':
        i = int(row['template'].split('_')[-1])
        pattern, replacement = plain[i]
        expected = re.sub(pattern, replacement, q, count=1, flags=re.I)
        assert expected != q
    else:
        family, role, en, zh = templates[row['template']]
        assert (family, role) == (row['family'], row['role'])
        expected = (zh if row['language']=='zh' else en).format(q=q)
        assert expected.count('“'+q+'”')==1
    assert row['question'] == expected and norm(row['question']) != norm(q)
    return row


def main(root, frozen_baseline, t2g_manifest):
    data = load(root/'inventory.private.json')
    frozen = load(frozen_baseline)['roles']['roles']
    sources = {int(k):v for k,v in data['sources'].items()}
    bases = load(root/'base_continuations.private.json')
    assert len(sources)==150 and len(data['candidates'])==1276 and len(bases)==1130
    assert {r['generation_config_sha256'] for r in bases.values()}=={
        v['generation_config_sha256'] for v in data['verdict_archive'].values()}
    actual_technique={norm(v['source_rows'][0]['question']) for v in sources.values()
                      if norm(v['source_rows'][0]['question']) in TECHNIQUE_SOURCE}
    assert actual_technique == TECHNIQUE_SOURCE

    labels = []
    image_types = defaultdict(set)
    for row in data['candidates']:
        source = sources[row['owner']]['source_rows'][0]
        image = row['image_sha256']
        assert image==source['image_sha256']
        b = bases[row['query_id']]
        status, reason, attr, edited_answer = classify_pair(source, row, b)
        if status == 'independent':
            image_types[image].add(attr)
        labels.append({**row,'label':status,'reason':reason,'attribute':attr,
                       'entailed_answer':edited_answer,'source_scope':scope(source['question'])})
    rng=random.Random(SEED)
    reserved={}
    for image, attrs in sorted(image_types.items()):
        if len(attrs)>=2:
            reserved[image]=rng.choice(sorted(attrs))
    roles_by_question={}
    by_owner=defaultdict(lambda:defaultdict(list))
    for r in labels:
        if r['label']=='independent':
            r['role']='holdout' if reserved.get(r['image_sha256'])==r['attribute'] else 'train'
            key=(r['image_sha256'], norm(r['question']))
            assert roles_by_question.setdefault(key,r['role'])==r['role']
            by_owner[r['owner']][r['role']].append(r)
        else:
            r['role']='consistency_probe' if r['label'].startswith('related_') else 'excluded_unresolved'
    save(root/'relation_labels.private.json',labels)

    lock=load(root/'template_lock.json')
    views=load(root/'view_candidates.private.json')
    reviewed=[review_view(r,sources,lock['templates'],lock['plain_rules']) for r in views]
    ancestor={(r['sha'],norm(r['question'])) for r in data['frozen']}
    assert not {(r['image_group'],norm(r['question'])) for r in reviewed}&ancestor
    templates=lock['templates']
    by_roles=defaultdict(set)
    for tid,(family,role,en,zh) in templates.items():
        by_roles[role].update((norm(en.split('{q}')[0]),norm(zh.split('{q}')[0])))
    by_id={r['query']:r['question'] for r in data['frozen']}
    indirect=[norm(by_id[r['query']]) for r in
         map(json.loads,t2g_manifest.open()) if r['variant']=='negation_uncertainty']
    new_modal_prefixes={norm(s.split('{q}')[0]) for k,(_,_,en,zh) in templates.items()
                        if 'modal' in k for s in (en,zh)}
    assert len(indirect)==141 and not any(q.startswith(p) for q in indirect for p in new_modal_prefixes)
    assert not (by_roles['train'] | by_roles['validation']) & (by_roles['holdout_modal'] | by_roles['holdout_context'])
    assert not any('alone' in en.lower() or '只依据' in zh for family, _, en, zh in templates.values()
                   if family=='context')
    sample=load(root/'blind_view_sample.private.json')
    for r in sample:
        check={k:v for k,v in r.items() if k not in ('source','number')}
        assert check in reviewed and r['source']==sources[r['owner']]['source_rows'][0]['question']
    with (root/'blind_view_review.tsv').open('x',newline='') as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerow(('number','original_question','modified_question','original_reference','verdict','reason'))
        for r in sample:
            writer.writerow((r['number'],r['source'],r['question'],r['reference'],'',''))

    by_view_owner=defaultdict(list)
    for r in reviewed:
        by_view_owner[r['owner']].append(r)
    eligible=[o for o,roles in by_owner.items()
              if roles['train'] and roles['holdout'] and
                 all(any(r['family']==f and r['role']=='train' for r in by_view_owner[o])
                     for f in ('declarative','modal','context'))]
    options=defaultdict(list)
    for o in eligible:options[sources[o]['source_rows'][0]['image_sha256']].append(o)
    order=sorted(options)
    rng.shuffle(order)
    panel=[]
    used=set()
    for scope_name, target in [('whole_image_findings',6),('specific_finding_or_property',4)]:
        for image in order:
            if target==0: break
            choices=[o for o in options[image] if frozen[str(o)]['T2L'] and
                     scope(sources[o]['source_rows'][0]['question'])==scope_name]
            if image not in used and choices:
                panel.append(rng.choice(choices));used.add(image);target-=1
    for image in order:
        if len(panel)>=30: break
        if image not in used:
            panel.append(rng.choice(options[image]));used.add(image)
    assert len(panel)==30 and sum(bool(frozen[str(o)]['T2L']) for o in panel)>=10
    output=[]
    for o in panel:
        source=sources[o]
        active=by_view_owner[o][:]
        if not any(r['family']=='ordinary' and r['role']=='train' for r in active):
            row=source['source_rows'][1]
            active.append({'family':'ordinary','role':'train','template':'historical_fit1',
                           'question':row['question'],'reference':row['reference']})
        for role in ('train','holdout'):
            for n in by_owner[o][role]:
                n['base_continuation']=bases[n['query_id']]
        output.append({'owner':o,'source':source,'scope':scope(source['source_rows'][0]['question']),
                       'views':active,'negative':dict(by_owner[o]),'frozen_support':frozen[str(o)]})
    save(root/'pilot_inputs.private.json',{'status':'MACHINE_READY_BLIND_REVIEW_PENDING','seed':SEED,'entries':output})
    summary={'status':'MACHINE_READY_BLIND_REVIEW_PENDING','source_edits':150,
        'labels':dict(Counter(r['label'] for r in labels)),
        'independent_roles':dict(Counter(r['role'] for r in labels if r['label']=='independent')),
        'independent_with_two_roles':len(eligible),'eligible_unique_images':len(options),
        'pilot_edits':len(panel),'pilot_images':len(used),
        'pilot_scope':dict(Counter(e['scope'] for e in output)),
        'pilot_with_frozen_t2l':sum(bool(e['frozen_support']['T2L']) for e in output),
        'pilot_frozen_t2l_events':sum(len(e['frozen_support']['T2L']) for e in output),
        'pilot_negatives':{role:sum(len(e['negative'][role]) for e in output) for role in ('train','holdout')},
        'pilot_views':dict(Counter(r['role'] for e in output for r in e['views'])),
        'pilot_ordinary_fit_fallback':sum(any(v['template']=='historical_fit1' for v in e['views']) for e in output),
        'view_mechanical_checks':len(reviewed),'blind_sample':len(sample),'human_blind_reviewed':0,
        'base_bound':len(bases),'training':0,'new_generation':0,'new_scoring':0}
    save(root/'prepared_summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':
    main(*(Path(x) for x in sys.argv[1:]))
