"""Prospective matched-polarity source screen. No model scores are inputs."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import re
import zipfile

TARGETS = ['apical pleural thickening', 'pleural effusion', 'laminar atelectasis',
           'atelectasis', 'costophrenic angle blunting', 'infiltrates', 'nodule',
           'calcified granuloma', 'callus rib fracture', 'interstitial pattern',
           'vascular hilar enlargement', 'diaphragmatic eventration']
PROTECT = ['pleural effusion', 'pneumothorax', 'consolidation', 'nodule', 'infiltrates']
NOUNS = {'pleural effusion':'pleural effusion', 'pneumothorax':'pneumothorax',
         'consolidation':'pulmonary consolidation', 'nodule':'nodules',
         'infiltrates':'pulmonary infiltrates'}
UNCERTAIN = re.compile(r'suggest|probab|possib|suspic|cannot|may |could |appears|borderline|at the limit|improv|resolv|disappear|newly|previous|prior|interval|increas.*compared', re.I)
NEG_LIMIT = re.compile(r'\b(significant|clear|evident|other|possible|suggestive|current|moment|left|right|previous|prior|suspicious|relevant)\b|pathological|infectious|contusion|pneumonic process',re.I)
PATTERNS = {'pleural effusion':r'pleural effusion', 'pneumothorax':r'pneumothorax',
            'consolidation':r'consolidat', 'nodule':r'nodul', 'infiltrates':r'infiltrat'}


def support():
    path = Path(__file__).resolve().parents[1]/'relation_source_v97/source_audit.py'
    spec = importlib.util.spec_from_file_location('source_support', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def positive(f):
    return f['abnormal'] and f.get('boxes') and len(f.get('labels', [])) == 1 and not UNCERTAIN.search(f['sentence_en'])


def negatives(record):
    present = set(label for f in record['findings'] if f['abnormal'] for label in f.get('labels', []))
    found = {}
    for label in PROTECT:
        if label in present or (label == 'consolidation' and 'alveolar pattern' in present):
            continue
        for f in record['findings']:
            sentence = f['sentence_en']
            if f['abnormal'] or NEG_LIMIT.search(sentence) or UNCERTAIN.search(sentence):
                continue
            if not re.search(r'\bno\b|\bwithout\b|do not', sentence, re.I):
                continue
            if re.search(PATTERNS[label], sentence, re.I):
                found[label] = f
                break
    return found


def freeze(source_root, output, previous):
    source_root, output, previous = map(Path, [source_root, output, previous])
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'CANDIDATES_FROZEN.json').exists(), 'Already frozen'
    mod = support()
    excluded = {r['patient'] for r in json.loads((previous/'CANDIDATES_FROZEN.json').read_text())}
    with zipfile.ZipFile(source_root/'metadata/master_table.csv.zip') as z:
        table = list(csv.DictReader(io.TextIOWrapper(z.open('master_table.csv'))))
    meta = {r['ImageID']:r for r in table if r['split']=='train'}
    all_records = json.loads((source_root/'metadata/grounded_reports_20240819.json').read_text())
    records = sorted([r for r in all_records if r['ImageID'] in meta and meta[r['ImageID']]['PatientID'] not in excluded],
                     key=lambda r:(meta[r['ImageID']]['PatientID'],r['StudyID'],r['ImageID']))
    candidates = {k:[] for k in ['left','right','bilateral']}
    negatives_by_image = {r['ImageID']:negatives(r) for r in records}
    for r in records:
        neg = negatives_by_image[r['ImageID']]
        if not neg:
            continue
        for f in r['findings']:
            if not positive(f) or f['labels'][0] not in TARGETS:
                continue
            laterality = mod.side(f)
            if laterality is None:
                continue
            same = [g for g in r['findings'] if set(g.get('labels',[])) & set(f['labels'])]
            if any(mod.side(g) != laterality for g in same):
                continue
            label = next((k for k in PROTECT if k in neg and k != f['labels'][0]), None)
            if label:
                candidates[laterality].append(dict(patient=meta[r['ImageID']]['PatientID'],
                    image=r['ImageID'], study=r['StudyID'], side=laterality,
                    target=f, protection_label=label, negative=neg[label], source_record=r))
                break
    selected, used = [], set(excluded)
    for side in candidates:
        count = 0
        for r in candidates[side]:
            if r['patient'] in used:
                continue
            used.add(r['patient']);selected.append(dict(r,group=f'n{len(selected)+1:02d}'));count += 1
            if count == 8:
                break
    for r in selected:
        label = r['protection_label']
        r['positive_control'] = None
        for c in records:
            patient = meta[c['ImageID']]['PatientID']
            if patient in used or label in negatives_by_image[c['ImageID']]:
                continue
            f = next((f for f in c['findings'] if positive(f) and f['labels']==[label]),None)
            if f is not None:
                r['positive_control'] = dict(patient=patient,image=c['ImageID'],study=c['StudyID'],finding=f,source_record=c)
                used.add(patient)
                break
    audit = dict(official_train_images=len(meta), prior_patients_excluded=len(excluded),
        eligible_source_counts={k:len(v) for k,v in candidates.items()},
        frozen_candidates=len(selected), side_counts=dict((k,sum(r['side']==k for r in selected)) for k in candidates),
        matched_controls=sum(r['positive_control'] is not None for r in selected), model_answers_seen=0)
    (output/'CANDIDATES_FROZEN.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    (output/'SOURCE_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit))
    for r in selected:
        print(r['group'],r['side'],r['target']['labels'][0],r['target']['sentence_en'],
              '| NEG',r['protection_label'],r['negative']['sentence_en'],
              '| POS',r['positive_control']['finding']['sentence_en'] if r['positive_control'] else 'MISSING')


def materialize(source_root, output):
    from PIL import Image
    import numpy as np
    source_root, output = Path(source_root), Path(output)
    review = json.loads((output/'SOURCE_REVIEW_FROZEN.json').read_text())
    selected = json.loads((output/'CANDIDATES_FROZEN.json').read_text())
    mod = support()
    archive = zipfile.ZipFile(mod.Volumes(sorted((source_root/'Padchest_GR_files').glob('PadChest_GR.zip.*'))))
    members = {Path(i.filename).name:i.filename for i in archive.infolist() if not i.is_dir()}
    (output/'images').mkdir(exist_ok=True)
    rows = []
    for r in selected:
        if r['group'] in review['excluded'] or r['positive_control'] is None:
            continue
        for suffix, record in [('negative',r),('positive',r['positive_control'])]:
            with Image.open(io.BytesIO(archive.read(members[record['image']]))) as im:
                im.load();pixels=np.asarray(im,dtype=np.float64)
            Image.fromarray(np.rint(pixels/257).astype('uint8')).convert('RGB').save(output/'images'/f"{r['group']}_{suffix}.png")
        noun = NOUNS[r['protection_label']]
        direct = ('Are there ' if r['protection_label'] in ['nodule','infiltrates'] else 'Is there ')+noun+' in this chest X-ray?'
        reverse = 'Is this chest X-ray free of '+noun+'?'
        target = f"On which side is the {r['target']['labels'][0]} in this chest X-ray: left, right, or bilateral?"
        questions = [('target',target,r['side'],'negative'),
            ('brief_control',target+' Answer briefly with only the side.',r['side'],'negative'),
            ('negative_direct',direct,'No','negative'),('negative_reverse',reverse,'Yes','negative'),
            ('positive_direct',direct,'Yes','positive'),('positive_reverse',reverse,'No','positive')]
        for role,q,ref,suffix in questions:
            rows.append(dict(qid=980000+len(rows),query_id=r['group']+'_'+role,source_group=r['group'],
                role=role,question=q,reference=ref,image_path=str((output/'images'/f"{r['group']}_{suffix}.png").resolve()),
                audit_role='SOURCE_QUALIFICATION',content_type='Laterality' if role in ['target','brief_control'] else 'Presence',
                protection_label=r['protection_label'],target_label=r['target']['labels'][0]))
    (output/'SOURCE_ROWS.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print('MATERIALIZED',len(rows),'queries',len(list((output/'images').glob('*.png'))),'images')
