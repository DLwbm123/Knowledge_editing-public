"""Bounded source-only H supplementation; clinical relationships stay UNKNOWN."""
import collections
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

entry = Path(os.environ.get('P0_ENTRY', Path(__file__).with_name('p0_admission.py')))
spec = importlib.util.spec_from_file_location('p0', entry)
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)


def family(question):
    q = p0.normalized(question).rstrip('?.')
    # Explicit template equivalence only. No removal of clinical qualifiers.
    if re.fullmatch(r'(what|which) (?:type of )?(?:imaging |image )?modality(?: is (?:used to (?:acquire|take)|seen in|seen|used in))?(?: the| this| here| above| to acquire| to take| used)?(?: above)?(?: image)?(?: here)?', q):
        return 'modality'
    if q in {'what is the imaging modality', 'what imaging modality is seen here',
             'what type of imaging modality is used to acquire the above image',
             'what image modality is this', 'what type of image modality is this',
             'which image modality is this', 'what imaging modality was used',
             'what modality is used to take this image', 'what type of modality is this image',
             'what is the modality of this image', 'what imaging modality is used in this image'}:
        return 'modality'
    if re.fullmatch(r'which organs/organ (?:is part of the digestive system in this image|in the picture belong to the digestive system)', q):
        return 'digestive_organs'
    if q in {'what diseases are included in the picture', 'what diseases are shown in this image'}:
        return 'diseases_whole_image'
    if q in {'what abnormal findings are present in this image', 'what pathology is seen in this image',
             'what is the pathology', 'what is abnormal in this image', 'what is wrong with the patient\'s scan',
             'describe the pathology', 'how would you describe the pathology seen in the above image',
             'pathology seen in this image'}:
        return 'pathology_whole_image'
    if q in {'where is the lesion located', 'where is the lesion', 'the lesion is located where in this image',
             'where on the image is the lesion'}:
        return 'location_single_lesion'
    if q in {'where is the mass located', 'where is the mass', 'what is the location of the mass'}:
        return 'location_single_mass'
    if q in {'where is the abnormality', 'where is the abnormality in this image',
             'where is the abnormality located in this image', 'where is the abnormality located'}:
        return 'location_abnormality'
    match = re.fullmatch(r'what is the (?:effect|function) of the (organ|tissue) on the (center|top|left|right) of this (?:picture|image)', q)
    if match:
        return 'function:' + ':'.join(match.groups())
    match = re.fullmatch(r'what is the (?:effect|function) of the (top|left|right) (organ|tissue) in this (?:picture|image)', q)
    if match:
        return 'function:' + ':'.join(reversed(match.groups()))
    return 'exact:' + q


def modality(answer):
    a = p0.normalized(answer)
    classes = set()
    if re.search(r'\b(?:mri|mr|magnetic resonance)\b', a): classes.add('MRI')
    if re.search(r'\b(?:ct|computed tomography)\b', a): classes.add('CT')
    if re.search(r'\b(?:x-ray|xray|x ray|radiograph)\b', a): classes.add('X-ray')
    return next(iter(classes)) if len(classes) == 1 else None


def proof(native, row):
    if family(native['question']) == family(row['question']) == 'modality':
        a, b = modality(native['reference']), modality(row['answer'])
        if a and b and a != b:
            return dict(kind='SOURCE_VERIFIED_BENCHMARK_MODALITY_CONTRAST', native_class=a,
                        source_class=b, same_proposition='image-conditioned imaging modality',
                        out_of_scope='A modality label for the native image cannot replace the incompatible original modality of the source image',
                        validation='original source-row binding and mutually exclusive major acquisition classes',
                        reviewer='Codex source audit with fixed logical type check', clinical_verification=False)
    return None


def selfcheck():
    assert family('What is the function of the top organ in this picture?') == family('What is the effect of the organ on the top of this image?')
    assert family('What is the function of the organ on the left of this picture?') != family('What is the effect of the black organ on the left of this picture?')
    assert family('Where is the mass?') != family('Where is the lesion?')
    assert family('What is the function of the tissue in the center of the organ in this picture?') != family('What is the function of the organ on the center of this image?')
    n = dict(question='What is the imaging modality?', reference='MRI - T2 weighted')
    assert proof(n, dict(question='Which image modality is this?', answer='CT'))
    assert not proof(n, dict(question='Which image modality is this?', answer='MR-FLAIR'))
    assert not proof(n, dict(question='Is contrast present?', answer='CT'))
    assert not proof(n, dict(question='Which image modality is this?', answer='CT or MRI'))


def main():
    start, cpu = time.time(), time.process_time()
    selfcheck()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    root = Path(os.environ['RUN_ROOT'])
    final_pass = os.environ.get('SUPPLEMENT_PASS') == 'final'
    out = root / 'private/source_supplement' / ('final' if final_pass else 'initial')
    manifest = json.loads((root / 'RUN_MANIFEST.json').read_text())
    paths = [root/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json',
             root/'private/legacy_stage17/ROLE_BOUNDARY_JOIN.json',
             Path(os.environ['SOURCE_QA_FILE']), Path(os.environ['SOURCE_ROLE_FILE']),
             Path(os.environ['SLAKE_TRAIN_FILE']), Path(os.environ['VQA_FILE'])]
    assert sum(p.stat().st_size for p in paths) <= 64*1024*1024
    ledger, join, rows, roles, slake, vqa = [json.loads(p.read_text()) for p in paths]
    assert ledger['freeze_id'] == 'fea41e32c8b7780f45b060192343f3bb11c2e43b2e046fd709db19df6c0eef92'
    assert len(ledger['main_T0']) == 146
    forbidden = {'PROTECTED_SPLIT','FORMAL_OR_RESERVED_SOURCE','CAL_CHECK_SUPPORT_RESERVED','FROZEN_FIT_SOURCE'}
    image_roles = collections.defaultdict(set)
    for row in roles: image_roles[(row['dataset'],row['image_id'])].update(row['roles'])
    banned = {p0.source_identity(q['dataset'],q['original_image_path']) for q in ledger['queries'].values()}
    banned |= {p0.source_identity(d,p) for d,p in join['reserved_source_groups']}
    banned |= {k for k,v in image_roles.items() if forbidden & v}
    if final_pass:
        banned |= {p0.source_identity('VQA-RAD',x['image_name']) for x in vqa
                   if x['phrase_type'].startswith('test_')}
    source_index = {('SLAKE',str(x['qid'])):x for x in slake}
    source_index.update({('VQA-RAD',str(x['qid'])):x for x in vqa})
    assert len(source_index) == len(slake)+len(vqa)
    banned_cases = {x.get('image_case_url') for x in vqa
                    if p0.source_identity('VQA-RAD',x['image_name']) in banned}
    banned_cases -= {None,'','NULL'}
    pool = []
    rejected_cases = set()
    for row in rows:
        key = (row['dataset'],row['image_id'])
        if key in banned or not image_roles[key] or not row['image_exists']: continue
        raw = source_index[(row['dataset'],str(row['qid']))]
        assert raw == row['raw_source']
        assert raw['question'] == row['question'] and str(raw['answer']) == row['answer']
        assert p0.source_identity(row['dataset'],raw.get('img_name',raw.get('image_name')))[1] == row['image_id']
        if row['dataset'] == 'VQA-RAD':
            if raw.get('image_case_url') in banned_cases:
                rejected_cases.add(row['image_id']); continue
            if raw['evaluation'] != 'evaluated': continue
            if final_pass and raw['phrase_type'] != 'freeform': continue
        if not Path(row['image_path']).is_file(): continue
        pool.append(dict(row, original_source_row=raw, global_source_roles=sorted(image_roles[key])))
    # Original Chinese train QA inherit the whole-image guard; translations are not generated.
    for raw in slake:
        if raw['q_lang'] != 'zh': continue
        key = p0.source_identity('SLAKE',raw['img_name'])
        if key in banned or not image_roles[key]: continue
        image = Path(os.environ['SLAKE_TRAIN_FILE']).parent / 'imgs' / raw['img_name']
        if image.is_file():
            pool.append(dict(dataset='SLAKE', image_id=key[1], qid=str(raw['qid']),
                             question=raw['question'],answer=raw['answer'],image_path=str(image),
                             source_file=os.environ['SLAKE_TRAIN_FILE'], original_source_row=raw,
                             global_source_roles=sorted(image_roles[key]),
                             source_QA_evaluation='ORIGINAL_DATASET_ANNOTATION'))
    task_map = {t['edit_id']:t for t in ledger['tasks']}
    proposals, fits, gaps = [], [], []
    by_family = collections.defaultdict(list)
    for row in pool: by_family[family(row['question'])].append(row)
    for position, eid in enumerate(ledger['main_T0'],1):
        assert time.time() < manifest['deadline_epoch'] and not (root/'STOP').exists()
        n = task_map[eid]['native']; seen = set(); matches=[]
        for row in sorted(by_family[family(n['question'])], key=lambda r:(r['dataset'],r['image_id'],r['qid'])):
            key=(row['dataset'],row['image_id'])
            if key == p0.source_identity(n['dataset'],n['original_image_path']) or key in seen: continue
            if p0.normalized(n['reference']) == p0.normalized(row['answer']): continue
            seen.add(key); matches.append(row)
        # Typed contradictions are the prescribed H target, not score-based selection.
        if family(n['question']) == 'modality':
            matches = [row for row in matches if proof(n,row)]
        accepted = 0
        for i,row in enumerate(matches[:4],1):
            evidence = proof(n,row)
            item=dict(review_id=f'SH{position:03d}-{i}',position=position,edit_id=eid,
                      native=n,source=row,template=family(n['question']),source_binding='PASS',
                      relation_label='H_SOURCE_VERIFIED' if evidence else 'UNKNOWN',
                      admitted_H_fit=bool(evidence),evidence=evidence,patient_independence='UNKNOWN',
                      clinical_verification=False, new_student_output_access=False)
            proposals.append(item)
            if evidence: fits.append(item); accepted+=1
        gaps.append(dict(position=position,edit_id=eid,template=family(n['question']),
                         available_source_images=len(matches),review_proposals=min(4,len(matches)),
                         source_verified_H_fit=accepted,
                         status='SOURCE_VERIFIED_BENCHMARK_H' if accepted else 'UNSUPPORTED_PENDING_EVIDENCE'))
    summary=dict(status='COMPLETE_PARTIAL_H_SUPPORT',N=146,
                 role_qualified_QA=len(pool),role_qualified_source_images=len({(x['dataset'],x['image_id']) for x in pool}),
                 QA_dataset_counts=dict(collections.Counter(x['dataset'] for x in pool)),
                 original_Chinese_QA_added=sum(x['dataset']=='SLAKE' and x['original_source_row']['q_lang']=='zh' for x in pool),
                 known_same_case_source_images_excluded=len(rejected_cases),
                 proposal_edits=sum(x['review_proposals']>0 for x in gaps),proposals=len(proposals),
                 source_verified_H_edits=sum(x['source_verified_H_fit']>0 for x in gaps),
                 source_verified_H_relations=len(fits),unknown_relations=len(proposals)-len(fits),
                 edits_without_verified_H=sum(x['source_verified_H_fit']==0 for x in gaps),
                 verification='SOURCE_ANNOTATION_AND_MAJOR_MODALITY_LOGIC_ONLY',
                 clinical_verified_H=0,patient_independence='UNKNOWN',
                 scientific_GPU_admission=all(x['source_verified_H_fit']>0 for x in gaps),
                 original_full_cohort_preserved=True,GPU_hours=0,new_generation=0,new_Judge=0)
    p0.write_new(out,'RELATION_REVIEW_PACKET.json',proposals)
    p0.write_new(out,'SOURCE_VERIFIED_H_FIT.json',fits)
    p0.write_new(out,'REMAINING_H_GAPS.json',gaps)
    public_prefix = 'H_SUPPLEMENT_FINAL' if final_pass else 'H_SUPPLEMENT'
    p0.write_new(root,'public/'+public_prefix+'_SUMMARY.json',summary)
    receipt=dict(status='COMPLETE',pid=os.getpid(),argv=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),
                 CUDA_VISIBLE_DEVICES='',CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,
                 input_bytes=sum(p.stat().st_size for p in paths),first_clock_preserved=True,
                 source_binding_and_negative_control_checks='PASS')
    assert receipt['CPU_seconds']<60
    p0.write_new(out,'EXECUTION_RECEIPT.json',receipt)
    p0.write_new(root,'public/'+public_prefix+'_EXECUTION.json',{k:v for k,v in receipt.items() if k not in {'pid','argv'}})
    print(json.dumps(summary))


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        selfcheck(); print('PASS: source template qualifiers and mutually exclusive modality checks')
    else:
        main()
