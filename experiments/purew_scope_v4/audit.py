"""Read historical failures and candidate annotations; never create training labels."""
import collections
import json
import os
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text())


def image_id(text):
    ids = set(re.findall(r'xmlab\d+|synpic\d+', str(text).lower()))
    return next(iter(ids)) if len(ids) == 1 else None


def normalized(text):
    return ' '.join(str(text).lower().split())


def audit():
    root = Path(os.environ['HISTORICAL_ROOT'])
    data = Path(os.environ['DATA_ROOT']) / 'knowledge_editing/data/m3bench'
    pool = read(root / 'medtrace-scope-anchor-audit-v1-20261009/run/private/ELIGIBLE_SOURCE_POOL.json')
    r3 = root / 'purew-scope-v3-20261010/run'
    scores = read(r3 / 'private/SEMANTIC.json')
    final = []
    historical_images = set()
    for score in scores:
        output = read(score['path'])
        item = output['binding']['input']
        ident = image_id(item['image_path'])
        if ident:
            historical_images.add((item['dataset'], ident))
        if output['stage'] == 'C_FULL_FIXED':
            final.append(dict(output=output, correct=score['correct'], path=score['path']))
    natives = {r['output']['owner']: r['output']['binding']['input'] for r in final if r['output']['role'] == 'NATIVE'}
    t2g = [r for r in final if r['output']['role'] == 'T2G']
    assert len(t2g) == 32 and len(natives) == 8
    def same(a, b):
        if 'image_sha256' in a and 'image_sha256' in b:
            return a['image_sha256'] == b['image_sha256']
        def path(item):
            ident = image_id(item['image_path'])
            if ident and item['dataset'] == 'VQA-RAD':
                return data / f'VQA-RAD/images/{ident}.jpg'
            if ident and item['dataset'] == 'SLAKE':
                return data / f'SLAKE/imgs/{ident}/source.jpg'
            return Path(item['image_path'])
        return path(a).samefile(path(b))
    failures = [r for r in t2g if r['correct'] == 0]
    assert len(failures) == 7 and all(same(r['output']['binding']['input'], natives[r['output']['owner']]) for r in failures)
    vqa = read(data / 'VQA-RAD/VQA_RAD Dataset Public.json')
    slake = {split: read(data / f'SLAKE/{split}.json') for split in ('train', 'validation', 'test')}
    source_checks = []
    for owner in sorted({r['output']['owner'] for r in failures}):
        native = natives[owner]
        if native['dataset'] == 'VQA-RAD':
            matches = [('public_annotation', row) for row in vqa if image_id(row['image_name']) == image_id(native['image_path']) and normalized(row['question']) == normalized(native['question'])]
        else:
            matches = [(split, row) for split, rows in slake.items() for row in rows if image_id(row['img_name']) == image_id(native['image_path']) and normalized(row['question']) == normalized(native['question'])]
        assert len(matches) == 1, (owner, len(matches))
        split, row = matches[0]
        assert normalized(row['answer']) == normalized(native['reference']), owner
        source_checks.append(dict(owner=owner, split=split, source=row, original_input=native, binding_match=True))
    known_cases = {row['image_case_url'] for row in vqa if ('VQA-RAD', image_id(row['image_name'])) in historical_images} - {None, '', 'NULL'}
    sources = {str(data / 'SLAKE/train.json'): {str(r['qid']): r for r in slake['train']}, str(data / 'VQA-RAD/VQA_RAD Dataset Public.json'): {str(r['qid']): r for r in vqa}}
    roles = read(root / 'scope-m3bench-exposure-audit-20261001/run/private/IDENTITY_ROLE_LEDGER.json')
    role_index = collections.defaultdict(set)
    for row in roles:
        role_index[row['dataset'], row['image_id']].update(row['roles'])
    forbidden = {'PROTECTED_SPLIT', 'FORMAL_OR_RESERVED_SOURCE', 'CAL_CHECK_SUPPORT_RESERVED', 'FROZEN_FIT_SOURCE'}
    eligible = []
    reasons = collections.Counter()
    for row in pool:
        raw = sources[row['source_file']][row['qid']]
        assert raw == row['raw_source'] and normalized(raw['question']) == normalized(row['question']) and normalized(raw['answer']) == normalized(row['answer'])
        assert Path(row['image_path']).is_file()
        key = row['dataset'], row['image_id']
        why = []
        if not role_index[key] or role_index[key] & forbidden:
            why.append('ROLE_CONFLICT_OR_UNKNOWN')
        if key in historical_images or raw.get('image_case_url') in known_cases:
            why.append('PREVIOUS_ROUND_IMAGE_OR_KNOWN_CASE')
        if row['dataset'] == 'VQA-RAD' and (raw['phrase_type'] != 'freeform' or raw['evaluation'] != 'evaluated'):
            why.append('SOURCE_ANNOTATION_RESTRICTION')
        if why:
            reasons.update(why)
        else:
            eligible.append(row)
    open_rows = [r for r in eligible if r['raw_source']['answer_type'] == 'OPEN']
    lexical = collections.defaultdict(set)
    for row in open_rows:
        lexical[normalized(row['answer'])].add((row['dataset'], row['image_id']))
    # Answer equality is only a candidate search, never scope proof or a positive label.
    with_two = [r for r in open_rows if len(lexical[normalized(r['answer'])]) >= 3]
    same_count = sum(same(r['output']['binding']['input'], natives[r['output']['owner']]) for r in t2g)
    public = dict(status='DATA_QUALIFICATION_PENDING', old_T2G=dict(n=len(t2g), same_source_image=same_count, different_source_image=len(t2g)-same_count, failed=len(failures), failed_same_source_image=len(failures), original_source_bindings_verified=len(source_checks), historical_scores_unchanged=True),
        source_pool=dict(previously_screened_QA=len(pool), previously_screened_images=len({(r['dataset'], r['image_id']) for r in pool}), current_role_eligible_QA=len(eligible), current_role_eligible_images=len({(r['dataset'], r['image_id']) for r in eligible}), additional_rejections=dict(reasons), dataset_counts=dict(collections.Counter(r['dataset'] for r in eligible)), open_QA=len(open_rows), open_QA_with_two_other_literal_answer_images=len(with_two), literal_answer_groups_with_three_images=sum(len(v) >= 3 for v in lexical.values()), complete_scope_reviewed_edit_units=0, patient_independence='UNKNOWN'),
        source_provenance_exact=True, new_training_labels=0, new_GPU_updates=0, model_calls=0, new_Judge=0, interpretation='No complete data bundle admitted. Lexical coverage is a screen, not an impossibility proof; boolean or semantically equivalent source-supported facts need separate scope review.')
    private = dict(failures=failures, original_source_checks=source_checks, role_eligible_candidates=eligible, provenance_checked=True)
    return public, private


if __name__ == '__main__':
    public, private = audit()
    out = Path(os.environ['AUDIT_OUTPUT'])
    for name, value in [('public/AUDIT.json', public), ('private/AUDIT_PACKET.json', private)]:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(public, ensure_ascii=False))
