"""Bounded source-only candidate audit; no model, Judge or scope certification."""
import collections
import csv
import json
import os
import re
import sys
import time
import traceback
import unicodedata
from pathlib import Path

ROOT = Path(os.environ.get('RUN_ROOT', '/tmp'))
DATA = Path(os.environ.get('DATA_ROOT', '/tmp'))
HISTORY = Path(os.environ.get('HISTORY_ROOT', '/tmp'))
SOURCE = Path(os.environ.get('SOURCE_ROOT', '/tmp'))
STOPWORDS = set('what which where when why how does do is are was were the this that these those a an of in on at to for with by from image picture shown show shows patient medical treatment used use disease organ effect function prevent caused cause and or can could would should it its there have has be'.split())
ALLOW = set('TASKS_R2_LOCKED.json TASKS_R2.json TASKS_MATRIX.json AUXILIARY_POOL.json ROLE_AUDIT_INPUTS.json PRESSURE_CANDIDATES.json PRESSURE_VALIDATION_FROZEN.json PRESSURE_TEST_FROZEN.json LOCALITY_STRESS_HOLDOUT.json G_SUPPORTS.json G_SUPPORTS_REPAIRED.json CHECK_POS.json CHECK_NEG.json CAL_NEG.json CAL_PLUS.json CAL_ROUTE.json CAL_NATIVE_PARAPHRASES.json SCOPE_ROLE_MANIFEST.json CANDIDATE_POOL.json U_bg.json FRESH_TASKS.json'.split())
PRUNE = set('models adapters checkpoints keys generations judge judge_sol base source source_patch .git __pycache__ user_model_outputs base_predictions'.split())
IDS = re.compile(r'xmlab\d+|synpic\d+', re.I)


def canon(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def image_ids(value):
    return {s.lower() for s in IDS.findall(str(value))}


def content_words(question):
    return {w for w in re.findall(r'[a-z]+', canon(question)) if len(w) >= 4 and w not in STOPWORDS}


def grounded(triple):
    return isinstance(triple, list) and len(triple) == 3 and all(str(x).lower() not in {'vhead', 'ktail', 'head', 'tail', 'v', 'k'} for x in triple)


def write(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def selfcheck():
    assert image_ids('/imgs/xmlab123/source.jpg') == {'xmlab123'}
    assert image_ids('synpic420.jpg') == {'synpic420'}
    assert canon(' Ａ? ') == 'a?' and canon('a') != canon('a?')
    assert not grounded(['vhead', 'treat', 'ktail'])
    assert grounded(['entity', 'relation', 'value'])
    assert content_words('What treatment is used for glaucoma?') == {'glaucoma'}
    assert bucket({'question': 'Q?', 'answer': 'A', 'attribute': 'KG'}, {'question': 'q?', 'answer': 'a', 'attribute': 'KG'}) == 'EXACT_Q_EXACT_A'
    assert bucket({'question': 'Q?', 'answer': 'A', 'attribute': 'KG'}, {'question': 'q?', 'answer': 'b', 'attribute': 'KG'}) == 'EXACT_Q_DIFFERENT_A'
    assert bucket({'question': 'What treats glaucoma?', 'answer': 'A', 'attribute': 'KG'}, {'question': 'What causes glaucoma?', 'answer': 'B', 'attribute': 'KG'}) == 'ATTRIBUTE_CONTENT_OVERLAP'
    assert bucket({'question': 'Q?', 'answer': 'A', 'attribute': 'KG'}, {'question': 'Unrelated?', 'answer': 'A', 'attribute': 'KG'}) is None


def bucket(anchor, candidate):
    if canon(anchor['question']) == canon(candidate['question']):
        return 'EXACT_Q_EXACT_A' if canon(anchor['answer']) == canon(candidate['answer']) else 'EXACT_Q_DIFFERENT_A'
    if anchor['attribute'] == candidate['attribute'] and content_words(anchor['question']) & content_words(candidate['question']):
        return 'ATTRIBUTE_CONTENT_OVERLAP'
    return None


def main():
    started = time.time()
    selfcheck()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    argv = Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert not any(x in argv.lower() for x in ['wangbomin', 'knowledge_editing', 'scope'])
    write('PROCESS_RECEIPT.json', dict(pid=os.getpid(), argv=argv, process_stat=Path('/proc/self/stat').read_text(), RUN_ROOT=str(ROOT), CUDA_VISIBLE_DEVICES='', started_epoch=started))
    cfg = json.loads((ROOT / 'public/AUDIT_CONFIG.json').read_text())
    manifest = json.loads((ROOT / 'RUN_MANIFEST.json').read_text())
    assert time.time() < manifest['deadline_epoch']
    tasks = json.loads((SOURCE / 'private/TASKS_R2_LOCKED.json').read_text())['tasks']
    known_native = [t['native'] for t in tasks if t['order'] in cfg['pilot_orders']]
    recovery = json.loads((ROOT / 'INITIALIZATION_RECOVERY.json').read_text()) if (ROOT / 'INITIALIZATION_RECOVERY.json').exists() else {}
    failed_cpu = recovery.get('failed_CPU_wall_upper_bound', 0)
    used_bytes = recovery.get('failed_input_bytes_read', 0) + (SOURCE / 'private/TASKS_R2_LOCKED.json').stat().st_size
    receipts = [{'path': str(SOURCE / 'private/TASKS_R2_LOCKED.json'), 'bytes': (SOURCE / 'private/TASKS_R2_LOCKED.json').stat().st_size}]
    excluded = collections.defaultdict(set)

    def check():
        assert time.time() - started + failed_cpu < cfg['cpu_task_seconds_limit'] and time.time() < manifest['deadline_epoch']
        assert not (ROOT / 'STOP').exists()

    def load(path):
        nonlocal used_bytes
        check()
        size = path.stat().st_size
        assert size <= cfg['per_file_bytes_limit']
        used_bytes += size
        assert used_bytes <= cfg['input_bytes_limit']
        receipts.append({'path': str(path), 'bytes': size})
        if path.suffix == '.csv':
            with path.open(encoding='utf-8-sig') as f:
                return list(csv.DictReader(f))
        if path.suffix == '.jsonl':
            return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        return json.loads(path.read_text())

    def exclude(value, reason):
        # Historical text is never a retrieval pool; extract source/image identity only.
        if isinstance(value, dict):
            for key, item in value.items():
                if any(word in key.lower() for word in ['image', 'source_group', 'source_id', 'case_id', 'patient_id', 'study_id']):
                    for ident in image_ids(item):
                        excluded[ident].add(reason)
                if isinstance(item, (dict, list)):
                    exclude(item, reason)
        elif isinstance(value, list):
            for item in value:
                exclude(item, reason)

    files = []
    for path, dirs, names in os.walk(HISTORY):
        rel = Path(path).relative_to(HISTORY)
        dirs[:] = [d for d in dirs if d not in PRUNE and len(rel.parts) < 5 and Path(path, d).resolve() != ROOT.resolve()]
        for name in names:
            p = Path(path, name)
            core = 'm3bench_task_specific_core9_data' in p.parts and any(k in name for k in ['CANDIDATES', 'MANIFEST', 'INVENTORY', 'FORMAL_RECORDS', 'LINEAGE'])
            if not name.startswith('._') and (name in ALLOW or core) and p.suffix in {'.json', '.jsonl'}:
                files.append(p)
    assert any(p.name == 'CANDIDATE_POOL.json' for p in files), 'Original Stage17 exclusion inventory missing'
    assert any('core9' in str(p) for p in files), 'Historical core9 inventory missing'
    for path in sorted(set(files)):
        exclude(load(path), 'HISTORICAL_OR_PROTECTED_MANIFEST')
    metadata = sorted(p for p in (DATA / 'benchmark/metadata').rglob('*.csv') if not p.name.startswith('._'))
    assert metadata
    for path in metadata:
        exclude(load(path), 'M3BENCH_SELECTED_METADATA')
    protected_anchors = []
    for name in ['validation.json', 'test.json']:
        rows = load(DATA / 'SLAKE' / name)
        exclude(rows, 'SLAKE_PROTECTED_SPLIT')
        # Only already-exposed native QA can supply its own provenance; never a new pool.
        for row in rows:
            if row.get('q_lang') == 'en' and any(n['dataset'] == 'SLAKE' and image_ids(n['original_image_path']) == image_ids(row['img_name']) and n['question'] == row['question'] and canon(n['reference']) == canon(row['answer']) for n in known_native):
                protected_anchors.append((name, row))
    write('private/EXCLUSION_RECEIPTS.json', {'files': receipts, 'excluded_image_ids': {k: sorted(v) for k, v in excluded.items()}, 'coverage': 'Known migrated manifests plus all selected benchmark metadata; not certified all-host history'})

    source_rows = []
    for dataset, filename in [('SLAKE', 'train.json'), ('VQA-RAD', 'VQA_RAD Dataset Public.json')]:
        for row in load(DATA / dataset / filename):
            if dataset == 'SLAKE' and row.get('q_lang') != 'en':
                continue
            ids = image_ids(row.get('img_name', row.get('image_name', '')))
            assert len(ids) == 1, 'Ambiguous or unparsed source image'
            image_id = next(iter(ids))
            path = DATA / dataset / ('imgs/' + row['img_name'] if dataset == 'SLAKE' else 'images/' + row['image_name'])
            source_rows.append(dict(dataset=dataset, qid=str(row['qid']), image_id=image_id, question=row['question'], answer=str(row['answer']), attribute=row.get('content_type', row.get('question_type', 'UNKNOWN')), triple=row.get('triple'), source_QA_evaluation=row.get('evaluation', 'ORIGINAL_DATASET_ANNOTATION'), image_path=str(path), image_exists=path.is_file(), source_file=str(DATA / dataset / filename), raw_source=row, historical_reasons=sorted(excluded.get(image_id, set())), provenance_verified_scope=False, patient_independence='UNKNOWN'))
    eligible = [r for r in source_rows if not r['historical_reasons'] and r['image_exists'] and (r['dataset'] != 'VQA-RAD' or r['source_QA_evaluation'] == 'evaluated')]
    anchor_rows = list(source_rows)
    for filename, row in protected_anchors:
        path = DATA / 'SLAKE' / 'imgs' / row['img_name']
        image_id = next(iter(image_ids(row['img_name'])))
        anchor_rows.append(dict(dataset='SLAKE', qid=str(row['qid']), image_id=image_id, question=row['question'], answer=str(row['answer']), attribute=row['content_type'], triple=row.get('triple'), source_QA_evaluation='ORIGINAL_DATASET_ANNOTATION', image_path=str(path), image_exists=path.is_file(), source_file=str(DATA / 'SLAKE' / filename), historical_reasons=sorted(excluded[image_id]), native_provenance_only=True, raw_source=row, provenance_verified_scope=False, patient_independence='UNKNOWN'))
    assert all(not r.get('native_provenance_only') and r['image_id'] not in excluded for r in eligible)
    cards, packet, conditional, per_edit = [], [], [], []
    for order in cfg['pilot_orders']:
        task = next(t for t in tasks if t['order'] == order)
        native = task['native']
        nid = image_ids(native['original_image_path'])
        anchors = [r for r in anchor_rows if r['dataset'] == native['dataset'] and r['image_id'] in nid and r['question'] == native['question'] and canon(r['answer']) == canon(native['reference'])]
        assert len(anchors) == 1, 'Native QA source binding ambiguous/missing'
        anchor = anchors[0]
        cards.append(dict(edit_order=order, edit_id=task['canonical_edit_id'], anchor=anchor, scope_definition_status='DRAFT_REQUIRES_REVIEW', target_fact=None, entity=None, applicability_conditions=None, exclusions=None, reviewer=None, verified=False, template_is_grounded_fact=grounded(anchor['triple'])))
        groups = {k: [] for k in cfg['buckets']}
        for r in eligible:
            if r['dataset'] == anchor['dataset']:
                key = bucket(anchor, r)
                if key:
                    groups[key].append(r)
        for key, rows in groups.items():
            # Stable source-ID order; no model features, answers from formal panels or output scores.
            rows.sort(key=lambda r: (r['image_id'], r['qid']))
            chosen = rows[:cfg['maximum_per_bucket']]
            for i, r in enumerate(chosen):
                packet.append(dict(review_id=f'R{order}-{key}-{i+1}', edit_order=order, candidate=r, retrieval_bucket=key, proposed_scope=None, scope_label='UNKNOWN', data_role='REVIEW_ONLY', admitted_CAL=False, reviewer_1=None, reviewer_2=None, evidence=None, adjudication=None))
        seen = {canon(native['question'])} | {canon(q) for k in ['fit_questions', 'semantic_fit_questions'] for q in task[k]}
        seen |= {canon(r['question']) for r in task['official_evaluation_full']}
        checkpos = json.loads((SOURCE / 'private/CHECK_POS.json').read_text())
        seen |= {canon(r['question']) for r in checkpos.get(str(order), [])}
        same = [r for r in source_rows if r['dataset'] == anchor['dataset'] and r['image_id'] == anchor['image_id'] and canon(r['question']) not in seen]
        same.sort(key=lambda r: r['qid'])
        for i, r in enumerate(same[:4]):
            conditional.append(dict(review_id=f'S{order}-{i+1}', edit_order=order, candidate=r, scope_label='UNKNOWN', data_role='SHARED_ANCHOR_REVIEW_ONLY', admitted_CAL=False, source_independent=False, reviewer=None, evidence=None))
        per_edit.append(dict(pilot_order=order, dataset=anchor['dataset'], attribute=anchor['attribute'], source_QA_evaluation=anchor['source_QA_evaluation'], triple_grounded=grounded(anchor['triple']), bucket_totals={k: len(v) for k, v in groups.items()}, selected=sum(min(len(v), cfg['maximum_per_bucket']) for v in groups.values()), unused_same_anchor_QA=len(same), conditional_selected=min(len(same), 4), verified_scope_positive=0, verified_scope_negative=0, admission='NOT_READY'))
    assert all(x['scope_label'] == 'UNKNOWN' and not x['admitted_CAL'] for x in packet + conditional)
    write('private/FACT_CARDS.json', cards)
    write('private/BLINDED_REVIEW_PACKET.json', packet)
    write('private/CONDITIONAL_SAME_IMAGE_REVIEW.json', conditional)
    write('private/SOURCE_QA_ROLE_AUDIT.json', source_rows)
    with (ROOT / 'private/BLINDED_REVIEW.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['review_id', 'edit_order', 'image_path', 'question', 'source_answer', 'source_annotation', 'retrieval_bucket', 'scope_label', 'reviewer_1', 'reviewer_2', 'evidence', 'adjudication'])
        writer.writeheader()
        for x in packet:
            r = x['candidate']
            writer.writerow(dict(review_id=x['review_id'], edit_order=x['edit_order'], image_path=r['image_path'], question=r['question'], source_answer=r['answer'], source_annotation=r['source_QA_evaluation'], retrieval_bucket=x['retrieval_bucket'], scope_label='UNKNOWN'))
    public = dict(status='COMPLETE_NOT_GPU_ADMITTED', pilot_count=8, source_QA_rows=len(source_rows), source_QA_by_dataset=dict(collections.Counter(r['dataset'] for r in source_rows)), exclusion_manifest_files=len(files), selected_metadata_files=len(metadata), excluded_source_image_ids=len(excluded), source_QA_with_excluded_images=sum(bool(r['historical_reasons']) for r in source_rows), remaining_source_QA_without_known_image_overlap=len(eligible), remaining_by_dataset=dict(collections.Counter(r['dataset'] for r in eligible)), missing_images=sum(not r['image_exists'] for r in source_rows), distinct_remaining_images=len({(r['dataset'], r['image_id']) for r in eligible}), private_review_records=len(packet), conditional_same_image_records=len(conditional), per_edit=per_edit, prior_triple_counts_corrected_to_template_only=True, all_scope_labels_UNKNOWN=True, new_verified_CAL=0, independent_CONFIRM=False, patient_independence='UNKNOWN', all_host_exposure_coverage='NOT_CERTIFIED', GPU_hours=0, new_Judge_attempts=0, training_steps=0, new_generation=0)
    write('public/CANDIDATE_AGGREGATES.json', public)
    check()
    seconds = time.time() - started
    write('RESOURCE_LEDGER.json', dict(CPU_seconds=seconds, failed_CPU_wall_upper_bound=failed_cpu, cumulative_CPU_wall_upper_bound=seconds+failed_cpu, input_bytes_read=used_bytes, GPU_hours=0, new_Judge_attempts=0, inherited_Judge_attempts=7472, permanent_missing=41, training_steps=0, new_generation=0))
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < cfg['storage_bytes_limit']
    write('public/FINAL_EXECUTION_AUDIT.json', dict(status='COMPLETE', source_binding='PASS', placeholder_rejection='PASS', exclusion_and_role_separation='PASS', CPU_seconds=seconds, failed_CPU_wall_upper_bound=failed_cpu, cumulative_CPU_wall_upper_bound=seconds+failed_cpu, input_bytes_read=used_bytes, new_verified_CAL=0, scientific_GPU_admission=False, public_delivery='PENDING_GITHUB'))
    write('RUN_STATUS.json', dict(status='COMPLETE', phase='CLOSED_WAITING_SCOPE_REVIEW', epoch=time.time()))
    print(json.dumps(public))


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        selfcheck()
        print('PASS: identity, placeholders, strict matching and irrelevant-candidate rejection')
    else:
        try:
            main()
        except Exception as error:
            write('FAILURE.json', dict(error=str(error), traceback=traceback.format_exc(), epoch=time.time()))
            write('RUN_STATUS.json', dict(status='FAILED_PRESERVED', epoch=time.time()))
            raise
