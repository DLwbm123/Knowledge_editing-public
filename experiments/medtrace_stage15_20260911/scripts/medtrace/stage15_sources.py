"""Release-bound external requests; no writer outputs participate in selection."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import time
import unicodedata

REV = 'd9f38639ec2285a0e9f541e22156ec14f87271d8'
BASELINE = '42ebd8a67fd54d23e724a37d85da5b4f264a0196'
SEED = 20260911
KAPPA = 0.7696741135364367
WRITERS = ('C_FACT', 'C_NO_H', 'BE', 'C_EXTRA_QA')
MODES = ('FORCED_ON', 'BE_ROUTE_R0', 'RC_FIXED_OLD16')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def identity(row):
    return tuple(row[k] for k in ('image', 'src', 'pred', 'alt'))


def canonical(row):
    return 'MedMKEB:'+REV+':'+digest(identity(row))


def paraphrases(question):
    # Derived from native text alone; official rephrase is inspected only by the collision check.
    return ['Please answer the following question: '+question,
            'Question: '+question, question+' Please provide an answer.',
            'Please respond to this question: '+question]


def select(rows, available, excluded):
    groups = defaultdict(list); seen = set(); counts = Counter()
    for r in rows:
        key = canonical(r)
        if key in seen: counts['duplicate_edit'] += 1; continue
        seen.add(key)
        if r['image'] not in available: counts['native_image_missing'] += 1; continue
        if available[r['image']]['sha256'] in excluded: counts['historical_identity_excluded'] += 1; continue
        groups[(r.get('clinical_VQA_task') or 'UNKNOWN', r.get('modality') or 'UNKNOWN')].append(r)
    queues = {k: deque(sorted(v, key=lambda r: hashlib.sha256(('20260911|'+canonical(r)).encode()).hexdigest())) for k, v in groups.items()}
    selected = []
    while any(queues.values()) and len(selected) < 200:
        for k in sorted(queues):
            if queues[k] and len(selected) < 200: selected.append(queues[k].popleft())
    counts.update(selected=len(selected), eligible=sum(map(len, groups.values())))
    return selected, dict(counts)


def assemble(run, storage):
    started = time.time(); run = Path(run); storage = Path(storage)
    if (run/'private/QUEUE.json').exists(): raise FileExistsError('Queue is already frozen')
    data = storage/'Knowledge_editing/datasets/MedMKEB'
    main, attack, train = [read(data/n) for n in ('eval_data_threehop_final.json', 'eval_data_attack.json', 'train_data.json')]
    assert (len(main), len(attack), len(train)) == (2497, 721, 4490)
    assets = {}; evidence = []; ambiguous = set()
    # Hash only newly used image identities: required cross-release exclusion and binding, not a transfer recheck.
    by_path = {}
    def bind(ref, path, source):
        path = Path(path)
        if not path.is_file(): return
        if path not in by_path: by_path[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        item = dict(path=str(path), sha256=by_path[path], provenance=source, original_ref=ref)
        if ref in assets and assets[ref]['sha256'] != item['sha256']: ambiguous.add(ref)
        else: assets.setdefault(ref, item)
    bundle = storage/'medmkeb_engram_projected_lora_bundle'
    for r in read(bundle/'records.json'):
        # Exact native tuple, not record-number or positional matching.
        matches = [x for x in main if identity(x) == (r['image_original_ref'], r['src'], r['pred'], r['alt'])]
        if len(matches) != 1: raise ValueError('Existing bundle does not bind uniquely to the pinned release')
        for field in ('image', 'image_rephrase', 'm_loc'):
            bind(matches[0][field], bundle/r[field], 'existing_threehop_bundle_exact_tuple')
    cross = storage/'Knowledge_editing/datasets/MedMKEB_cross_edit_v1'
    for r in read(cross/'source_pool.json')['records']:
        matches = [x for x in attack if identity(x) == (r['source_native_image'], r['src'], r['pred'], r['alt'])]
        if len(matches) != 1: raise ValueError('Existing attack pool does not bind uniquely to the pinned release')
        for field, ref in [('image', 'source_native_image'), ('image_rephrase', 'source_alternate_image'), ('m_loc', 'm_loc')]:
            bind(r[ref], cross/r[field], 'existing_attack_pool_exact_tuple')
    for r in read(data/'eval.json'):
        matches = [x for x in attack if all(x[k] == r[k] for k in ('id', 'src', 'pred', 'alt'))]
        if len(matches) != 1: raise ValueError('Smoke adapter identity ambiguity')
        x = matches[0]
        for field in ('image', 'image_rephrase', 'm_loc'):
            # The exact original tuple and original manifest support the adapter mapping.
            bind(x[field], data/r[field], 'existing_attack_adapter_exact_tuple_and_manifest')
    for key in ambiguous: assets.pop(key, None)
    outputs = storage/'Knowledge_editing/outputs'
    # Existing indexed image identities only; no retained-set answer file is opened.
    old = read(outputs/'medtrace_stage2_20260908_r02/private/NEW_EPISODE_MANIFEST_PRIVATE.json')
    ignored = {'historical_medtrace_all_roles', 'historical_medtrace_input_images', 'stage1_all_roles', 'existing_image_hash_identity_closure'}
    blocked_groups = {(r['dataset'], r['image_id']) for r in old['exposure_exclusion_ledger'] if set(r['reasons'])-ignored}
    def idname(value):
        p = Path(str(value)); return p.parent.name if p.parent.name.startswith('xmlab') else p.name.split('::')[0].split('_')[0] if p.name.startswith('xmlab') else p.name
    excluded = set()
    catalog = outputs/'m3bench_data_runtime_finalization_v3/20260904T014138Z/data_static/STATIC_QUERY_INVENTORY.jsonl'
    for line in catalog.open():
        r = json.loads(line)
        if (r['dataset'], idname(r['image_id'])) in blocked_groups and r.get('image_sha256'): excluded.add(r['image_sha256'])
    # Include realized later student inputs without treating unexecuted candidate pools as exposure.
    def hashes(obj):
        if isinstance(obj, dict):
            if obj.get('image_sha256'): excluded.add(obj['image_sha256'])
            for value in obj.values(): hashes(value)
        elif isinstance(obj, list):
            for value in obj: hashes(value)
    for p in outputs.glob('medtrace_stage*/private/edits/e*.json'): hashes(read(p))
    selected, counts = select(main, assets, excluded)
    eval_refs = {r[k] for r in main+attack for k in ('image', 'image_rephrase', 'm_loc')}
    train_available = [r for r in train if r['image'] in assets and r['image'] not in eval_refs]
    # There is no original GMAI QA/label binding in these evaluation-only image bundles.
    # `pred` is not promoted to a clinical source label. Missing H is a branch-level limitation.
    support_audit = dict(train_rows_scanned=len(train), train_native_images_resolved=sum(r['image'] in assets for r in train),
        train_native_rows_outside_all_eval_image_roles=len(train_available), verified_original_source_labels=0,
        status='H_SOURCE_LABEL_AND_SCOPE_UNSUPPORTED', no_pred_as_unverified_gold=True,
        upstream_access='GMAI gated; no existing authentication found; no terms accepted or access bypassed',
        missing_image_policy='Reuse bound existing assets; no full upstream image archive downloaded',
        full_train_scanned_once=True, previous_SLAKE_candidate_pool_rescanned=False)
    # Reuse only explicitly approved training support, not any previous evaluation or retained QA.
    old_tasks = read(outputs/'medtrace_stage13r_20260911_r01/private/paired/private/TASKS.json')
    u_pool = {}
    for t in old_tasks:
        for r in t['data']['rows']:
            if r['logical_id'] not in t['u_ids']: continue
            assert r['role'] == 'fit' and r['source_role'] == 'train' and Path(r['image_path']).is_file()
            assert not any(a['sha256'] == r['image_sha256'] for a in assets.values())
            u_pool.setdefault((r['dataset'], r['source_qid']), r)
    us = [dict(r, logical_id='U_fit-'+str(i)) for i, r in enumerate(u_pool.values())][:4]
    attacks = defaultdict(list)
    for a in attack: attacks[identity(a)].append(a)
    tasks = []
    for order, r in enumerate(selected, 1):
        fits = paraphrases(r['src'])
        official_questions = {norm(r['rephrase'])} | {norm(p['Q&A']['Question']) for p in r.get('port_new', [])}
        for a in attacks[identity(r)]: official_questions |= {norm(p['Q&A']['Question']) for p in a['port_new']}
        # No semantic information from official probes is fed back into the derivation.
        if len({norm(q) for q in fits}) != 4 or any(norm(q) in official_questions for q in fits):
            fit_status = 'UNSUPPORTED_FIT_OVERLAP'
        else: fit_status = 'SUPPORTED'
        probes = []
        def probe(kind, q, answer, ref, **metadata):
            a = assets.get(ref) if ref is not None else None
            probes.append(dict(probe_id=str(len(probes)), metric=kind, question=q, reference=answer,
                source_image=ref, image_path=a['path'] if a else None, image_sha256=a['sha256'] if a else None,
                status='AVAILABLE' if ref is None or a else 'UNSUPPORTED_IMAGE_MISSING', **metadata))
        probe('Reliability', r['src'], r['alt'], r['image'])
        probe('T_Generality', r['rephrase'], r['alt'], r['image'])
        probe('I_Generality', r['src'], r['alt'], r['image_rephrase'])
        probe('T_Locality', r['loc'], r['loc_ans'], None)
        probe('I_Locality', r['m_loc_q'], r['m_loc_a'], r['m_loc'])
        for p in r.get('port_new', []): probe('Portability', p['Q&A']['Question'], p['Q&A']['Answer'], r['image'], hop=p['port_type'])
        for a in attacks[identity(r)]:
            for p in a['port_new']: probe('Robustness', p['Q&A']['Question'], p['Q&A']['Answer'], r['image'], attack_type=p['attack_type'])
        key = canonical(r)
        tasks.append(dict(order=order, canonical_edit_id=key, source_id=r['id'], raw_record=r, probes=probes,
            image=assets[r['image']], fit_questions=fits, fit_status=fit_status, U=us, H=[], G=[],
            seed=int(hashlib.sha256((str(SEED)+'\0'+key).encode()).hexdigest()[:8], 16)&0x7fffffff,
            branches={'C_FACT':'UNSUPPORTED_NO_H', 'C_EXTRA_QA':'UNSUPPORTED_NO_H_G',
                'C_NO_H':'PENDING' if us and fit_status=='SUPPORTED' else 'UNSUPPORTED_NO_FIT_OR_U', 'BE':'PENDING'},
            stratum=[r.get('clinical_VQA_task') or 'UNKNOWN', r.get('modality') or 'UNKNOWN']))
    assert tasks and len(tasks) <= 200
    cfg = read(outputs/'medtrace_stage14_20260911_r01/private/CAMPAIGN_CONFIG.json')
    cfg.update(kind='MEDTRACE_STAGE15', allowed_physical_gpus=[0,1], worker_gpus=[0,1], judge_gpu=0,
        wall_hours=24, gpu_hours=48, train_seconds=21*3600, seed=SEED, public_baseline=BASELINE,
        fixed_kappa=KAPPA, campaign_epoch=None, medmkeb_revision=REV)
    cfg.pop('forbidden_physical_gpus', None)
    write(run/'private/CAMPAIGN_CONFIG.json', cfg)
    write(run/'private/QUEUE.json', tasks)
    write(run/'private/ASSET_BINDINGS.json', assets)
    write(run/'public/COVERAGE_AND_COST.json', dict(status='PREPARED', counts=counts, raw_rows=dict(train=len(train), main=len(main), attack=len(attack)),
        image_refs_reused=len(assets), images_downloaded=0, JSON_downloaded=3, ambiguous_image_refs=len(ambiguous),
        source_support=support_audit, U_QAs_reused=len(us), H_supported=0, C_EXTRA_QA_supported=0,
        preparation_seconds=time.time()-started, patients='UNKNOWN', pretraining_overlap='UNKNOWN',
        branches=dict(Counter(s for t in tasks for s in t['branches'].values())),
        actual_queue=[dict(order=t['order'], source_id=t['source_id'], stratum=t['stratum'], branches=t['branches']) for t in tasks],
        exclusions='Indexed actual student/development, retained/evaluation and quality identities; Base-only outputs not used for filtering'))
    print('FROZEN', len(tasks), 'H=0', 'U=', len(us), counts, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-root', type=Path, required=True); p.add_argument('--storage-root', type=Path, required=True)
    a = p.parse_args(); assemble(a.run_root, a.storage_root)
