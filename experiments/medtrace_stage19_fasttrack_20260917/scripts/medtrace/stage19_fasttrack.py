"""Two-arm, paired-prefix FASTTRACK worker; qualification is a separate frozen phase."""
import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage17_single import setup
from scripts.medtrace.stage19_fasttrack_budget import check, write


def query_id(row):
    return digest([row['image_sha256'], row['question']])


def load(cfg):
    setup(cfg)
    import importlib.metadata
    import torch
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    for name, version in cfg['runtime_lock']['actual_environment'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Environment changed: '+name)
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])))
    runtime.run_root = Path(cfg['run'])/'private/work'; runtime.model.eval()
    if runtime.generation_config != cfg['generation_lock'] or next(runtime.model.parameters()).dtype != torch.float16:
        raise ValueError('Generation/precision changed')
    return runtime


def record_for(task):
    from m3bench_repro.editors.llava_runtime import EditorRecord
    n = task['native']
    return EditorRecord(task['canonical_edit_id'], n['dataset'], n['question'], n['reference'],
        task['fit_questions'][0], Path(n['image_path']), n['image_path'], task['order'], 'AI_REVIEWED_DEV', 'NATIVE_ONLY_FIT')


def base_phase(cfg):
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import assert_base_off
    from m3bench_repro.editors.llava_runtime import EditorRecord
    root = Path(cfg['run']); source = read(root/'private/BASE_QUEUE.json')
    if digest(source['rows']) != cfg['base_queue_binding']:
        raise ValueError('Qualification queue changed')
    runtime = load(cfg); outputs = []
    for row in source['rows']:
        check(cfg); assert_base_off(runtime)
        record = EditorRecord(query_id(row), row['dataset'], row['question'], '', '', Path(row['image_path']), row['image_path'], len(outputs), 'BASE_ONLY', 'NONE')
        out, _, _, _ = stage15.base_output(runtime, root, row, record)
        outputs.append(dict(query_id=query_id(row), source=row, output=out))
        write(root/'public/PROGRESS.json', dict(phase='BASE', completed=len(outputs), total=len(source['rows'])))
        print('BASE', len(outputs), len(source['rows']), flush=True)
    write_new(root/'private/FRESH_BASE_OUTPUTS.json', dict(records=outputs, code=cfg['code_commit'], runtime=cfg['runtime_lock']))


def train_one(runtime, root, cfg, task, record, branch):
    from methods.medtrace.hsic import collect_features, select_layer, layer_path
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import teachers_for, train, state_hash, assert_base_off, artifact_binding
    from scripts.medtrace.stage18_support import validate_task
    validate_task(task, fasttrack_branch=branch); assert_base_off(runtime)
    directory = root/'private/edits'/f"e{task['order']:03d}"; directory.mkdir(parents=True, exist_ok=True)
    selected = directory/'SELECTION.json'
    binding = dict(task=digest(task), code=cfg['code_commit'], runtime=cfg['runtime_lock'], branch=branch)
    if selected.exists():
        selection = read(selected)
        artifact_binding(selection['binding'], binding, cfg)
    else:
        if branch == 'C_FACT':
            features, counts = collect_features(runtime, record, task['fit_questions'])
            selection = dict(select_layer(features), token_counts=counts)
            del features
        else: selection = dict(layer_id=21)
        selection['binding'] = binding; write_new(selected, selection)
    layer = selection['layer_id']
    adapted = dict(canonical_edit_id=task['canonical_edit_id'], order=task['order'], seed=task['seed'],
                   probes=[task['native']], U=task['U_fit'], fit_questions=task['fit_questions'])
    cp = stage15.initialize(runtime, root, cfg, adapted, record=record, seed_base=20260912, layer_path=layer_path(layer))
    expert = LowRankExpert(cp, task['seed'], rank=4).to(runtime.device)
    import torch
    x = torch.linspace(-1, 1, cp.d_in, device=runtime.device).reshape(1, -1)
    if not torch.allclose(cp.residual(x), expert.residual(x), rtol=2e-4, atol=2e-5):
        raise ValueError('CP transfer changed function')
    del cp, x
    w0 = state_hash(expert); teachers = teachers_for(runtime, root, cfg, task, record)
    train(runtime, root, cfg, task, expert, branch, record, teachers, w0, layer_id=layer)
    del teachers, expert
    assert_base_off(runtime)
    return dict(kind=branch, path=str(directory/branch/'latest.pt'), layer_id=layer, W0=w0,
                selection=digest(selection), task=digest(task), code=cfg['code_commit'])


def audited_resume(root, cfg, ids):
    """This recovery is restricted to the first fully validated paired insertion."""
    import torch
    audit=cfg.get('resume_audit')
    outputs=root/'private/OUTPUTS.jsonl'
    if not audit:
        if outputs.exists() or (root/'private/BANKS.pt').exists():
            raise ValueError('Output resume requires explicit evidence audit')
        return dict(banks={'A':{},'B':{}}, routes={}, inserted=[])
    rows=[json.loads(line) for line in outputs.read_text().splitlines()]
    bank=torch.load(root/'private/BANKS.pt',map_location='cpu',weights_only=False)
    if (audit['completed_pairs'] != 1 or bank['inserted'] != ids[:1]
            or bank['stream'] != cfg['stream_binding'] or digest(rows) != audit['outputs_binding']
            or len(rows) != 4 or set(bank['routes']) != set(ids[:1])
            or any(set(bank['banks'][arm]) != set(ids[:1]) for arm in ('A','B'))
            or read(root/'public/INTEGRATION_001.json')['status'] != 'PASS'):
        raise ValueError('Recovery evidence differs from audited completed pair')
    for arm in ('A','B'):
        pair=[r for r in rows if r['arm']==arm]
        if (len(pair)!=2 or [r['mode'] for r in pair]!=['insertion','integration_replay']
                or any(r['inserted']!=ids[:1] or r['code'] not in cfg['approved_predecessor_commits'] for r in pair)
                or pair[0]['output']['raw_token_ids']!=pair[1]['output']['raw_token_ids']
                or bank['banks'][arm][ids[0]]['code'] not in cfg['approved_predecessor_commits']):
            raise ValueError('Recovery paired provenance/replay mismatch')
    return bank


def campaign(cfg):
    import torch
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from methods.medtrace.hsic import bound_hook
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter, balanced_radius
    from scripts.medtrace import stage15
    from scripts.medtrace.stage18_cfact import assert_base_off, state_hash
    from scripts.medtrace.run_selective_write import save
    root = Path(cfg['run']); stream = read(root/'private/STREAM.json')
    if digest(stream) != cfg['stream_binding'] or cfg['mode'] != 'STAGE19_FASTTRACK_TWO_ARMS':
        raise ValueError('Stream/authorization changed')
    tasks = stream['tasks']; anchors = stream['anchors']; ids = [t['canonical_edit_id'] for t in tasks]
    if ids[:11] != anchors or len(set(ids)) != len(ids): raise ValueError('Anchor/order identity changed')
    resumed=audited_resume(root,cfg,ids)
    runtime = load(cfg); target = runtime.target_lock['balancedit']['targets'][0]
    frozen = [(p, p._version, p.data_ptr()) for p in runtime.model.parameters()]
    records = {t['canonical_edit_id']:record_for(t) for t in tasks}; routes = resumed['routes']; banks = resumed['banks']
    baseline = {r['query_id']:r for r in read(root/'private/FRESH_BASE_OUTPUTS.json')['records']}
    cache = {}; outputs = root/'private/OUTPUTS.jsonl'
    expert = LowRankExpert(AsymmetricCPExpert(14336, 4096, 4).to(runtime.device), 20260912, rank=4).to(runtime.device).requires_grad_(False)

    def guard():
        check(cfg); assert_base_off(runtime)
        if any(p._version != v or p.data_ptr() != ptr or p.requires_grad for p,v,ptr in frozen):
            raise ValueError('Frozen Base changed')
        if shutil.disk_usage(root).free < 8*1024**3:
            raise OSError('Eight GiB disk reserve reached')

    def key(record):
        assert_base_off(runtime)
        with torch.inference_mode():
            return runtime.extract_layer_input_key(runtime.build_question_batch(record), module_path=target, pooling='mean')

    def prepared(row):
        q = query_id(row)
        if q not in cache:
            guard(); raw, _, binding = stage15.prepared(runtime, row, next(iter(records.values())))
            base = baseline[q]['output']
            if binding != base['binding']: raise ValueError('Fresh Base prompt binding changed')
            r = replace(next(iter(records.values())), question=row['question'], target='', image_path=Path(row['image_path']))
            # Keep only the tiny key/Base on CPU; do not cache hundreds of GPU image embeddings.
            cache[q] = (base, key(r).cpu())
        base, k = cache[q]
        raw, _, binding = stage15.prepared(runtime, row, next(iter(records.values())))
        if binding != base['binding']: raise ValueError('Cached query changed')
        return base, raw, k.to(runtime.device)

    def generated(arm, inserted, row, mode):
        guard(); base, raw, k = prepared(row)
        router = MemoryRouter.from_state(dict(distance='euclidean', entries=[routes[e] for e in inserted]), device=runtime.device)
        decision = router.route(k); selected = decision.logical_edit_id
        out = base; weight = None
        if selected is not None:
            state = banks[arm][selected]; saved = torch.load(state['path'], map_location='cpu', weights_only=False)
            if state['code'] not in [cfg['code_commit']]+cfg.get('approved_predecessor_commits',[]): raise ValueError('Expert execution version changed')
            if state['kind'] == 'BE':
                if saved['code'] != state['code']: raise ValueError('BE producer binding changed')
                editor = BalanceEditPaperSpecEditor(runtime); base_layer = editor.wrapper.base
                try:
                    editor.wrapper.load_exported_state(saved['wrapper'])
                    with editor._activated(selected): out = stage15.generate(runtime, raw, base['binding'])
                finally:
                    editor.reset_editor_state(); runtime.replace_module(target, base_layer)
                weight = state['weight_binding']
            else:
                if saved['step'] != 320 or digest(saved['binding']['task']) != state['task'] or saved['binding']['code'] != state['code']:
                    raise ValueError('Incomplete/misbound expert')
                expert.load_state_dict(saved['expert']); weight = state_hash(expert)
                with bound_hook(runtime, expert, dict(layer_id=state['layer_id'], expert_id=selected, W0_id=state['W0'])) as hook:
                    out = stage15.generate(runtime, raw, base['binding'], hook)
            del saved
        guard()
        result = dict(arm=arm, mode=mode, prefix=len(inserted), inserted=list(inserted), query_id=query_id(row), source=row,
                      route=asdict(decision), selected_kind=banks[arm][selected]['kind'] if selected else 'BASE',
                      weight_binding=weight, Base=base, output=out, code=cfg['code_commit'],
                      expert_producer_code=banks[arm][selected]['code'] if selected else None, semantic_status='UNJUDGED')
        with outputs.open('a') as f: f.write(json.dumps(result)+'\n')
        return result

    def endpoint(inserted):
        rows = {query_id(t['native']):t['native'] for t in tasks[:len(inserted)]}
        rows.update({query_id(r):r for r in stream['core_rows']})
        for arm in ('A','B'):
            for row in rows.values(): generated(arm, inserted, row, 'endpoint')
        write(root/'public'/f'PREFIX_{len(inserted):03d}.json', dict(status='GENERATED_NOT_SCORED', N=len(inserted), K=min(11,len(inserted)), track=stream['track'], outputs_per_arm=len(rows)))

    inserted = list(resumed['inserted']); costs = []; completed_prefixes = []
    for task in tasks[len(inserted):]:
        guard(); index = task['order']; began = time.time()
        # Reserve endpoint work before starting an indivisible paired insertion.
        estimate = max(costs, default=600)*1.25
        if inserted and cfg['gpu_deadline_epoch']-time.time() < 7200+estimate:
            print('TRAINING_RESERVE_BOUNDARY', len(inserted), flush=True); break
        if shutil.disk_usage(root).free < 8*1024**3 + 650*1024**2:
            print('STORAGE_BOUNDARY', len(inserted), flush=True); break
        edit = task['canonical_edit_id']; record = records[edit]
        k = key(record); pos = key(replace(record, question=task['fit_questions'][0]))
        black = runtime.make_black_image(record, root/'private/black'); neg = key(replace(record, image_path=black))
        router = MemoryRouter('euclidean'); router.add(edit, k, balanced_radius(k,pos,neg,alpha=.2,distance='euclidean'))
        entry = router.export_state()['entries'][0]; entry['label']=[]; routes[edit]=entry
        shared = stream['track']=='S' and edit not in anchors
        branch = 'C_NO_H' if shared else 'C_FACT'
        banks['A'][edit] = train_one(runtime, root, cfg, task, record, branch)
        if shared:
            banks['B'][edit] = dict(banks['A'][edit])
        else:
            guard(); editor = BalanceEditPaperSpecEditor(runtime); base_layer = editor.wrapper.base
            try:
                lock = editor.config_lock()
                if lock['steps_per_edit'] != 50 or lock['learning_rate'] != .01 or lock['alpha'] != .2: raise ValueError('BE recipe changed')
                receipt = editor.apply_edit(record)
                if not receipt['finite_losses'] or not receipt['finite_gradients']: raise ValueError('BE failed')
                actual = editor.router.export_state()['entries'][0]
                if not torch.equal(actual['key'],entry['key']) or actual['radius'] != entry['radius']: raise ValueError('Paired route anchors differ')
                state = dict(wrapper=editor.wrapper.export_state(), task=task, training=receipt, code=cfg['code_commit'])
                path = root/'private/BE'/f'e{index:03d}.pt'; save(path,state)
                slot = editor.wrapper.logical_to_slot[edit]
                original_weight_hash = state_hash(editor.wrapper.edited[slot])
                restored = torch.load(path, map_location='cpu', weights_only=False)
                editor.reset_editor_state(); editor.wrapper.load_exported_state(restored['wrapper'])
                weight_binding = state_hash(editor.wrapper.edited[editor.wrapper.logical_to_slot[edit]])
                if weight_binding != original_weight_hash: raise ValueError('BE saved weights changed')
                banks['B'][edit] = dict(kind='BE',path=str(path),code=cfg['code_commit'],weight_binding=weight_binding,bytes=path.stat().st_size)
            finally:
                editor.reset_editor_state(); runtime.replace_module(target,base_layer)
            del editor, state, restored
        inserted.append(edit); save(root/'private/BANKS.pt', dict(banks=banks, routes=routes, inserted=inserted, stream=cfg['stream_binding']))
        insertion = [generated(arm,inserted,task['native'],'insertion') for arm in ('A','B')]
        if index in (1,12):
            # Actual saved/reloaded active outputs and Base OFF parity for the first pair and mixed bank.
            for arm, first in zip(('A','B'), insertion):
                again = generated(arm,inserted,task['native'],'integration_replay')
                if again['output']['raw_token_ids'] != first['output']['raw_token_ids']: raise ValueError('Restored output changed')
            base,raw,_ = prepared(task['native']); off = stage15.generate(runtime,raw,base['binding'])
            if off['raw_token_ids'] != base['raw_token_ids']: raise ValueError('Base OFF restore failed')
            visible=[dict(opaque_query_id=digest(r),question=r['source']['question'],gold_answer=r['source']['reference'],raw_base_answer=r['output']['raw_answer']) for r in insertion]
            write_new(root/'private'/f'INTEGRATION_JUDGE_PACKET_{index:03d}.json',dict(records=visible,status='PACKAGED_NOT_JUDGED',method_labels_visible=False))
            write(root/'public'/f'INTEGRATION_{index:03d}.json',dict(status='PASS',paired_save_load=True,Base_OFF=True,source_only_routes=True,semantic_accuracy_gate=False,mixed_bank=shared))
        if index in (11,50,100): endpoint(inserted); completed_prefixes.append(index)
        costs.append(time.time()-began); gc.collect(); torch.cuda.empty_cache()
        write(root/'public/PROGRESS.json', dict(phase='PAIRED_TRAINING', completed=len(inserted), planned=len(tasks), track=stream['track'], last_pair_seconds=costs[-1]))
    if not inserted: raise RuntimeError('No common completed endpoint')
    if len(inserted) not in completed_prefixes: endpoint(inserted)
    guard()
    from collections import Counter
    training=[]
    for t in tasks[:len(inserted)]:
        edit=t['canonical_edit_id'];state=banks['A'][edit]
        receipt=read(Path(state['path']).parent/'TRAINING.json')
        training.append(dict(position=t['order'],kind=state['kind'],layer=state['layer_id'],steps=receipt['steps'],H_nonzero_steps=receipt['extra_gradient_nonzero_steps'],weight_bytes=Path(state['path']).stat().st_size))
    write(root/'public/TRAINING_ACCEPTANCE.json',dict(rows=training,layer_counts=dict(Counter(r['layer'] for r in training)),
        FACT_steps=sum(r['steps'] for r in training if r['kind']=='C_FACT'),H_nonzero_steps=sum(r['H_nonzero_steps'] for r in training),
        BE_bytes=sum(s.get('bytes',0) for s in banks['B'].values()),background_weights_shared=stream['track']=='S',remaining_disk_bytes=shutil.disk_usage(root).free))
    write_new(root/'public/GENERATED.json',dict(status='PAIRED_ENDPOINTS_GENERATED_NOT_SCORED',N=len(inserted),K=min(11,len(inserted)),track=stream['track'],prefixes=sorted(set(completed_prefixes+[len(inserted)])),planned_N=len(tasks),cleanup_permitted=False))


def cleanup(cfg):
    root=Path(cfg['run']).resolve()
    if read(root/'public/EXIT_train.json')['exit_code']!=0 or read(root/'public/GENERATED.json')['status']!='PAIRED_ENDPOINTS_GENERATED_NOT_SCORED':
        raise ValueError('Weight consumers have not finished')
    ledger=read(root/'public/GPU_BUDGET_LEDGER.json')
    if any('seconds' not in s for s in ledger['sessions']):raise ValueError('A GPU worker is still open')
    for s in ledger['sessions']:
        if s.get('pid') and Path('/proc',str(s['pid'])).exists():raise ValueError('Recorded worker PID still exists; preserve weights')
    files=[p for p in (root/'private').rglob('*.pt') if p.name!='BANKS.pt']
    if any(not p.resolve().is_relative_to(root/'private') or p.is_symlink() for p in files):raise ValueError('Cleanup path escapes run')
    manifest=[dict(path=str(p),bytes=p.stat().st_size) for p in files]
    write_new(root/'private/CHECKPOINT_CLEANUP_PLAN.json',dict(files=manifest,reason='All registered weight consumers completed; small repro artifacts copied privately by controller'))
    for p in files:p.unlink()
    receipt=dict(files=len(files),bytes=sum(r['bytes'] for r in manifest),retained_router_bank=True,pretrained_models_untouched=True,requires_retraining_to_recreate=True,reason='Insertion/prefix/save-load consumers complete; original outputs and bindings preserved')
    write_new(root/'private/CHECKPOINT_CLEANUP.json',dict(receipt,deleted=manifest));write_new(root/'public/CHECKPOINT_CLEANUP.json',receipt)


if __name__ == '__main__':
    config = read(os.environ['JOB_CONFIG'])
    # Runtime modules read model/source environment variables at import time.
    setup(config)
    if config['phase'] == 'base': base_phase(config)
    elif config['phase'] == 'train': campaign(config)
    elif config['phase'] == 'cleanup': cleanup(config)
    else: raise ValueError('Unknown FASTTRACK phase')
