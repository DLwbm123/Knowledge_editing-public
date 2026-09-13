"""First Stage17 dispatch: frozen BE single T0 queue; no automatic extra branches."""
import argparse
from dataclasses import replace
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest


def setup(cfg):
    """Do this before importing the runtime (its source/model paths are constants)."""
    project = Path(cfg['project'])
    for key, value in dict(CUDA_VISIBLE_DEVICES=str(cfg['gpu']),
            M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(cfg['gpu']),
            M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(cfg['gpu']),
            M3BENCH_FORMAL_EXPECTED_GPU_UUID=cfg['gpu_uuid'],
            M3BENCH_LLAVA_SOURCE=str(project/'worktrees/source'),
            M3BENCH_EXPECTED_LLAVA_SOURCE=str(project/'worktrees/source'),
            M3BENCH_MODEL_PATH=str(project/'models/medical_vlms/llava_med_v1_5_mistral_7b'),
            M3BENCH_VISION_PATH=str(project/'models/openai/clip-vit-large-patch14-336'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4', TMPDIR=str(project/'tmp')).items():
        os.environ[key] = value
    if subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip() != cfg['code_commit']:
        raise ValueError('Runtime source commit differs from dispatch lock')
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Runtime checkout must be clean')


def record_for(t):
    from m3bench_repro.editors.llava_runtime import EditorRecord
    n = t['native']
    if t['fit_status'] != 'SUPPORTED':
        raise ValueError('No legal native-only route anchor')
    return EditorRecord(t['edit_id'], n['dataset'], n['question'], n['reference'],
        t['fit_questions'][0], Path(n['image_path']), n['original_image_path'], t['order'],
        'VERIFIED_SOURCE_ANSWER', 'NATIVE_ONLY_CONSERVATIVE_FIT_NOT_OFFICIAL_EVALUATION_REPHRASE')


def smoke(cfg):
    from scripts.medtrace.stage17_smoke import run
    directory = Path(cfg['run'])
    setup(cfg)
    value = dict(scope='DEV_NATIVE_ONLY_MECHANICAL', gpu=str(cfg['gpu']),
        cpu_gate=cfg['cpu_gate'], runtime_lock=cfg['runtime_lock'],
        dev_manifest=str(directory/'private/LORA_DEV16_V2_MANIFEST.json'),
        dev_inputs=str(directory/'private/LORA_DEV16_V2_INPUTS.jsonl'),
        relocated_project=cfg['project'], previous_smoke=str(directory/'private/SMOKE_PRIVATE.json'))
    run(value, directory/'private/migration/GPU_SMOKE.json')


def routes(distance, radius):
    return dict(R0=distance <= radius, RC=distance <= radius * 0.7696741135364367)


def worker(cfg):
    setup(cfg)
    import torch
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor, record_seed
    from m3bench_repro.editors.routing import MemoryRouter
    from m3bench_repro.editors.llava_runtime import write_json_atomic
    from scripts.medtrace.run_selective_write import save
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    run = Path(cfg['run'])
    if read(run/'private/migration/GPU_SMOKE.json')['status'] != 'PASS':
        raise ValueError('Migration mechanical check not passed')
    ledger = read(run/'private/COHORT_AND_SUPPORT_LEDGER.json')
    if ledger['freeze_id'] != cfg['freeze_id']:
        raise ValueError('Dispatch queue changed')
    if digest({k:v for k,v in ledger.items() if k != 'freeze_id'}) != ledger['freeze_id']:
        raise ValueError('Frozen queue content changed')
    all_tasks = {t['edit_id']:t for t in ledger['tasks']}
    tasks = [all_tasks[q] for q in ledger['main_T0']]
    if len(tasks) != cfg['N'] or cfg['method'] != 'balancedit' or cfg['mode'] != 'single_main_T0':
        raise ValueError('Unsupported or changed dispatch')
    if any(len(set(group) & set(ledger['main_T0'])) > 1 for group in ledger['duplicates']):
        raise ValueError('Resolve duplicate main-stream inputs before execution')
    # The first dispatch has a bounded disk budget; other tasks are NOT silently excluded.
    required = cfg['checkpoint_bytes_per_edit'] * len(tasks) + 8 * 1024**3
    if shutil.disk_usage(run).free < required:
        raise OSError('Insufficient data-disk capacity for this frozen dispatch')
    print('LOADING_RUNTIME', flush=True)
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])))
    runtime.run_root = run/'private/work'
    frozen = [(p, p._version, p.data_ptr()) for p in runtime.model.parameters()]
    editor = BalanceEditPaperSpecEditor(runtime)
    if editor.target != cfg['method_lock']['target']:
        raise ValueError('Unexpected BE target')
    actual = editor.config_lock()
    for k in ('alpha','steps_per_edit','learning_rate','target','gradient_clip','edited_state_precision'):
        if actual[k] != cfg['method_lock'][k]:
            raise ValueError('Frozen BE recipe changed: '+k)
    bindings = read(run/'private/BINDINGS.json')
    started = time.time()
    status = run/'public/SINGLE_PROGRESS.json'
    completed = 0
    for t in tasks:
        directory = run/'private/single_BE'/f"e{t['order']:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        binding = dict(freeze_id=ledger['freeze_id'], input=t['native'],
            runtime=cfg['runtime_lock'], generation=runtime.generation_config,
            writer='BalancEdit_adaptation', prefix=1,
            training=dict(method=cfg['method_lock'], record_id=t['edit_id'],
                          seed=record_seed(t['edit_id'], 'balancedit'),
                          native_fit=t['fit_questions'], independent_frozen_Base=True),
            execution=dict(commit=cfg['code_commit'], gpu_uuid=cfg['gpu_uuid']))
        if (directory/'COMPLETE.json').exists():
            if read(directory/'COMPLETE.json')['binding'] != binding:
                raise ValueError('Completed state binding mismatch')
            completed += 1
            continue
        if (directory/'FAILURE.json').exists():
            raise RuntimeError('Prior failure requires explicit same-task recovery; do not silently retrain')
        if (run/'STOP').exists():
            raise RuntimeError('Explicit stop requested; preserving all completed states')
        if shutil.disk_usage(run).free < 5 * 1024**3:
            raise OSError('Data disk reserve reached')
        editor.reset_editor_state()
        assert not editor.edit_history and len(editor.router) == 0 and not editor.wrapper.logical_to_slot
        record = record_for(t)
        checkpoint = directory/'state.pt'
        began = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            write_json_atomic(status, dict(status='RUNNING', phase='TRAINING', completed=completed,
                N=len(tasks), current_order=t['order'], method='BalancEdit', scope='single_main_T0'))
            print('TRAIN_START', t['order'], flush=True)
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if state['binding'] != binding:
                    raise ValueError('Checkpoint binding mismatch')
                editor.router = MemoryRouter.from_state(state['router'], device=runtime.device)
                editor.wrapper.load_exported_state(state['wrapper'])
                editor.edit_history = state['edit_history']
                training = state['training']
                del state
            else:
                training = editor.apply_edit(record)
                if not training['finite_losses'] or not training['finite_gradients']:
                    raise FloatingPointError('Nonfinite BE training; keep failure')
                # BE distance routing never needs the teacher labels; keep them
                # out of the saved deployment bank as well as out of query inputs.
                editor.router.labels = [()] * len(editor.router)
                state = dict(binding=binding, training=training, method=editor.method, target=editor.target,
                    router=editor.router.export_state(), wrapper=editor.wrapper.export_state(),
                    edit_history=list(editor.edit_history))
                save(checkpoint, state)
                # One first-edit save/load equality check, not a repeated full checkpoint hash.
                if t['order'] == 1:
                    restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    assert restored['binding'] == binding
                    for edit, weights in state['wrapper']['edits'].items():
                        assert all(torch.equal(w, restored['wrapper']['edits'][edit][k]) for k,w in weights.items())
                    editor.reset_editor_state()
                    editor.router = MemoryRouter.from_state(restored['router'], device=runtime.device)
                    editor.wrapper.load_exported_state(restored['wrapper'])
                    editor.edit_history = restored['edit_history']
                    del restored
                del state
            print('TRAIN_DONE', t['order'], training['steps'], flush=True)
            assert all(p._version == version and p.data_ptr() == pointer and not p.requires_grad
                       for p, version, pointer in frozen)
            queries = list(dict.fromkeys([t['edit_id']] + [q for e in t['events'] for q in e['all_probe_query_ids']]))
            write_json_atomic(status, dict(status='RUNNING', phase='GENERATING', completed=completed,
                N=len(tasks), current_order=t['order'], method='BalancEdit', scope='single_main_T0'))
            for i, qid in enumerate(queries):
                q = ledger['queries'][qid]
                b = bindings[q['opaque_Base_id']]
                expected = dict(input=q, runtime=cfg['runtime_lock'], generation=runtime.generation_config,
                    writer=binding['writer'], prefix=1, training_binding=digest(binding))
                output = directory/f'query_{i:03d}.json'
                if output.exists():
                    if read(output)['binding'] != expected:
                        raise ValueError('Output cache binding mismatch')
                    continue
                query = replace(record, record_id='query', question=q['question'], target='',
                    official_rephrase='', image_path=Path(q['image_path']))
                raw = runtime.adapter.prepare_inputs(query.image_path, query.question, None)
                if (raw['input_ids'].tolist() != [b['prompt_ids']]
                        or raw['attention_mask'].tolist() != [b['attention_mask']]
                        or raw['image_sha256'] != b['image_sha256']
                        or runtime.generation_config != b['generation']):
                    raise ValueError('Realized input does not match the accepted Base binding')
                with torch.inference_mode():
                    decision = editor._route(query)  # Base-only features; query has no gold or edit ID.
                    distance = float(decision.nearest_distance)
                    radius = float(editor.router.radii[0])
                    mode_on = routes(distance, radius)
                    with editor._activated(t['edit_id']):
                        result = runtime.adapter.generate_prepared_with_result(raw, runtime.generation_config)
                    forced = dict(raw_answer=result.decoded_text, raw_token_ids=list(result.raw_token_ids))
                    base = dict(raw_answer=b['output']['model_answer_raw'],
                                raw_token_ids=b['output']['raw_generated_token_ids'])
                    # Verify real routed deployment once per edit before deriving duplicate modes.
                    if i == 0:
                        with editor._activated(t['edit_id'] if mode_on['R0'] else None):
                            routed = runtime.adapter.generate_prepared_with_result(raw, runtime.generation_config)
                        assert list(routed.raw_token_ids) == (forced if mode_on['R0'] else base)['raw_token_ids']
                assert editor.wrapper.active_logical_id is None
                write_new(output, dict(binding=expected, Base_cache_id=q['opaque_Base_id'],
                    distance=distance, radius_R0=radius, route_on=mode_on,
                    FORCED_ON=forced, R0=forced if mode_on['R0'] else base,
                    RC=forced if mode_on['RC'] else base,
                    routing_derivation='single immutable expert; actual native R0 parity checked each edit',
                    canonical_cap=1024, student_Judge_status='PENDING'))
            receipt = dict(status='GENERATED_NOT_SCORED', binding=binding, training=training,
                queries=len(queries), seconds=time.time()-began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), checkpoint_bytes=checkpoint.stat().st_size)
            write_new(directory/'COMPLETE.json', receipt)
            completed += 1
            print('EDIT_GENERATED', t['order'], 'of', len(tasks), flush=True)
        except Exception as exc:
            write_new(directory/'FAILURE.json', dict(binding=binding, error=repr(exc),
                traceback=traceback.format_exc(), completed=completed, N=len(tasks)))
            write_json_atomic(status, dict(status='EXECUTION_FAILURE', current_order=t['order'],
                completed=completed, N=len(tasks), error_type=type(exc).__name__))
            raise
        finally:
            editor.reset_editor_state()
            gc.collect()
    write_json_atomic(status, dict(status='GENERATED_NOT_SCORED', completed=completed, N=len(tasks),
        seconds=time.time()-started, scope='single_main_T0', method='BalancEdit',
        remaining='other branches, task-specific queues, sequential, student Judge, report and public closeout'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['smoke','worker'])
    p.add_argument('config', type=Path)
    a = p.parse_args()
    cfg = read(a.config)
    if a.action == 'smoke':
        smoke(cfg)
    else:
        with (Path(cfg['run'])/'private/worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(cfg)
