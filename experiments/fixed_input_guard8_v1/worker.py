"""Production current-source worker. Real-model imports occur only after authorization."""
import json
import os
import sys
from pathlib import Path
import torch
from .configuration import require_execution, LAYER, GPU_SECONDS, SLOTS, SEEDS
from .executor import Editor, Budget, atomic_json
from .configuration import VERSION


class CooperativeStop(RuntimeError):
    pass


def run_current(runtime, arm, calibration, record, fits, owner, folder, seed=None, resume=False, stop_requested=None, diagnostic=False):
    folder = Path(folder)
    budget = Budget(folder/'budget.json')
    editor = Editor(runtime, arm, calibration, budget)
    try:
        checkpoint = folder/'active.pt'
        if resume:
            editor.load(checkpoint)
            if editor.owner != owner or editor.boundary:
                raise ValueError('Wrong resume boundary')
            editor.attach_source(record, fits)
        elif owner > 1:
            consumed = json.loads((folder/'CONSUMED.json').read_text())
            if consumed.get('owner') != owner-1 or consumed.get('insertion_complete') is not True or (owner-1 in (4, 8) and consumed.get('prefix_complete') is not True):
                raise ValueError('Previous native consumers are not complete')
            editor.load(checkpoint)
            if editor.owner != owner-1 or not editor.boundary:
                raise ValueError('Missing previous edit boundary')
        elif checkpoint.exists() or (folder/'native.pt').exists():
            raise FileExistsError('New run would overwrite active state')
        if seed is not None and not resume:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_initialized():
                torch.cuda.manual_seed_all(seed)
        if not resume:
            editor.prepare_source(record, fits, owner)
        events = []
        for slot in range(editor.slot+1, SLOTS+1):
            event = editor.attempt(slot)
            atomic_json(folder/'steps'/str(owner)/(str(slot)+'.json'), event)
            events.append(event)
            if diagnostic and owner in (7, 8) and event['accepted']:
                from .diagnostic import save_snapshot
                save_snapshot(editor.weight, folder, owner, slot)
            if stop_requested is not None and stop_requested():
                editor.save(checkpoint)
                raise CooperativeStop('Exact slot checkpoint saved; explicit resume required')
        if diagnostic and owner == 6:
            from .diagnostic import save_snapshot
            save_snapshot(editor.weight, folder, owner, SLOTS)
        summary = editor.finish_source()
        editor.save(checkpoint)
        atomic_json(folder/'sources'/(str(owner)+'.json'), summary)
        # Only one current native export; prefix consumers must finish before replacement.
        destination = folder/'native.pt'
        if destination.is_symlink() or destination.with_suffix('.tmp').exists():
            raise ValueError('Unsafe native export path')
        torch.save(dict(version=VERSION, native=editor.weight.detach().cpu(), owner=owner, arm=arm), destination.with_suffix('.tmp'))
        destination.with_suffix('.tmp').replace(destination)
        return dict(summary=summary, accepted=sum(e['accepted'] for e in events), slots=len(events),
                    counters=budget.data['counts'], native_dtype=str(editor.weight.dtype))
    finally:
        editor.adapter.close()


def load_native(runtime, path, arm, owner):
    state = torch.load(path, map_location='cpu', weights_only=True)
    weight = runtime.get_module(LAYER).weight
    if state['version'] != VERSION or state['arm'] != arm or state['owner'] != owner:
        raise ValueError('Native export identity mismatch')
    native = state['native']
    if native.shape != weight.shape or native.dtype != weight.dtype or not torch.isfinite(native).all():
        raise ValueError('Native export shape/dtype mismatch')
    with torch.no_grad():
        weight.copy_(native.to(weight.device))


def admit_current(runtime, arm, calibration, record, fits, folder):
    editor = Editor(runtime, arm, calibration, Budget(Path(folder)/'budget.json'))
    original = editor.weight.detach().cpu().clone()
    try:
        editor.prepare_source(record, fits, 1)
        event = editor.attempt(1, admission=True)
        atomic_json(Path(folder)/'ADMISSION_ATTEMPT.json', event)
        if not event['accepted'] or event.get('native_roundtrip') != 'PASS':
            raise ValueError('Real native admission failed; no formal slots authorized')
        if not torch.equal(editor.weight.detach().cpu(), original) or editor.optimizer.state:
            raise ValueError('Admission did not restore original state')
        return dict(status='PASS', native_dtype=str(editor.weight.dtype), event=event)
    finally:
        editor.adapter.close()


def validate_calibration(manifest):
    if manifest.get('role') != 'CALIBRATION_TRAIN' or not manifest.get('calibration_ids'):
        raise ValueError('Missing training-only calibration provenance')
    if any(not manifest.get(k) for k in ('calibration_source_groups', 'edit_and_eval_ids', 'edit_and_eval_source_groups')):
        raise ValueError('Incomplete calibration separation audit')
    if len(manifest['calibration_ids']) != 32 or len(set(manifest['calibration_ids'])) != 32:
        raise ValueError('Exactly 32 distinct training calibration records required')
    if set(manifest['calibration_ids']) & set(manifest['edit_and_eval_ids']):
        raise ValueError('Calibration query leakage')
    if set(manifest['calibration_source_groups']) & set(manifest['edit_and_eval_source_groups']):
        raise ValueError('Calibration source-group leakage')


def calibration_inputs(runtime, records, manifest, budget):
    from .adapter import RuntimeAdapter
    validate_calibration(manifest)
    if [r.record_id for r in records] != manifest['calibration_ids']:
        raise ValueError('Calibration order/binding mismatch')
    adapter = RuntimeAdapter(runtime, budget)
    try:
        rows = []
        for record in records:
            prepared = runtime.build_edit_batch(record)
            rows.append(adapter.inputs([adapter.pack(prepared)]))
        from experiments.fixed_input_guard_v1.guard import right_basis, normalized_rows
        basis = right_basis(normalized_rows(torch.cat(rows)), 32).T
        result = torch.zeros((32, basis.shape[1]), dtype=torch.float32)
        result[:len(basis)] = basis
        return result
    finally:
        adapter.close()


def main(config):
    require_execution(config)
    if config.get('resume') and config['authorization'].get('resume') is not True:
        raise PermissionError('Explicit resume authorization required')
    validate_calibration(config['calibration_manifest'])
    sys.path.insert(0, config['source_root'])
    from experiments.purew_original146_v1 import common as c
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from m3bench_repro.editors.llava_runtime import EditorRecord
    from types import SimpleNamespace
    gpu = config['physical_GPU']
    if Path(config['arm_folder']).resolve() != (c.RUN/'private/fixed8'/config['arm']).resolve():
        raise ValueError('Arm output path must stay inside this run')
    if Path(config['source_root']).resolve() != (c.RUN/'private/source').resolve():
        raise ValueError('Runtime source must be the bound run checkout')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Physical GPU binding mismatch')
    uuid = c.read(c.RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    if uuid != config['GPU_UUID']:
        raise ValueError('GPU UUID mismatch')
    os.environ.update(M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),
        M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu), M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid,
        M3BENCH_LLAVA_SOURCE=str(c.RUN/'private/official_llava'),
        M3BENCH_EXPECTED_LLAVA_SOURCE=str(c.RUN/'private/official_llava'))
    import subprocess
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=config['source_root'], text=True).strip() != config['source_commit']:
        raise ValueError('Runtime source binding mismatch')
    with c.lease(gpu):
        with c.resources() as resources:
            current = next(x for x in resources['gpu_sessions'] if x['pid'] == os.getpid() and not x.get('ended_epoch'))
            current['arm'] = config['arm']
        original_check = Budget.check
        def real_budget(self):
            original_check(self)
            c.budget()
            rows = c.read(c.RUN/'RESOURCE_LEDGER.json')['gpu_sessions']
            import time
            used = sum(x.get('resident_seconds', time.time()-x['started_epoch']) for x in rows if x.get('arm') == config['arm'])
            if used >= GPU_SECONDS:
                raise TimeoutError('Per-arm GPU budget exhausted')
        Budget.check = real_budget
        try:
            runtime = load_real_runtime(SimpleNamespace(cpu_gate=Path(config['cpu_gate'])))
            c.process_audit(gpu)
            if str(runtime.get_module(LAYER).weight.dtype) != 'torch.'+config['native_dtype']:
                raise ValueError('Real runtime changed frozen native dtype')
            from experiments.purew_medical_evidence100_v1.access import install
            window = c.RUN/'private/window'/config['arm']
            if Path(config['current_source']).resolve() != (window/'current.json').resolve():
                raise ValueError('Current-source path mismatch')
            install(c.RUN, window, config['source_image_root'])
            packet = json.loads(Path(config['current_source']).read_text())
            if packet.get('role') != 'SOURCE_EDIT' or packet['order'] not in range(1, 9):
                raise ValueError('Wrong current-source window')
            if packet['seed'] != SEEDS[packet['order']-1]:
                raise ValueError('Source seed changed')
            row = packet['record']
            row['image_path'] = Path(row['image_path'])
            if window.resolve() not in row['image_path'].resolve().parents:
                raise ValueError('Source image must be in the isolated current window')
            record = EditorRecord(**row)
            calibration = torch.load(config['calibration_inputs'], map_location='cpu', weights_only=True)
            if (calibration['manifest'] != config['calibration_manifest'] or calibration['inputs'].shape != (32, 14336)
                    or calibration['native_dtype'] != str(runtime.get_module(LAYER).weight.dtype) or calibration['layer'] != LAYER):
                raise ValueError('Calibration artifact mismatch')
            if config.get('action', 'learn') == 'admit':
                if packet['order'] != 1:
                    raise ValueError('Admission only uses source1')
                result = admit_current(runtime, config['arm'], calibration['inputs'], record, packet['fit_questions'], config['arm_folder'])
                result['source_commit'] = config['source_commit']
                atomic_json(Path(config['arm_folder'])/'ADMISSION.json', result)
                return result
            if config.get('action', 'learn') != 'learn':
                raise ValueError('Unknown worker action')
            admission = json.loads((Path(config['arm_folder'])/'ADMISSION.json').read_text())
            if admission.get('status') != 'PASS' or admission.get('source_commit') != config['source_commit']:
                raise ValueError('Missing matching native admission receipt')
            import signal
            stop = [False]
            def request_stop(*_):
                stop[0] = True
            previous = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                return run_current(runtime, config['arm'], calibration['inputs'], record, packet['fit_questions'],
                                   packet['order'], config['arm_folder'], packet['seed'],
                                   resume=config.get('resume', False), stop_requested=lambda: stop[0],
                                   diagnostic=config.get('source6_diagnostic', False))
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        finally:
            Budget.check = original_check
