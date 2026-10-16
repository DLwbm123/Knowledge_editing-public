"""Finite first-eight deployment: separate current-source and evaluation processes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .configuration import ARMS, CAPS, GPU_SECONDS, SEEDS, VERSION, evaluation_panel


def plan():
    yield 'P', 'calibration', 0
    for arm in ARMS:
        yield arm, 'base', 0
        yield arm, 'admit', 1
    for arm in ARMS:
        for owner in range(1, 9):
            yield arm, 'learn', owner
            yield arm, 'insertion', owner
            if owner in (4, 8):
                yield arm, 'prefix', owner
        yield arm, 'reload', 8


def helpers():
    from experiments.purew_original146_v1 import common as c
    return c, c.RUN


def install_budget(arm):
    from .executor import Budget
    c, root = helpers()
    original = Budget.check
    original_consume = Budget.consume
    def check(self):
        original(self)
        c.budget()
        used = sum(s.get('resident_seconds', time.time()-s['started_epoch'])
                   for s in c.read(root/'RESOURCE_LEDGER.json')['gpu_sessions'] if s.get('arm') == arm)
        if used >= GPU_SECONDS:
            raise TimeoutError('Per-arm GPU ceiling')
        folder = root/'private/fixed8'/arm
        rows = [c.read(p)['counts'] for p in folder.glob('*budget.json')]
        for key, cap in CAPS.items():
            if sum(r[key] for r in rows) > cap:
                raise TimeoutError('Combined learner/evaluator budget: '+key)
    Budget.check = check
    def consume(self, key):
        rows = [c.read(p)['counts'] for p in (root/'private/fixed8'/arm).glob('*budget.json')]
        if sum(r[key] for r in rows) >= CAPS[key]:
            raise TimeoutError('Combined budget exhausted before call: '+key)
        original_consume(self, key)
    Budget.consume = consume
    return Budget


def image_path(path):
    text = str(path)
    if '/DataP/' in text:
        return Path(os.environ['DATA_ROOT'])/text.split('/DataP/', 1)[1]
    return Path(text)


def provide(owner, arm):
    c, root = helpers()
    task = c.read(root/'private/BENCHMARK146_QUEUE.json')['tasks'][owner-1]
    assert task['seed'] == SEEDS[owner-1]
    folder = root/'private/window'/arm
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.iterdir():
        assert p.is_file() and not p.is_symlink()
        p.unlink()
    native = task['native']
    source = image_path(native['image_path'])
    target = folder/('current'+source.suffix)
    shutil.copyfile(source, target)
    record = dict(record_id=task['edit_id'], dataset=native['dataset'], question=native['question'],
                  target=native['reference'], official_rephrase=task['fit_questions'][0],
                  image_path=str(target), relative_image_path=native['original_image_path'],
                  formal_sequence_position=owner, question_type='VERIFIED_SOURCE_ANSWER',
                  router_positive_source='NATIVE_ONLY_CONSERVATIVE_FIT_NOT_OFFICIAL_EVALUATION_REPHRASE')
    c.write(folder/'current.json', dict(role='SOURCE_EDIT', order=owner, seed=task['seed'],
                                     record=record, fit_questions=task['fit_questions']))


def generate(runtime, bindings, row, destination, phase, budget):
    import torch
    c, root = helpers()
    if destination.exists() or destination.with_suffix('.STARTED.json').exists():
        raise FileExistsError('Generation cannot be replayed')
    raw = runtime.adapter.prepare_inputs(image_path(row['image_path']), row['question'], None)
    bound = bindings[row['opaque_Base_id']]
    assert raw['image_sha256'] == bound['image_sha256']
    assert raw['input_ids'].tolist() == [bound['prompt_ids']]
    assert raw['attention_mask'].tolist() == [bound['attention_mask']]
    c.write(destination.with_suffix('.STARTED.json'), dict(phase=phase, epoch=time.time()))
    budget.consume('generation_attempts')
    with torch.inference_mode():
        value = runtime.adapter.generate_prepared_with_result(raw, runtime.generation_config)
    result = dict(binding=dict(input=row, judge_input=bound, phase=phase),
                  R0=dict(raw_answer=value.decoded_text, raw_token_ids=list(value.raw_token_ids)),
                  added_parameters=0, native_inference=True)
    c.write(destination, result)
    return result


def auxiliary(arm, action, owner, config):
    import torch
    from .worker import calibration_inputs, load_native
    from .adapter import RuntimeAdapter
    c, root = helpers()
    Budget = install_budget(arm)
    folder = root/'private/fixed8'/arm
    budget = Budget(folder/('budget.json' if action == 'calibration' else 'evaluation-budget.json'))
    gpu = config['physical_GPU']
    with c.lease(gpu):
        with c.resources() as r:
            next(s for s in r['gpu_sessions'] if s['pid'] == os.getpid() and not s.get('ended_epoch'))['arm'] = arm
        runtime, bindings = c.load(gpu)
        if action == 'calibration':
            from m3bench_repro.editors.llava_runtime import EditorRecord
            records = []
            for row in c.read(root/'private/calibration_records.json'):
                row['image_path'] = Path(row['image_path'])
                records.append(EditorRecord(**row))
            inputs = calibration_inputs(runtime, records, config['calibration_manifest'], budget)
            torch.save(dict(inputs=inputs, manifest=config['calibration_manifest'],
                            native_dtype=str(runtime.get_module(config['layer']).weight.dtype), layer=config['layer']),
                       root/'private/calibration_inputs.pt')
            c.write(root/'public/CALIBRATION_COMPLETE.json', dict(status='PASS', records=32, shape=list(inputs.shape)))
            return
        hooks = RuntimeAdapter(runtime, budget)
        try:
            for p in runtime.model.parameters():
                p.requires_grad_(False)
            all_tasks = c.read(root/'private/BENCHMARK146_QUEUE.json')['tasks']
            tasks, ledger = all_tasks[:8], c.read(root/'private/EVAL_LEDGER.json')['queries']
            if action != 'base':
                load_native(runtime, folder/'native.pt', arm, owner)
            if action == 'base':
                from experiments.purew_original146_v1.contract import base_completion
                for index in (1, 50, 100):
                    row = all_tasks[index-1]['native']
                    value = generate(runtime, bindings, row, root/'private/evaluation'/arm/'identity'/(str(index)+'.json'),
                                     dict(arm=arm, stage='BASE_IDENTITY'), budget)
                    assert value['R0'] == base_completion(bindings[row['opaque_Base_id']]['output'])
                c.write(root/'public'/('BASE_IDENTITY_'+arm+'.json'), dict(status='PASS', outputs=3))
            phase = dict(arm=arm, mode=action, owner=owner if action == 'insertion' else 0,
                         prefix=owner, code=config['implementation_commit'])
            output = root/'private/outputs'/arm/action/str(owner)
            if action == 'reload':
                ids = [tasks[0]['edit_id'], next(e['all_probe_query_ids'][0] for e in tasks[0]['events'] if e['task'] == 'T2G')]
            elif action == 'insertion':
                task = tasks[owner-1]
                ids = list(dict.fromkeys([task['edit_id']]+[q for e in task['events'] if e['task'] == 'T2G' for q in e['all_probe_query_ids']]))
            else:
                ids = evaluation_panel(tasks, ledger, owner, c.read(root/'private/protection_ids.json'))
            for q in ids:
                value = generate(runtime, bindings, ledger[q], output/(q+'.json'), phase, budget)
                if action == 'reload':
                    assert value['R0'] == c.read(root/'private/outputs'/arm/'prefix/8'/(q+'.json'))['R0']
            c.write(output/'COMPLETE.json', dict(status='PASS', phase=phase, outputs=len(ids)))
            if action in ('insertion', 'prefix'):
                c.write(folder/'CONSUMED.json', dict(owner=owner, insertion_complete=True,
                                                   prefix_complete=action == 'prefix'))
            if action == 'reload':
                c.write(root/'private'/('FINAL_CONSUMED_'+arm+'.json'), dict(status='PASS', two_outputs_exact=True))
        finally:
            hooks.close()


def child():
    import torch
    c, root = helpers()
    arm, action, owner = os.environ['ARM'], os.environ['ACTION_KIND'], int(os.environ['OWNER'])
    config = c.read(root/'private/config.json')
    config.update(arm=arm, action=action, arm_folder=str(root/'private/fixed8'/arm),
                  current_source=str(root/'private/window'/arm/'current.json'))
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    if action in ('admit', 'learn'):
        from .worker import main
        install_budget(arm)
        result = main(config)
        c.write(root/'private/control'/arm/(action+'_'+str(owner)+'.json'), result)
    else:
        auxiliary(arm, action, owner, config)


def controller():
    c, root = helpers()
    try:
        for arm, action, owner in plan():
            c.budget()
            if action in ('admit', 'learn'):
                provide(owner, arm)
            label = action+'_'+arm+'_'+str(owner)
            env = dict(os.environ, ARM=arm, ACTION=label, ACTION_KIND=action, OWNER=str(owner), ENTRY_MODE='child')
            c.write(root/'public/PROGRESS.json', dict(stage=label, status='RUNNING', epoch=time.time()))
            with (root/'logs'/(label+'.log')).open('ab') as log:
                process = subprocess.Popen([env['TRAIN_PYTHON'], '-u', env['NEUTRAL_ENTRY']], env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                c.write(root/'private/control'/(label+'.json'), dict(pid=process.pid, started_epoch=time.time()))
                try:
                    code = process.wait(timeout=max(1, c.read(root/'RUN_MANIFEST.json')['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if code:
                raise RuntimeError('Stage failed without retry: '+label+' exit='+str(code))
            if action == 'reload':
                assert c.read(root/'private'/('FINAL_CONSUMED_'+arm+'.json'))['status'] == 'PASS'
                active = root/'private/fixed8'/arm/'active.pt'
                assert not active.is_symlink()
                receipt = dict(bytes=active.stat().st_size, reason='All insertion/prefix/native-reload consumers durable',
                               retained='current native export', regeneration_requires_training=True)
                active.unlink()
                c.write(root/'public'/('OPTIMIZER_RETIRED_'+arm+'.json'), receipt)
        c.write(root/'private/GENERATION_COMPLETE.json', dict(status='PASS', scoring_pending=True))
        c.write(root/'public/PROGRESS.json', dict(status='GENERATION_COMPLETE_SCORING_PENDING', epoch=time.time()))
    except BaseException:
        c.write(root/'private/failures/controller.json', dict(traceback=traceback.format_exc(), epoch=time.time()))
        c.write(root/'public/PROGRESS.json', dict(status='STOPPED_FAILURE_NO_RETRY', epoch=time.time()))
        raise


if __name__ == '__main__':
    (child if os.environ['ENTRY_MODE'] == 'child' else controller)()
