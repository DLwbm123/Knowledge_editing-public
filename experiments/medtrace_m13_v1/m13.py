"""One M13 arm; reuse the frozen implementation and all completed comparators."""
import os
from pathlib import Path
import time
import traceback
# The runtime captures this path at import, before train() calls load().
os.environ['M3BENCH_LLAVA_SOURCE'] = str(Path(os.environ['RUN_ROOT']) / 'private/official_llava')
import module_run as r
from module_recovery import admitted_stage

c, RUN = r.c, r.RUN
PARENT = Path(os.environ['MODULE_PARENT'])
r.mm.ARMS = ('C', 'M13')
r.GPUS = (2, 4, 7)
if os.environ['ACTION'] == 'module_fisher':
    r.GPUS = (int(os.environ['GPU']), 4, 7)


def admission():
    assert c.read(PARENT / 'EXIT.json')['exit_code'] == 0
    assert c.read(PARENT / 'private/REFERENCE_EXACT.json')['status'] == 'PASS'
    assert c.read(PARENT / 'public/RESULTS.json')['missing_consumers'] == 0
    assert r.mm.flags('M13') == (True, False, True)
    tasks = r.s.tasks(); assert [t['order'] for t in tasks] == list(range(1, 9))
    for task in tasks:
        state = r.p.load_state(r.s.point(task, 'C'))
        assert sum(v.numel() for v in state['states']['BASE'].values()) == 7168
        reference = c.read(RUN / 'private/results/C' / f"{task['order']}.json")
        assert len(reference['curve']) == 160 and reference['save_load_exact']
    c.write(RUN / 'private/REFERENCE_EXACT.json', dict(status='PASS', candidates=8,
        prior_exact_source_outputs=56, retained_C_reused=True, training_reconstruction=False,
        prior_receipt=str(PARENT / 'private/REFERENCE_EXACT.json')))


def controller():
    assert not (RUN / 'private/M13_START.json').exists(), 'No implicit controller restart'
    c.write(RUN / 'private/M13_START.json', dict(epoch=time.time(), pid=os.getpid()))
    admission()
    c.write(RUN / 'public/PROGRESS.json', dict(status='M13_OPERATOR_FISHER_REBUILD_RUNNING', complete=False))
    admitted_stage('m13.py', 'module_fisher', [0])
    c.write(RUN / 'public/PROGRESS.json', dict(status='M13_TRAINING_RUNNING', complete=False))
    admitted_stage('m13.py', 'module_train', range(3))
    finish()


def finish():
    paths = [r.s.point(task, 'M13') for task in r.s.tasks()]
    assert len(paths) == 8 and all(p.is_file() and not p.is_symlink() for p in paths)
    for task in r.s.tasks():
        result = c.read(RUN / 'private/results/M13' / f"{task['order']}.json")
        assert len(result['curve']) == 160 and result['save_load_exact']
        calibration = c.read(RUN / 'private/calibration' / f"{task['order']}.json")
        previous = c.read(PARENT / 'private/calibration' / f"{task['order']}.json")
        assert abs(calibration['coefficient'] - previous['coefficient']) <= 1e-12 + 1e-4 * abs(previous['coefficient'])
    counts = [c.read(p) for p in (RUN / 'private').glob('COUNTS_*.json')]
    assert sum(x['updates'] for x in counts) == 1280
    assert sum(x['backwards'] for x in counts) == 5792
    assert sum(x['operator_backwards'] for x in counts) == 1280
    c.write(RUN / 'private/CANDIDATES_FROZEN.json', dict(candidates=8, epoch=time.time(), selection=False))
    cache = RUN / 'private/OPERATOR_FISHER.pt'; assert not cache.is_symlink()
    c.write(RUN / 'private/FISHER_CONSUMED.json', dict(bytes=cache.stat().st_size,
        reason='All eight M13 trajectories and final operator-penalty consumers complete'))
    cache.unlink()
    c.write(RUN / 'public/PROGRESS.json', dict(status='M13_ALL_FROZEN_EVALUATION_RUNNING', complete=False))
    admitted_stage('m13.py', 'module_evaluate', range(3))
    admitted_stage('m13.py', 'module_route', [0])
    outputs = list((RUN / 'private/outputs').glob('*/*/*.json'))
    counts = [c.read(p) for p in (RUN / 'private').glob('COUNTS_*.json')]
    assert len(outputs) == 1158 and sum(x['generation_attempts'] for x in counts) <= 1158
    assert sum(x['non_generation_forwards'] for x in counts) <= 10000
    c.write(RUN / 'private/MODULE_GENERATION_COMPLETE.json', dict(outputs=1158, counts=counts, epoch=time.time()))
    finish_generated()


def finish_generated():
    import pipeline
    paths = [r.s.point(task, 'M13') for task in r.s.tasks()]
    pipeline.wait([pipeline.launch('m13_queue.py', 'module_ingest')])
    c.write(RUN / 'private/CANDIDATES_CONSUMED.json', dict(candidates=8,
        bytes=sum(p.stat().st_size for p in paths), reason='Forced/natural outputs, save/load and durable scoring bindings complete'))
    for path in paths:
        path.unlink()
    root = RUN / 'private/judge_m13_astra_medium'
    c.write(RUN / 'public/PROGRESS.json', dict(status='M13_GENERATED_SCORING_PENDING', complete=False))
    while not (root / 'ALL_WORKERS_COMPLETE.json').exists():
        c.budget(); assert not list((root / 'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(30)
    pipeline.wait([pipeline.launch('m13_report.py', 'm13_report')])


def generated_admission():
    from module_recovery import live
    failure = c.read(RUN / 'private/FAILURE_m13_m13_recover_unstarted_0.json')
    assert failure['error'] == 'NameError("name \'pipeline\' is not defined")'
    assert bool(failure['counts']) and all(v == 0 for v in failure['counts'].values())
    assert c.read(RUN / 'EXIT.json')['exit_code'] == 1 and not live(c.read(RUN / 'START.json'))
    ledger = c.read(RUN / 'RESOURCE_LEDGER.json')
    assert len(ledger['gpu_sessions']) == 11 and all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    assert c.read(RUN / 'private/CANDIDATES_FROZEN.json')['candidates'] == 8
    generation = c.read(RUN / 'private/MODULE_GENERATION_COMPLETE.json')
    assert generation['outputs'] == 1158 and len(list((RUN / 'private/outputs').glob('*/*/*.json'))) == 1158
    totals = {k: sum(x[k] for x in generation['counts']) for k in r.COUNTS}
    assert totals['updates'] == 1280 and totals['backwards'] == 5792 and totals['operator_backwards'] == 1280
    assert totals['generation_attempts'] <= 1158 and totals['non_generation_forwards'] <= 10000
    assert ledger['Judge_attempts'] == c.read(RUN / 'private/INHERITED_COST.json')['Judge_attempts']
    assert not (RUN / 'private/CANDIDATES_CONSUMED.json').exists()
    for task in r.s.tasks():
        result = c.read(RUN / 'private/results/M13' / f"{task['order']}.json")
        assert len(result['curve']) == 160 and result['save_load_exact']
        assert r.s.point(task, 'M13').is_file() and not r.s.point(task, 'M13').is_symlink()
    return dict(status='PASS', complete_training_and_generation=True, new_GPU_work=0,
                new_Judge_before_takeover=0, epoch=time.time())


def recover_generated():
    evidence = generated_admission()
    archive = RUN / 'private/generated_handoff_recovery1'; archive.mkdir(exist_ok=False)
    c.write(archive / 'ADMISSION.json', evidence)
    for path in (RUN / 'START.json', RUN / 'EXIT.json', RUN / 'private/FAILURE_m13_m13_recover_unstarted_0.json'):
        path.rename(archive / path.name)
    import shutil
    shutil.copy2(RUN / 'logs/import_recovery_controller.log', archive / 'controller.log')
    c.write(RUN / 'START.json', c.read(RUN / 'private/GENERATED_HANDOFF_START.json'))
    finish_generated()


if __name__ == '__main__':
    try:
        action = os.environ['ACTION']
        if action == 'module_evaluate':
            r.evaluate(expected_candidates=8)
        elif action == 'module_route':
            import module_route
            module_route.main(expected_candidates=8)
        elif action == 'm13_recover_unstarted':
            import m13_recovery
            m13_recovery.main()
        elif action == 'm13_generated_preflight':
            print(generated_admission())
        elif action == 'm13_recover_generated':
            recover_generated()
        else:
            {'m13_controller': controller, 'module_train': r.train, 'module_fisher': r.fisher}[action]()
    except BaseException as error:
        c.write(RUN / 'private' / f"FAILURE_m13_{os.environ.get('ACTION')}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error), traceback=traceback.format_exc(), counts=r.COUNTS, epoch=time.time()))
        c.write(RUN / 'public/PROGRESS.json', dict(status='M13_STOPPED_RETAINED', complete=False))
        raise
