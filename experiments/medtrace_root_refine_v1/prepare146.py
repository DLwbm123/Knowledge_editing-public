"""Prepare now; final admission waits for sources, full8 outcome and delivery."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import full146_contract as contract
from followup146 import gate

UP = Path(os.environ['RUN_ROOT']).resolve()
RUN = Path(os.environ['FOLLOWUP_ROOT']).resolve()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); tmp.replace(path)


def source(task, anchors):
    native = [dict(task['native'], query_id=f"edit_{task['order']}_{i}", question=q,
                   audit_role='NATIVE' if i == 0 else 'FIT')
              for i, q in enumerate([task['native']['question']]+task['fit_questions'])]
    return native+[dict(r, audit_role='GFIT') for r in anchors if r['owner'] == task['order']]


def init():
    assert RUN != UP and RUN.parent.parent == UP.parent.parent
    with (RUN/'private/CODE_PREP_ONCE').open('x'):
        pass
    checks = contract.selfcheck()
    for path in (UP/'private/tools').glob('*.py'):
        if not (RUN/'private/tools'/path.name).exists():
            shutil.copy2(path, RUN/'private/tools'/path.name)
    for name in ('SOURCE_COMMIT.json', 'EVAL_BINDINGS.json', 'OPTIMIZER160_LOCK.json', 'HELD_BASE.json'):
        shutil.copy2(UP/'private'/name, RUN/'private'/name)
    for name in ('source', 'official_llava', 'cpu_gate', 'derived_inputs'):
        path = RUN/'private'/name
        assert not path.exists()
        path.symlink_to((UP/'private'/name).resolve(), target_is_directory=True)
    for folder in ('results/C', 'results/M1', 'results/REFINE', 'candidates/C', 'candidates/M1',
                   'candidates/REFINE', 'resume146', 'retired146', 'calibration146', 'outputs'):
        (RUN/'private'/folder).mkdir(parents=True, exist_ok=True)
    env = read(UP/'private/LAUNCH_ENV.json')
    env.update(UPSTREAM_ROOT=str(UP), RUN_ROOT=str(RUN), FOLLOWUP_ROOT=str(RUN))
    write(RUN/'private/LAUNCH_ENV.json', env)
    original = read(RUN/'private/ORIGINAL146_TASKS.json')
    paper = Path(env['PAPER_PARENT'])
    queries = read(Path(env['GATE_PREVIOUS'])/'private/PAPER_QUERIES.json')['146']
    mask = read(paper/'private/EVAL_LEDGER.json')['Base_correctness']
    write(RUN/'private/EVAL_QUERIES146.json', queries)
    write(RUN/'private/ORIGINAL_BASE_MASK146.json', mask)
    qualification = read(Path(env['FP32_BASELINE_PARENT'])/'private/FP32_SEMANTIC_QUALIFICATION.json')['rows']
    forbidden = {r['source_group'] for r in queries.values()}
    forbidden.update(t['native']['source_group'] for t in original)
    forbidden.update(v['row']['source_group'] for v in qualification)
    revised = read(Path(env['SUPPLEMENT_PARENT'])/'private/REVISED_EVENT_ROWS.json')
    forbidden.update(v['source_group'] for v in revised)
    forbidden.update(v.split(':')[0] for v in list(forbidden) if v.startswith('PMC'))
    write(RUN/'private/FORBIDDEN_SOURCE_GROUPS146.json', sorted(forbidden))
    routers = {str(t['order']): str(paper/'private/edits'/t['anonymous_edit']/'ROUTER.pt') for t in original}
    assert all(Path(p).is_file() for p in routers.values())
    write(RUN/'private/ROUTER_BINDINGS146.json', routers)
    # Placeholder rows exercise coverage only; never freeze them as training data.
    dummy = [dict(query_id=f'UNQUALIFIED_GFIT146_{i}_{j}', owner=i, source_group=f'UNQUALIFIED_{i}_{j}')
             for i in range(1, 147) for j in range(2)]
    plan = contract.evaluation_plan(original, queries, dummy, mask)
    write(RUN/'private/EVAL_PLAN_TEMPLATE146.json', plan)
    histogram = {method: sum(v['method'] == method for v in plan) for method in ('M1', 'REFINE', 'BASE')}
    write(RUN/'public/FULL146_CODE_PREPARATION.json', dict(status='CPU_CODE_PREPARED_NOT_TRAINING_ADMITTED',
        selfcheck=checks, output_consumers=histogram, total_consumers=len(plan),
        original_R0_router_records=146, T2L_frozen_probes=54, T2L_frozen_edit_units=31,
        source_admission=False, training_started=False, new_model_calls=0, new_Judge=0,
        maximum_counts=contract.CAPS, epoch=time.time()))
    print(json.dumps(read(RUN/'public/FULL146_CODE_PREPARATION.json')))


def admit():
    assert gate(read(UP/'public/FULL_RESULTS.json'))
    assert read(UP/'private/FULL_COMPLETE.json')['status'] == 'COMPLETE'
    assert read(UP/'private/FULL_HANDOFF_EXIT1.json')['exit_code'] == 0
    assert read(UP/'public/FULL_EXECUTION_AUDIT.json')['status'] == 'PASS'
    assert read(UP/'private/FULL_DELIVERY.json')['remote_SHA_match']
    checks = read(RUN/'public/FULL146_CPU_CHECKS.json')
    assert checks['status'] == 'PASS' and checks['full_queue_and_report_integration']
    fisher_binding = read(RUN/'private/FULL146_INHERITED_FISHER_BINDING.json')
    assert fisher_binding['status'] == 'PASS' and fisher_binding['questions'] == 61
    inherited_fisher = RUN/'private/inherited/OPERATOR_FISHER.pt'
    stat = inherited_fisher.stat()
    assert (stat.st_dev, stat.st_ino) == (fisher_binding['device'], fisher_binding['inode'])
    assert read(RUN/'private/SOURCE_COMMIT.json') == read(UP/'private/SOURCE_COMMIT.json')
    original = read(RUN/'private/ORIGINAL146_TASKS.json')
    tasks = read(RUN/'private/FROZEN_TASKS146.json')
    anchors = read(RUN/'private/FROZEN_ANCHORS146.json')
    reviews = read(RUN/'private/SOURCE_REVIEW146.json')
    forbidden = set(read(RUN/'private/FORBIDDEN_SOURCE_GROUPS146.json'))
    admission = contract.source_review(original, tasks, anchors, reviews['owners'], forbidden)
    write(RUN/'public/FULL146_SOURCE_ADMISSION.json', admission)
    assert admission['status'] == 'PASS'
    for row in anchors:
        image = Path(row['image_path'])
        assert image.is_file() and image.stat().st_size > 1000
        record = next(v for v in reviews['owners'][str(row['owner'])] if v['candidate_id'] == row['candidate_id'])
        assert record['image_binding']['path'] == str(image)
        # Evidence paths stay private; require actual original annotation/full-text material.
        assert all(Path(v['path']).is_file() for v in record['source_evidence'])
    plan = contract.evaluation_plan(tasks, read(RUN/'private/EVAL_QUERIES146.json'), anchors,
                                    read(RUN/'private/ORIGINAL_BASE_MASK146.json'))
    assert shutil.disk_usage(RUN).free >= 8*1024**3
    with tempfile.NamedTemporaryFile(dir=RUN) as f:
        f.write(b'finite146-storage'); f.flush()
        assert Path(f.name).read_bytes() == b'finite146-storage'
    write(RUN/'private/STORAGE146.json', dict(free_bytes=shutil.disk_usage(RUN).free,
        mount=subprocess.check_output(['findmnt', '-T', str(RUN), '-J'], text=True), write_read=True))
    with (RUN/'private/ADMISSION146_ONCE').open('x'):
        pass
    env = read(RUN/'private/LAUNCH_ENV.json')
    existing = read(RUN/'private/EXISTING_SOURCE_BINDINGS.json')
    reuse = {}
    for task in tasks[:7]:
        prior = existing['tasks'][task['order']-1]
        if contract.task_binding(prior) != contract.task_binding(task):
            continue
        bound = contract.source_binding(source(task, anchors))
        if bound != contract.source_binding(source(prior, existing['anchors'])):
            continue
        for arm in ('C', 'M1', 'REFINE'):
            parent = Path(env['MODULE_PARENT']) if arm == 'M1' else UP
            path = (RUN/'private/inherited/REFINE'/f"{task['order']}.pt") if arm == 'REFINE' else parent/'private/candidates'/arm/f"{task['order']}.pt"
            result = parent/'private/results'/arm/f"{task['order']}.json"
            if path.is_file() and result.is_file():
                reuse[f"{arm}:{task['order']}"] = dict(weight=str(path), result=str(result),
                    task=contract.task_binding(task), source=bound)
    write(RUN/'private/REUSE146.json', reuse)
    lock = dict(stage='original146_refine', tasks=[contract.task_binding(t) for t in tasks],
        sources=contract.source_binding(anchors), arms=list(contract.ARMS), steps=160, parameters=7168,
        source_weights=[.25, .25, .5], source_order='native/cyclic4FIT/cyclic2GFIT',
        Fisher_fraction=.1, source_gain_fraction=.9, four_fractions=[1., .5, .25, .125],
        maximum_repairs=3, trust_fractions=[.25, .125, .0625], prefixes=list(contract.PREFIXES),
        original_mask=True, original_R0=True, zero_function_seeded_initialization=True)
    write(RUN/'private/SCOPE_LOCK.json', lock)
    write(RUN/'private/FULL146_BINDING.json', dict(source=digest([tasks, anchors, reviews]),
        upstream_full8_source='be310287487a72576157bbb84b491434d9a3ab42',
        fisher=read(UP/'public/FISHER_ADMISSION.json'), BASIS_lock=fisher_binding['BASIS_binding'],
        runtime_binding=fisher_binding['runtime_binding']))
    write(RUN/'private/EVAL_PLAN146.json', plan)
    ledger = read(UP/'RESOURCE_LEDGER.json')
    assert all(v.get('ended_epoch') for v in ledger['gpu_sessions'])
    inherited = {k: ledger[k] for k in ('gpu_seconds_used', 'Judge_attempts')}
    write(RUN/'private/INHERITED_COST.json', inherited)
    write(RUN/'RESOURCE_LEDGER.json', dict(inherited, gpu_sessions=[], weights_observed_peak_bytes=0))
    manifest = read(UP/'RUN_MANIFEST.json')
    manifest.update(stage='original146_refine', Judge_limit=inherited['Judge_attempts']+17373,
        starting_epoch=time.time(), owned_weight_limit_bytes=2*1024**3, min_free_bytes=8*1024**3)
    write(RUN/'RUN_MANIFEST.json', manifest)
    plan_config = read(UP/'PLAN_CONFIG.json'); plan_config.update(stage='original146_refine', maximum_counts=contract.CAPS)
    write(RUN/'PLAN_CONFIG.json', plan_config)
    write(RUN/'STAGE_CAP.json', read(UP/'STAGE_CAP.json'))
    write(RUN/'private/GPU_SOURCE_VERSION.json', dict(commit=os.environ['FULL146_COMMIT'],
        reused_finite_solver_source='be310287487a72576157bbb84b491434d9a3ab42'))
    write(RUN/'private/FULL146_READY.json', dict(status='PASS', source_gate=True, all8_compatibility=True,
        output_consumers=17373, maximum_counts=contract.CAPS, reusable_trajectories=len(reuse), epoch=time.time()))


def digest(value):
    import sys
    sys.path.insert(0, str(RUN/'private/tools'))
    import common
    return common.digest(value)


if __name__ == '__main__':
    if os.environ.get('ACTION') == 'prepare146_init': init()
    elif os.environ.get('ACTION') == 'prepare146_admit': admit()
    else: raise ValueError('Explicit CPU preparation/admission action required')
