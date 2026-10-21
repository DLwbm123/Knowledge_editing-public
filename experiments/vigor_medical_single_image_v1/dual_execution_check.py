"""Control-path unit checks only: no model instance, optimizer or generation."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from .dual_execution import ExecutionBudget, Journal, public_event, validate_manifest
from .dual_guard import birth_tick, math_finite_order
from .dual_scoring import prepare_packets


def check():
    config = json.loads((Path(__file__).resolve().parents[2] / 'reports/vigor_dualSolver_v1_20261021/FREEZE.json').read_text())
    config['source_binding'] = 'UNIT_BINDING'
    record = SimpleNamespace(image_path=Path('/unit-input'), question='UNIT_QUESTION', target='UNIT_REFERENCE')
    generation = {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 1024, 'use_cache': True}
    runtime = {'model_snapshot': 'UNIT_SNAPSHOT', 'module': config['module'], 'native_dtype': 'float16'}
    manifest = {'source': {'binding_id': 'UNIT_BINDING', 'current_use_permitted': True, 'reserved_exclusion_certified': True, 'fit_exclusion_certified': True, 'split': 'train', 'record': {'image_path': str(record.image_path.resolve()), 'question': record.question, 'target': record.target}}, 'F0': runtime, 'KV': dict(runtime, source_binding='UNIT_BINDING', status='PASS', actual_class='LlavaMistralForCausalLM', generation=generation), 'generation': generation}
    checks = {}
    checks['manifest_exact_binding'] = validate_manifest(manifest, config, record) == 'UNIT_BINDING'
    for field in ('current_use_permitted', 'reserved_exclusion_certified', 'fit_exclusion_certified'):
        bad = copy.deepcopy(manifest); bad['source'][field] = False
        try: validate_manifest(bad, config, record)
        except PermissionError: checks[field + '_missing_rejected'] = True
        else: raise AssertionError(field)
    bad = dict(config, lr=.02)
    try: validate_manifest(manifest, bad, record)
    except PermissionError: checks['science_config_mutation_rejected'] = True
    else: raise AssertionError('mutation')
    bad = copy.deepcopy(manifest); bad['KV']['status'] = 'CPU_ONLY_PASS'
    try: validate_manifest(bad, config, record)
    except PermissionError: checks['CPU_not_real_KV_pass'] = True
    else: raise AssertionError('CPU gate')
    with tempfile.TemporaryDirectory(prefix='q.') as parent:
        journal = Journal(Path(parent) / 'run', config['disk_bytes_max'])
        journal.event({'status': 'UNIT_CHECK'})
        for row in ({'raw_answer': 'PRIVATE'}, {'basis_private': []}):
            try: journal.event(row)
            except ValueError: pass
            else: raise AssertionError('private public field')
        journal.private('RESERVED.json', {'attempts': 1})
        try: journal.private('RESERVED.json', {})
        except FileExistsError: checks['reservation_no_replay'] = True
        else: raise AssertionError('replay')
        checks['private_fields_not_written'] = 'PRIVATE' not in (journal.root / 'public/EVENTS.jsonl').read_text()
    budget = ExecutionBudget(config, time.time() - 2)
    checks['original_clock_preserved'] = time.monotonic() - budget.started >= 2
    budget.begin_generation()
    for _ in range(128): budget.before_forward(None, None)
    try: budget.before_forward(None, None)
    except TimeoutError: checks['per_request_forward_cap'] = budget.forwards == 128
    else: raise AssertionError('request cap')
    budget.generating = False
    budget.forwards = 780
    try: budget.before_forward(None, None)
    except TimeoutError: checks['shared_work_reserve_protected'] = True
    else: raise AssertionError('work reserve')
    budget.restoring = True; budget.before_forward(None, None)
    checks['restore_uses_same_counter'] = budget.forwards == 781
    rows = [{'alias': f'{a}_{i}', 'status': 'COMPLETE', 'question': 'UNIT_Q', 'gold_answer': 'UNIT_G', 'raw_base_answer': 'UNIT_A'} for a in ('IND2', 'JOINT2') for i in range(2)]
    packets, mapping, missing = prepare_packets(rows)
    checks['exact_current_duplicate_scored_once'] = len(packets) == 1 and len(next(iter(mapping.values()))) == 4 and not missing
    rows[0]['status'] = 'INCOMPLETE_NO_RETRY'
    packets, mapping, missing = prepare_packets(rows)
    checks['missing_not_replaced_or_zero_scored'] = missing == ['IND2_0'] and sum(map(len, mapping.values())) == 3
    stat = '42 (neutral process (x)) ' + ' '.join(['S'] + ['0'] * 18 + ['123456'] + ['0'] * 3)
    checks['birth_identity_handles_comm_spaces'] = birth_tick(stat) == '123456'
    checks['expired_guard_clock_rejected'] = not math_finite_order(0, 20, 10, 30)
    for name in ('dual_solver.py', 'dual_execution.py', 'dual_guard.py', 'dual_scoring.py'):
        ast.parse((Path(__file__).parent / name).read_text())
    checks['changed_modules_parse'] = True
    assert all(checks.values()), checks
    return {'status': 'PASS_CONTROL_PATH_ONLY', 'checks': checks, 'model_instances': 0, 'model_forward': 0, 'GPU': 0, 'Judge': 0, 'optimizer_runs': 0, 'guards_spawned': 0, 'signals_sent': 0}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
