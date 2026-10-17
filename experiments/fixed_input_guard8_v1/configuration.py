"""Preparation-only eight-edit contrast; no implicit launch or budget borrowing."""
VERSION = 'FIXED_INPUT_CONTEXT_PIN8_V2_20261017'
ARMS = ('U', 'P')
ORDERS = tuple(range(1, 9))
SEEDS = (1442709718, 39524989, 4278467101, 2255805473, 225034559, 173959200, 958324087, 1907321266)
SLOTS = 20
PREFIXES = (0, 4, 8)
LAYER = 'model.layers.21.mlp.down_proj'
OPTIMIZER = dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)
CAPS = dict(attempt_slots=160, admission_slots=1, backwards=161, candidate_checks=161,
            committed_updates=161, forwards=200000, generation_attempts=512, new_Judge=512)
GPU_SECONDS = 3 * 3600
WALL_SECONDS = 12 * 3600
CHECKPOINT_BYTES = 1280 * 1024**2


def require_execution(config):
    if config.get('version') != VERSION:
        raise ValueError('Wrong execution contract')
    auth = config.get('authorization', {})
    if not all(auth.get(k) is True for k in ('execution', 'model_calls', 'hardware')):
        raise PermissionError('PREPARATION_ONLY')
    gates = config.get('admission', {})
    for gate in ('calibration_disjoint', 'source_order', 'storage', 'hardware', 'runtime_binding'):
        if gates.get(gate) != 'PASS':
            raise PermissionError('Missing admission: ' + gate)
    if config.get('arm') not in ARMS or config.get('native_dtype') not in ('float16', 'bfloat16'):
        raise ValueError('Invalid arm/native dtype')


def schedule():
    for arm in ARMS:
        yield arm, 'base', 0
        yield arm, 'admit', 1
        for owner in ORDERS:
            yield arm, 'learn', owner
            yield arm, 'insertion', owner
            if owner in PREFIXES:
                yield arm, 'prefix', owner
        yield arm, 'reload', 8


def evaluation_panel(tasks, ledger, prefix, protection):
    if [t['order'] for t in tasks] != list(ORDERS) or prefix not in PREFIXES:
        raise ValueError('Wrong cohort/prefix')
    if len(protection) > 54 or len(set(protection)) != len(protection):
        raise ValueError('Invalid fixed protection panel')
    ids = list(protection)
    for task in tasks if prefix == 0 else tasks[:prefix]:
        ids.append(task['edit_id'])
        for name in ('T1G', 'T2G'):
            probes = [q for e in task['events'] if e['task'] == name for q in e['all_probe_query_ids']]
            if len(probes) > 4:
                raise ValueError('Panel exceeds frozen four-probe bound; do not truncate')
            ids.extend(probes)
    if any(q not in ledger for q in ids):
        raise ValueError('Missing evaluation binding')
    return list(dict.fromkeys(ids))
