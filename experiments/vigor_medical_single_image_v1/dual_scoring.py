"""Once-only terminal scoring handoff; no historical output reads or GPU calls."""
import json
import os
from pathlib import Path
import random
import time
import uuid

from .dual_execution import ARMS


def prepare_packets(rows):
    expected = {f'{arm}_{i}' for arm in ARMS for i in range(2)}
    if len(rows) != 4 or {x['alias'] for x in rows} != expected:
        raise ValueError('Need exactly the four fixed final endpoints')
    records, mapping, keys, missing = [], {}, {}, []
    for row in rows:
        if row['status'] != 'COMPLETE':
            missing.append(row['alias'])
            continue
        key = (row['question'], row['gold_answer'], row['raw_base_answer'])
        if key not in keys:
            opaque = uuid.uuid4().hex
            keys[key] = opaque
            mapping[opaque] = []
            records.append(dict(opaque_query_id=opaque, question=key[0], gold_answer=key[1], raw_base_answer=key[2]))
        mapping[keys[key]].append(row['alias'])
    random.SystemRandom().shuffle(records)
    return records, mapping, missing


def score_once(run_root, scorer_source, cli, closed_receipt):
    """Local CPU scorer stage only, explicitly invoked after GPU restoration receipt."""
    root = Path(run_root)
    execution = json.loads((root / 'private' / 'EXECUTION_BINDING.json').read_text())
    guard = execution['manifest']['guard']
    if (closed_receipt.get('verified_exit') is not True or closed_receipt.get('gpu_absent') is not True
            or closed_receipt.get('pid') != guard['target_pid'] or closed_receipt.get('birth') != guard['target_birth']):
        raise PermissionError('Need current experiment process-exit/GPU-absence receipt')
    if time.time() + 600 >= execution['manifest']['start_epoch'] + execution['config']['whole_seconds']:
        raise TimeoutError('Insufficient original overall clock for one scoring attempt')
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) >= execution['config']['disk_bytes_max'] - 16 * 1024**2:
        raise OSError('Scoring storage reserve unavailable')
    terminal = json.loads((root / 'private' / 'NUMERIC_TERMINAL.json').read_text())
    events = [json.loads(line) for line in (root / 'public' / 'EVENTS.jsonl').read_text().splitlines()]
    if terminal['status'] != 'COMPLETE' or events[-1]['status'] != 'RESTORED':
        raise PermissionError('Need terminal GPU restoration before scoring')
    path = root / 'private' / 'SCORE_RESERVED.json'
    with path.open('x') as f:
        json.dump({'attempts': 1, 'maximum_unique_payloads': 4}, f)
        f.flush(); os.fsync(f.fileno())
    rows = [json.loads((root / 'private' / f'{arm}_{i}_GENERATION.json').read_text()) for arm in ARMS for i in range(2)]
    records, mapping, missing = prepare_packets(rows)
    packet = root / 'private' / 'SCORE_PACKETS.json'
    with packet.open('x') as f:
        json.dump({'records': records}, f)
    with (root / 'private' / 'SCORE_MAPPING.json').open('x') as f:
        json.dump({'mapping': mapping, 'missing': missing}, f)
    summary = {'status': 'NO_COMPLETE_OUTPUTS', 'decisions': []}
    if records:
        config = {'scorer_source': str(scorer_source), 'private_root': str(root / 'private' / 'scoring'), 'packets': str(packet), 'payload_max': len(records), 'attempts': 1, 'cli': str(cli)}
        config_file = root / 'private' / 'SCORE_CONFIG.json'
        with config_file.open('x') as f:
            json.dump(config, f)
        from .factF3_judge import main as judge_main
        import sys
        sys.path.insert(0, str(scorer_source))
        from scripts.medtrace import stage17_judge as engine
        previous_env = os.environ.get('CONFIG_FILE')
        old_profile, old_flags = engine.profile, engine.flags
        try:
            os.environ['CONFIG_FILE'] = str(config_file)
            judge_main()
            summary = json.loads((root / 'private' / 'scoring' / 'SUMMARY.json').read_text())
        except Exception as error:
            summary = {'status': 'SCORING_FAILED_WITH_MISSING', 'decisions': [], 'exception_type': type(error).__name__}
        finally:
            engine.profile, engine.flags = old_profile, old_flags
            if previous_env is None:
                os.environ.pop('CONFIG_FILE', None)
            else:
                os.environ['CONFIG_FILE'] = previous_env
    scores = {}
    for decision in summary['decisions']:
        if decision['opaque_query_id'] not in mapping or type(decision['is_correct']) is not bool:
            raise ValueError('Unexpected semantic decision')
        for alias in mapping[decision['opaque_query_id']]:
            if alias in scores:
                raise ValueError('Duplicate semantic decision')
            scores[alias] = decision['is_correct']
    arms = {}
    for arm in ARMS:
        grades = [scores.get(f'{arm}_{i}') for i in range(2)]
        numeric = terminal['arms'][arm]['numeric_pair_pass']
        pair = False if not numeric or False in grades else None if None in grades else True
        arms[arm] = {'numeric_pair_pass': numeric, 'semantic_scores': grades, 'H1_pair_pass': pair, 'H2': 'NOT_MEASURED', 'H3': 'NOT_MEASURED'}
    public = {'status': summary['status'], 'arms': arms, 'unique_payloads': len(records), 'attempts_per_payload': 1, 'retries': 0, 'missing_endpoints': [f'{arm}_{i}' for arm in ARMS for i in range(2) if f'{arm}_{i}' not in scores]}
    with (root / 'public' / 'SCORES.json').open('x') as f:
        json.dump(public, f, indent=2)
    return public
