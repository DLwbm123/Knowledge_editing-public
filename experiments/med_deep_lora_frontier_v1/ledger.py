"""Freeze a bounded, explicitly incomplete index of available scoring evidence."""
import json
from pathlib import Path
import history_index
import partition
import score_index


def build(destination, baseline_bindings, prior_decisions, fixed_packet, fixed_decisions, manifests):
    destination = Path(destination)
    base = json.loads(Path(baseline_bindings).read_text())['rows']
    if len(base) != 508:
        raise ValueError('Fixed baseline coverage mismatch')
    keys = {r[arm]['payload_key'] for r in base for arm in ('BASE', 'MED')}
    valid = score_index.decisions(prior_decisions, keys)
    prior_packet = json.loads(Path(fixed_packet).read_text())
    if prior_packet.get('status') != 'SCORE_PACKET_FROZEN' or len(prior_packet['consumers']) != 1016:
        raise ValueError('Prior fixed-depth packet mismatch')
    prior_scores = json.loads(Path(fixed_decisions).read_text())
    fixed_valid = prior_scores['decisions']
    if set(fixed_valid) & set(prior_scores['unknown']):
        raise ValueError('Historical result conflicts with unknown status')
    conflicts = set()
    for key, row in fixed_valid.items():
        if key in valid and any(valid[key][k] != row[k] for k in partition.FIELDS):
            conflicts.add(key)
        else:
            valid[key] = row
    semantic_valid, semantic_attempts = {}, set()
    for key, record in prior_packet['payloads'].items():
        identity = partition.semantic(record)
        if key in fixed_valid:
            semantic_valid.setdefault(identity, []).append(fixed_valid[key])
        elif key in prior_scores['unknown']:
            semantic_attempts.add(identity)
    attempts = set()
    for manifest_path in manifests:
        manifest_path = Path(manifest_path)
        batches = history_index.manifest(manifest_path)
        attempts.update(history_index.attempted(manifest_path.parent, batches))
    result = dict(status='AVAILABLE_HISTORY_INDEX_FROZEN', history_coverage_complete=False,
                  valid=valid, conflicts=sorted(conflicts), attempts=sorted(attempts), semantic_valid=semantic_valid,
                  semantic_attempts=sorted(semantic_attempts), manifests_indexed=len(manifests))
    with destination.open('x') as f:
        json.dump(result, f, ensure_ascii=False)
    destination.chmod(0o600)
    return dict(status=result['status'], valid=len(valid), attempts=len(attempts),
                semantic_valid=len(semantic_valid), manifests=len(manifests), history_coverage_complete=False)
