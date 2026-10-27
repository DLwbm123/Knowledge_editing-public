"""Recover known cached-decode cost omissions without changing formal records."""
import json
import os
from pathlib import Path
import time

LEGACY_PRIMARY_SOURCES = {
    'b99a4d6674eaee1991a2e39ee5c40aad3ed23753',
    'a60e51e4af582d094015e65a9b941e92547f475b',
    'ee960b276419758e5eeaca3ba847c119b8abf69d',
    '28689d1d500c931fde15e3a70d4368fc86164cbb',
    'c265e39b90d62205bd17452e108e01a0a8df3e12',
}


def main():
    out = Path(os.environ['V131_ROOT'])/'MEMIT_MM'
    progress = json.loads((out/'PROGRESS.json').read_text())
    assert 'counts' in progress and 'LM_input_positions' in progress['counts']
    receipt = out/'COUNTER_FIX_DEPLOYED.json'
    fixed = set(json.loads(receipt.read_text()).get('fixed_counter_execution_identities',[])) if receipt.exists() else set()
    by_source, unknown, fixed_outputs = {}, {}, 0
    for path in (out/'outputs').glob('*.json'):
        record = json.loads(path.read_text())
        if 'parallel_gpu' in record or record['time'] > progress['time']:
            continue
        source = record['execution']
        if source in fixed:
            fixed_outputs += 1
            continue
        if source not in LEGACY_PRIMARY_SOURCES:
            unknown[source] = unknown.get(source,0)+1
            continue
        row = by_source.setdefault(source,dict(primary_outputs=0,omitted_decode_positions=0))
        row['primary_outputs'] += 1
        row['omitted_decode_positions'] += max(0,len(record['R0']['raw_token_ids'])-1)
    correction = sum(row['omitted_decode_positions'] for row in by_source.values())
    raw = progress['counts']['LM_input_positions']
    result = dict(time=time.time(),status='SNAPSHOT_COST_CORRECTED' if not unknown
        else 'UNCLASSIFIED_PRIMARY_SOURCE_REQUIRES_COUNTER_AUDIT',
        primary_progress_time=progress['time'],raw_LM_input_positions=raw,
        known_omitted_completed_primary_decode_positions=correction,
        corrected_snapshot_LM_input_positions=raw+correction,by_legacy_execution=by_source,
        unclassified_primary_sources=unknown,fixed_counter_primary_outputs=fixed_outputs,formal_records_modified=False,
        in_flight_positions_not_in_progress_snapshot_excluded=True,
        failed_without_output_not_reconstructed=True,
        helper_counters_merge_pending=any(not p.with_name('END.json').exists()
             for p in (out/'parallel').glob('prefix*/*/START.json')))
    path = out/'LM_POSITION_COST_CORRECTION.json'
    tmp = path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    return result


if __name__ == '__main__':
    print(json.dumps(main()))
