"""Add sanitized route, mechanism and lifecycle accounting; no model calls."""
import argparse
import collections
import json
from pathlib import Path


def complete(private, output):
    read = lambda path: json.loads(path.read_text())
    result = read(output)
    mapping = read(private / 'BLIND_MAPPING.json')
    labels = {r['id']: r['label'] for r in read(private / 'BLIND_VERDICTS.json')['rows']}
    base = {r['query_id']: labels[r['id']] for r in mapping if r['arm'] == 'BASE'}
    evaluation = {c['row']['query_id'] for c in read(private / 'CANDIDATES_FROZEN.json')['candidates'] if c['split'] == 'eval'}
    correct = {q for q in evaluation if base[q] == 'CORRECT'}
    tasks = read(private / 'DATA_ADMISSION.json')['tasks']
    natural, mechanism, training = {}, {}, {}
    for arm in result['arms']:
        entries = [r for r in mapping if r['arm'] == arm]
        result['panels'][arm]['EOS'] = sum(r['EOS'] for r in entries)
        result['panels'][arm]['at_cap'] = sum(r['at_cap'] for r in entries)
        part = 0 if arm.startswith('OFF') or arm == 'BALANCEDIT' else 1
        training[arm] = {}
        for task in tasks:
            key = task['key']
            receipt = read(private / f'remote/{part}/{key}/{arm}_TRAIN.json')
            assert receipt['status'] == 'FROZEN' and receipt['save_load_exact']
            training[arm][key] = {k: receipt[k] for k in ('steps', 'parameter_count') if k in receipt}
            if arm.endswith('_SEM'):
                curve = receipt['curve']
                mechanism[f'{arm}/{key}'] = dict(steps=len(curve), retention_active_steps=sum(r.get('retention_active', False) for r in curve), initial_score=curve[0]['score']['score'], final_score=receipt['final_semantic']['score'])
            if arm.startswith('BALANCEDIT'):
                values, routes = {}, collections.Counter()
                for row in entries:
                    if row['task'] != key:
                        continue
                    q = row['query_id']
                    route = read(private / f'remote/{part}/{key}/routing/{arm}/{q}.json')
                    choice = route['natural_output_reuses']
                    assert choice in ('BASE', 'FORCED')
                    routes[choice] += 1
                    values[q] = base[q] if choice == 'BASE' else labels[row['id']]
                source = task['source']['query_id']
                natural[f'{arm}/{key}'] = dict(routes=dict(routes), source_correct=values[source] == 'CORRECT', cross_correct=values[source.removesuffix('_native') + '_cross'] == 'CORRECT', rephrase_correct=values[task['rephrase']['query_id']] == 'CORRECT', new_damage=sum(values[q] != 'CORRECT' for q in correct), base_correct_denominator=len(correct), heldout_four_category=dict(collections.Counter(values[q] for q in evaluation)))
    result.update(natural_balancedit=natural, semantic_activation=mechanism, training=training)
    result['gpu_process_hours'] = sum(read(private / f'remote/{i}/EXIT.json')['seconds'] for i in (0, 1)) / 3600
    result['formal_and_fisher_forwards'] = result['counts']['forwards'] - 126
    result['audit_forwards'] = 120
    result['engineering_repair_forwards'] = 6
    retired = read(private / 'remote/RETIREMENT_COMPLETE.json')
    result['retirement'] = dict(files=len(retired['files']), bytes=retired['total_bytes'], historical_assets_touched=retired['historical_assets_touched'], reconstruction_requires_training=True)
    result['base_eval_four_category'] = dict(collections.Counter(base[q] for q in evaluation))
    result['review_consistency_corrections'] = len(read(private / 'REVIEW_CONSISTENCY_AUDIT.json')['changes'])
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    complete(args.private, args.output)
