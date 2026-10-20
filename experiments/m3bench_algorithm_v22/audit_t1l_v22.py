"""Read-only audit of existing T1L answers; no models or verdict changes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

ARMS = ('SOURCE', 'CE', 'KL', 'CE_PROJ', 'RESERVE', 'BALANCEDIT', 'BALANCEDIT_CE')


def audit(root):
    def read(path):
        return json.loads(path.read_text())

    def normalized(text):
        # Only case, whitespace and trailing periods; no semantic equivalence claim.
        return re.sub(r'\s+', ' ', text.strip()).rstrip('.').casefold()

    config = read(root / 'DATA_ADMISSION.json')
    assert len(config['tasks']) == 10
    mapping = read(root / 'BLIND_MAPPING.json')
    verdicts = read(root / 'BLIND_REUSE.json')
    verdicts.update(read(root / 'BLIND_VERDICTS.json'))
    stats, transitions, per_edit = Counter(), Counter(), {}
    source_copy = {arm: 0 for arm in ARMS}
    residuals = {arm: 0 for arm in ARMS[:5]}
    unique = set()
    for task in config['tasks']:
        folder = root / str((task['order'] - 1) % 5) / task['key']
        local = Counter()
        for row in task['eval_rows']:
            if 'T1L' not in row['roles']:
                continue
            qid = row['query_id']
            unique.add(qid)
            paths = [folder / 'outputs' / arm / (qid + '.json') for arm in ARMS]
            values = [read(path) for path in paths]
            base_paths = list((root / 'base').glob('*/' + qid + '.json'))
            assert len(base_paths) == 1
            base = read(base_paths[0])
            labels = [verdicts[mapping[str(path.relative_to(root))]]['label'] for path in paths]
            base_label = verdicts[mapping[str(base_paths[0].relative_to(root))]]['label']
            checks = {
                'consumers': True,
                'all_seven_same_answer': all(v['raw_answer'] == values[0]['raw_answer'] for v in values),
                'all_seven_same_tokens': all(v['raw_token_ids'] == values[0]['raw_token_ids'] for v in values),
                'all_seven_same_label': len(set(labels)) == 1,
                'all_seven_same_binding': all(v['binding'] == values[0]['binding'] for v in values),
                'base_core_binding_matches': all(all(v['binding'][k] == base['binding'][k] for k in ('image_sha256', 'question', 'reference', 'prompt_ids', 'attention_mask', 'runtime', 'generation')) for v in values),
                'same_question_as_source': row['question'] == task['source']['question'],
                'different_image_from_source': row['image_sha256'] != task['source']['image_sha256'],
                'same_reference_as_source': normalized(row['reference']) == normalized(task['source']['reference']),
                'all_seven_generation_attempts_present': all((folder / 'attempts' / arm / (qid + '.json')).is_file() for arm in ARMS),
                'source_tokens_equal_base': values[0]['raw_token_ids'] == base['raw_token_ids'],
            }
            for key, value in checks.items():
                stats[key] += int(value)
                local[key] += int(value)
            transitions[base_label + ' -> ' + labels[0]] += 1
            for arm, value in zip(ARMS, values):
                source_copy[arm] += normalized(value['raw_answer']) == normalized(task['source']['reference'])
                stats['reuse_flags'] += 'reuse_kind' in value
                if arm in residuals:
                    residuals[arm] += any(t.get('active_residual_norm', 0) > 0 for t in value['generation_trace'])
        if local:
            per_edit[task['key']] = dict(local)
    assert stats['consumers'] == 84 and len(unique) == 50
    return dict(status='EXISTING_OUTPUT_AUDIT_COMPLETE', model_calls=0,
                outputs_read=7 * stats['consumers'], unique_queries=len(unique),
                counts=dict(stats), source_target_copy_normalized=source_copy,
                TT_nonzero_generation_residual=residuals, base_to_method_labels=dict(transitions),
                per_edit=per_edit, normalization='case, whitespace, trailing periods only',
                limitation='Bindings and source inspection do not reconstruct retired model states or prove causal visual reliance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
