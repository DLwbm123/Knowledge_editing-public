"""Bind existing full-supported Base completions to each fixed protection TRAIN request."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def identities(base_private, current_private):
    base_private, current_private = Path(base_private), Path(current_private)
    base_labels = {(r['task_id'], r['query_id']): r
        for r in read(base_private / 'SCORED_BINDINGS_PRIVATE.json') if r['arm'] == 'BASE'}
    current = {(r['task_id'], r['query_id']): r
        for r in read(current_private / 'SCORED_BINDINGS_PRIVATE.json')
        if r['kind'] in ('protect', 'protect_replay')}
    assert len(base_labels) == 56 and len(current) == 84
    rows = []
    for task in read(current_private / 'completed/TASKS.json')['tasks']:
        for request in task['replay_protect']:
            tid, qid = task['task_id'], request['query_id']
            base = read(base_private / 'completed' / tid / 'BASE' / (qid + '.json'))
            actual = read(current_private / 'completed' / tid / 'REFERENCE_PRIORITY_COMPLETION/outputs' / (qid + '.json'))
            for field in ('question', 'reference', 'image_path'):
                assert base['binding'][field] == actual['binding'][field] == request[field]
            assert base['binding']['image_sha256'] == actual['binding']['image_sha256']
            assert base['EOS'] and not base['at_cap'] and base['raw_token_ids']
            label = base_labels.get((tid, qid))
            if label is None:
                assert base['raw_answer'] == actual['raw_answer']
                assert base['raw_token_ids'] == actual['raw_token_ids']
                label = current[(tid, qid)]
            rows.append(dict(task_id=tid, query_id=qid, question=request['question'],
                reference=request['reference'], Base_answer=base['raw_answer'], Base_token_ids=base['raw_token_ids'],
                binding=base['binding'], base_core=label['core_correct'], base_full=label['full_supported'],
                v126_core=current[(tid, qid)]['core_correct'], v126_full=current[(tid, qid)]['full_supported'],
                v126_answer=actual['raw_answer'], v126_token_ids=actual['raw_token_ids']))
    assert len(rows) == len({(r['task_id'], r['query_id']) for r in rows}) == 84
    return rows


def tokens(rows, adapter, tokenize_image, build_labels, image_index):
    output = []
    for row in rows:
        tokenizer = adapter.tokenizer
        prefix = tokenize_image(adapter._prompt(row['question'], None), tokenizer,
            image_index, return_tensors='pt').unsqueeze(0)
        assert prefix[0].tolist() == row['binding']['prompt_ids']
        full = tokenize_image(adapter._prompt(row['question'], row['reference']), tokenizer,
            image_index, return_tensors='pt').unsqueeze(0)
        _, labels = build_labels(full, prefix, full.new_ones(full.shape), image_token_index=image_index)
        short, base = list(labels), row['Base_token_ids']
        assert short[-1] == base[-1] == tokenizer.eos_token_id
        assert tokenizer.eos_token_id not in base[:-1]
        chosen = base if row['base_full'] else short
        first = next((i for i, (a, b) in enumerate(zip(short, base)) if a != b), min(len(short), len(base)))
        output.append(dict(task_id=row['task_id'], query_id=row['query_id'],
            short_reference_token_ids=short, Base_token_ids=base, selected_token_ids=chosen,
            selection='FROZEN_FULL_SUPPORTED_BASE' if row['base_full'] else 'ORIGINAL_SOURCE_REFERENCE',
            short_count=len(short), Base_count=len(base), selected_count=len(chosen),
            first_divergence_zero_based=None if short == base else first,
            short_EOS_index=len(short) - 1, Base_EOS_index=len(base) - 1,
            actual_sequence_changes=short != chosen, prompt_token_parity=True))
    assert len(output) == 84
    return output


def summarize(rows, token_rows):
    assert {(r['task_id'], r['query_id']) for r in rows} == {(r['task_id'], r['query_id']) for r in token_rows}
    result = dict(identities=84, model_forwards=0, generations=0, updates=0,
        Base_full_selected=sum(r['base_full'] for r in rows), fallback_reference=sum(not r['base_full'] for r in rows),
        actual_sequence_changes=sum(r['actual_sequence_changes'] for r in token_rows),
        original_supervised_tokens_per_pass=sum(r['short_count'] for r in token_rows),
        candidate_supervised_tokens_per_pass=sum(r['selected_count'] for r in token_rows))
    result['transitions'] = {scope: {name: sum((r['base_' + scope], r['v126_' + scope]) == flags for r in rows)
        for name, flags in [('kept_correct', (True, True)), ('new_damage', (True, False)),
                            ('repaired', (False, True)), ('remains_incorrect', (False, False))]}
        for scope in ('core', 'full')}
    result['frozen_rule'] = 'Own exact Base tokens iff own previously frozen full support; otherwise original source tokens'
    return result
