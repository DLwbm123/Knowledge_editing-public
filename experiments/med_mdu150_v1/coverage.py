"""Check token coverage only for the previously admitted source/FIT entries."""
import collections
import json
import os
from pathlib import Path
import resource
import signal
import time


def main():
    import sys
    sys.path.insert(0, os.environ['OFFICIAL_SOURCE'])
    from llava.conversation import conv_templates
    from transformers import AutoTokenizer

    source = Path(os.environ['SOURCE_ROOT'])
    output = Path(os.environ['OUTPUT_ROOT'])
    admitted = json.loads((source / 'ROLE_ADMISSION.json').read_text())
    entries = json.loads((source / 'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    old = json.loads((source / 'SPAN_LOCK.private.json').read_text())
    prior = {v['order']: v for v in old['records'] if v['status'] == 'WEAK_TOKEN_MATCH'}
    if len(prior) != 0:
        raise ValueError('Prior span coverage changed')
    if admitted['status'] != 'SOURCE_49_INPUTS_CHECKED' or len(entries) != 49:
        raise ValueError('Prior admitted source count changed')
    if [e['task']['order'] for e in entries] != admitted['owners']:
        raise ValueError('Prior owner bindings changed')
    tok = AutoTokenizer.from_pretrained(os.environ['TOKENIZER'], use_fast=True, local_files_only=True)
    if type(tok).__name__ != 'LlamaTokenizerFast' or tok.eos_token_id != 2:
        raise ValueError('Tokenizer contract mismatch')

    def prompt(question, answer):
        conv = conv_templates['mistral_instruct'].copy()
        conv.append_message(conv.roles[0], '<image>\n' + question)
        conv.append_message(conv.roles[1], answer)
        return conv.get_prompt()

    def encode(text):
        parts = text.split('<image>')
        if len(parts) != 2:
            raise ValueError('Image placeholder mismatch')
        left = tok(parts[0], return_offsets_mapping=True)
        right = tok(parts[1], return_offsets_mapping=True)
        if left.input_ids[0] != tok.bos_token_id or right.input_ids[0] != tok.bos_token_id:
            raise ValueError('BOS mismatch')
        shift = len(parts[0]) + len('<image>')
        return (left.input_ids + [-200] + right.input_ids[1:],
                list(left.offset_mapping) + [(0, 0)] +
                [(a + shift, b + shift) if b > a else (0, 0) for a, b in right.offset_mapping[1:]])

    counts = collections.Counter()
    locks = []
    for entry in entries:
        rows = entry['source_rows']
        if len(rows) != 5 or [x['audit_role'] for x in rows] != ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']:
            raise ValueError('Training roles changed')
        answers = [r['reference'].strip() for r in rows]
        if len(set(answers)) != 1:
            raise ValueError('References differ across views')
        answer = answers[0]
        views = []
        common = None
        reason = None
        for row in rows:
            pre, full = prompt(row['question'], None), prompt(row['question'], answer)
            if not full.startswith(pre):
                reason = 'prompt_prefix'; break
            start = full.find(answer, len(pre)) if answer else -1
            if start < 0 or full[len(pre):start].strip():
                reason = 'empty_or_boundary'; break
            pref_ids, _ = encode(pre)
            ids, offsets = encode(full)
            if ids[:len(pref_ids)] != pref_ids:
                reason = 'token_prefix'; break
            end = start + len(answer)
            positions = [i for i, (a, b) in enumerate(offsets) if i >= len(pref_ids) and b > a and a < end and start < b]
            if not positions or positions != list(range(positions[0], positions[-1] + 1)):
                reason = 'empty_or_noncontiguous'; break
            if any(full[a:min(b,start)].strip() or full[max(a,end):b].strip() for i in positions for a,b in [offsets[i]]):
                reason = 'mixed_boundary'; break
            ids = [ids[i] for i in positions]
            if common is None:
                common = ids
            if ids != common:
                reason = 'view_token_mismatch'; break
            views.append({'answer_start_label': positions[0], 'answer_end_label': positions[-1] + 1})
        record = {'order': entry['task']['order'], 'status': 'UNAVAILABLE'}
        if reason:
            counts[reason] += 1
        else:
            n = len(common)
            spans = [(0, n)] if n == 1 else [(0, n // 2), (n // 2, n)]
            previous = prior.get(entry['task']['order'])
            if previous is not None and (previous['answer_tokens'] != n or any(
                    (a['answer_start_label'], a['answer_end_label']) !=
                    (b['answer_start_label'], b['answer_end_label'])
                    for a, b in zip(views, previous['views']))):
                raise ValueError('Prior matched token bindings changed')
            record.update(status='TOKEN_BOUND', answer_tokens=n, spans=spans, views=views)
            counts['bound'] += 1
            counts['singleton'] += n == 1
            counts['prior_matched_bound'] += previous is not None
        locks.append(record)
    result = {'status': 'SOURCE_TOKEN_CHECK_ONLY', 'examined': len(entries),
              'unresolved_not_read': 0, 'counts': dict(counts), 'records': locks}
    path = output / 'TOKEN_LOCK.private.json'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(result, f, separators=(',', ':'))
    return {'examined': len(entries), 'unresolved_not_read': 0, 'counts': dict(counts),
            'models': 0, 'training_updates': 0, 'generations': 0, 'judge': 0}


if __name__ == '__main__':
    root = Path(os.environ['OUTPUT_ROOT'])
    deadline = float(os.environ['DEADLINE'])
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU audit deadline')))
    signal.alarm(max(1, min(600, int(deadline - time.time() - 120))))
    result = main()
    with (root / 'COVERAGE_COMPLETE.json').open('x') as f:
        json.dump(result, f)
