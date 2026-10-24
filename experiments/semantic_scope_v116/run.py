"""Frozen in-model question-relation classification, without medical answer inputs."""
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V116_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
START = time.time()
counts = dict(forwards=0, comparisons=0, identity_matches=0, vision_forwards=0,
              generations=0, backwards=0, updates=0)
INSTRUCTION = (
    'Compare two questions about the SAME fixed medical image. Decide whether they request exactly '
    'the SAME fact. Do not answer either medical question and do not infer its medical answer. '
    'Output 1 for the same requested fact, or 0 for a different requested fact. '
    'Same means the same subject, attribute or relation, and spatial qualifier. '
    'Paraphrases, synonyms, word order and imperative wording do not change a fact. '
    'Imaging modality, body region, organ identity, largest size, organ position, number, scan plane '
    'and disease findings are distinct requested attributes. Asking which organ is largest is '
    'different from asking which organ is at a position. A broader, narrower or different attribute '
    'is not the same fact. Return only the digit 1 or 0.'
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digit_prefix(tokenizer, text):
    special = not text.startswith(tokenizer.bos_token or '<s>')
    a = tokenizer.encode(text + ' 0', add_special_tokens=special)
    b = tokenizer.encode(text + ' 1', add_special_tokens=special)
    length = 0
    while length < min(len(a), len(b)) and a[length] == b[length]:
        length += 1
    assert 0 < length < min(len(a), len(b)) and a[0] == b[0] == tokenizer.bos_token_id
    assert a[length + 1:] == b[length + 1:]
    assert tokenizer.decode([a[length]]).strip() == '0' and tokenizer.decode([b[length]]).strip() == '1'
    return a[:length], a[length], b[length]


lock = read(ROOT / 'PROTOCOL_LOCK.json')


def budget():
    assert time.time() < lock['hard_end'] - 15 and not (ROOT / 'STOP.json').exists()
    assert time.time() - read(ROOT / 'GUARD_HEALTH.json')['time'] < 30
    for key, maximum in lock['maximum_counts'].items():
        assert counts[key] <= maximum, (key, counts[key], maximum)


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen wall limit')))
    signal.alarm(max(1, int(lock['hard_end'] - time.time() - 15)))
    sys.path[:0] = [str(PARENT / 'private/tools'), str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    from llava.conversation import conv_templates
    r.c.budget = budget
    random.seed(lock['seed'])
    torch.manual_seed(lock['seed'])
    torch.cuda.manual_seed_all(lock['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tasks = read(ROOT / 'SCOPE_INPUTS.json')['tasks']
    assert len(tasks) == 14
    assert all(set(t) == {'task_id', 'role', 'source_question', 'queries'} for t in tasks)
    assert all(set(q) == {'query_id', 'question', 'kind'} for t in tasks for q in t['queries'])
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    def count(*_):
        budget()
        counts['forwards'] += 1
    fh = runtime.model.register_forward_pre_hook(count)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards'] + 1))
    cache = {}

    def compare(task, row):
        identity = (task['task_id'], row['query_id'])
        if identity in cache:
            return cache[identity]
        budget()
        counts['comparisons'] += 1
        if task['source_question'] == row['question']:
            counts['identity_matches'] += 1
            value = dict(ON=True, score=None, identity_by_definition=True)
        else:
            message = INSTRUCTION + '\nQuestion A: ' + json.dumps(task['source_question'])
            message += '\nQuestion B: ' + json.dumps(row['question']) + '\nDecision:'
            conversation = conv_templates['mistral_instruct'].copy()
            conversation.append_message(conversation.roles[0], message)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            prefix, zero, one = digit_prefix(runtime.adapter.tokenizer, prompt)
            ids = torch.tensor([prefix], dtype=torch.long, device=runtime.device)
            with torch.no_grad():
                output = runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = output.logits[0, -1].detach().double()
            assert torch.isfinite(logits).all()
            score = float(logits[one] - logits[zero])
            probability = logits.softmax(-1)
            value = dict(ON=score > 0, score=score, identity_by_definition=False,
                logit_zero=float(logits[zero]), logit_one=float(logits[one]),
                digit_probability_mass=float(probability[zero] + probability[one]),
                prompt=prompt, prefix_token_ids=prefix, zero_token_id=zero, one_token_id=one)
            del output, logits, probability, ids
        cache[identity] = value
        write(ROOT / 'scores' / task['task_id'] / (row['query_id'] + '.json'), value)
        write(ROOT / 'PROGRESS.json', dict(comparisons=counts['comparisons'], counts=dict(counts), time=time.time()))
        return value

    calibration = []
    for task in tasks:
        if task['role'] != 'DEV_CANDIDATE':
            continue
        for row in task['queries']:
            if row['kind'] == 'heldout':
                continue
            value = compare(task, row)
            expected = row['kind'] != 'protect'
            correct = value['ON'] == expected and (value['identity_by_definition'] or value['score'] != 0)
            calibration.append(dict(task_id=task['task_id'], query_id=row['query_id'],
                expected=expected, predicted=value['ON'], correct=correct,
                identity_by_definition=value['identity_by_definition']))
    assert len(calibration) == 58 and counts['forwards'] == 50 and counts['identity_matches'] == 8
    errors = sum(not r['correct'] for r in calibration)
    write(ROOT / 'CALIBRATION_FROZEN.json', dict(errors=errors, rows=58, comparisons=calibration,
        instruction=INSTRUCTION, optimized_parameters=0, other_six_used=False,
        medical_answer_fields_absent=True, time=time.time()))
    if errors:
        write(ROOT / 'COMPLETE.json', dict(status='SEMANTIC_SCOPE_CALIBRATION_FAILED',
            calibration_errors=errors, cache_replay=False, counts=dict(counts), time=time.time()))
        return
    routes = {}
    source_rejected = []
    for task in tasks:
        routes[task['task_id']] = {}
        for row in task['queries']:
            value = compare(task, row)
            routes[task['task_id']][row['query_id']] = dict(ON=value['ON'], score=value['score'])
            if row['kind'] in ('native', 'FIT') and not value['ON']:
                source_rejected.append((task['task_id'], row['query_id']))
    assert counts['comparisons'] == 126 and counts['identity_matches'] == 14 and counts['forwards'] == 112
    assert counts['vision_forwards'] == counts['generations'] == counts['backwards'] == counts['updates'] == 0
    write(ROOT / 'ALL_ROUTING_FROZEN.json', dict(routes=routes, source_rejected=source_rejected,
        classifier_inputs='questions_only_without_medical_reference', time=time.time()))
    write(ROOT / 'COMPLETE.json', dict(status='SOURCE_FIT_REJECTED_NOT_REPLAYABLE' if source_rejected else 'CACHE_REPLAY_READY',
        cache_replay=not source_rejected, counts=dict(counts), time=time.time()))
    fh.remove()
    vh.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(ROOT / 'FAILURE.json', dict(error=traceback.format_exc(), counts=counts, time=time.time()))
        raise
    finally:
        write(ROOT / 'WORKER_EXIT.json', dict(code=code, counts=counts, seconds=time.time() - START, time=time.time()))
