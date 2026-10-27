"""One complete Base answer for each fixed, source-supported A/B question."""
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V106_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
START = time.time()
counts = dict(forwards=0, generations=0, vision_forwards=0, backwards=0, updates=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


lock = read(ROOT / 'PROTOCOL_LOCK.json')


def budget():
    assert time.time() < lock['hard_end'] - 15
    assert time.time() - START < lock['new_gpu_cap_seconds'] - 15
    assert not (ROOT / 'STOP.json').exists()
    assert time.time() - read(ROOT / 'GUARD_HEALTH.json')['time'] < 30
    assert counts['forwards'] < 40000 and counts['generations'] <= 14


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original budget')))
    signal.alarm(max(1, int(min(lock['new_gpu_cap_seconds'] - 10, lock['hard_end'] - time.time() - 15))))
    sys.path[:0] = [str(PARENT / 'private/tools'), str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    r.c.budget = budget
    random.seed(lock['seed'])
    torch.manual_seed(lock['seed'])
    torch.cuda.manual_seed_all(lock['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    template = read(PARENT / 'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    rows = read(ROOT / 'SOURCE_ROWS.json')
    assert len(rows) == 14 and len({x['source_group'] for x in rows}) == 7
    assert len({x['query_id'] for x in rows}) == 14
    assert all(x['reference'] == 'Yes' for x in rows)

    def count(*_):
        budget()
        counts['forwards'] += 1
    fh = runtime.model.register_forward_pre_hook(count)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards'] + 1))
    try:
        for row in rows:
            budget()
            dest = 'outputs/' + row['query_id'] + '.json'
            assert not (ROOT / dest).exists()
            runtime.model.half()
            try:
                item = r.s.prepare(runtime, bindings, template, [row], False)[0]
            finally:
                r.old.fp32(runtime)
            assert item[0]['query_id'] == row['query_id']
            write('attempts/' + row['query_id'] + '.json', dict(time=time.time(), attempt=1))
            counts['generations'] += 1
            answer, _ = r.s.fit.short.generate(runtime, None, item[2], item[3])
            tokens = list(answer.raw_token_ids)
            write(dest, dict(binding=item[4], raw_answer=answer.decoded_text, raw_token_ids=tokens,
                             EOS=bool(tokens) and tokens[-1] == runtime.adapter.tokenizer.eos_token_id,
                             at_cap=len(tokens) >= 1024))
            write('PROGRESS.json', dict(counts=counts, time=time.time()))
            del item
        assert counts['generations'] == counts['vision_forwards'] == 14
        assert counts['updates'] == counts['backwards'] == 0
        write('COMPLETE.json', dict(status='BASE_REVIEW_PENDING', counts=counts, time=time.time()))
    finally:
        fh.remove()
        vh.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('FAILURE.json', dict(error=traceback.format_exc(), counts=counts, time=time.time()))
        raise
    finally:
        write('WORKER_EXIT.json', dict(code=code, counts=counts, seconds=time.time() - START, time=time.time()))
