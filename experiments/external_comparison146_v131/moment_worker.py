"""MEMIT native second moments, with fixed Wiki data.

Only layers4..8 are evaluated. Early stopping after layer8 matches the native
layer_stats stop=True behavior. All100000 documents are consumed regardless of
any edit result; 512 tokens/document is a disclosed finite resource adaptation.
"""
import gzip
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V131_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
OUT = ROOT / 'STATISTICS'
lock = json.loads((ROOT / 'PROTOCOL_LOCK.json').read_text())
START = time.time()
counts = dict(forwards=0, documents=0, tokens=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def budget():
    assert time.time() < lock['hard_end'] - 20 and not (OUT / 'STOP.json').exists()
    assert time.time() - read(OUT / 'GUARD_HEALTH.json')['time'] < 60


def documents():
    cached = Path(lock['statistics']['existing_first10000_corpus'])
    with gzip.open(cached, 'rt') as stream:
        for index, line in enumerate(stream):
            assert index < 10000
            yield json.loads(line)['text']
    assert index == 9999
    while not (ROOT / 'CORPUS_DOWNLOAD_COMPLETE.json').exists():
        budget()
        assert not (ROOT / 'CORPUS_DOWNLOAD_PROXY_FAILURE.json').exists(), 'Pinned corpus transfer failed'
        time.sleep(5)
    sys.path.insert(0, str(ROOT / 'deps'))
    import pyarrow.parquet as parquet
    index = 0
    for batch in parquet.ParquetFile(ROOT / 'wiki-shard.parquet').iter_batches(batch_size=256, columns=['text']):
        for text in batch.column(0).to_pylist():
            index += 1
            if index <= 10000:
                continue
            if index > 100000:
                return
            yield text
    assert index >= 100000


def main():
    with (OUT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original wall')))
    signal.alarm(max(1, int(lock['hard_end'] - time.time() - 20)))
    sys.path[:0] = [str(PARENT / 'private/tools'), str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget = budget
    torch.manual_seed(13120261025)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    tokenizer = runtime.adapter.tokenizer
    layers = lock['parameter_edit_layers']
    assert layers == [4, 5, 6, 7, 8]
    widths = {layer: runtime.get_module(f'model.layers.{layer}.mlp.down_proj').in_features for layer in layers}
    assert set(widths.values()) == {14336}
    moments = {layer: torch.zeros(width, width, dtype=torch.float32, device=runtime.device) for layer, width in widths.items()}
    captured = {}
    handles = []
    for layer in layers:
        def collect(_module, args, layer=layer):
            captured[layer] = args[0][0].detach().float()
        handles.append(runtime.get_module(f'model.layers.{layer}.mlp.down_proj').register_forward_pre_hook(collect))
    class FinishedStatistics(Exception):
        pass
    def stop(*_):
        raise FinishedStatistics()
    handles.append(runtime.get_module('model.layers.8').register_forward_hook(stop))
    try:
        for text in documents():
            budget()
            tokens = tokenizer(text, return_tensors='pt', truncation=True,
                max_length=lock['statistics']['max_tokens'], add_special_tokens=True)
            ids = tokens['input_ids'].to(runtime.device)
            assert 0 < ids.shape[1] <= 512
            try:
                with torch.inference_mode():
                    runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
            except FinishedStatistics:
                pass
            else:
                raise AssertionError('Native statistics early stop was bypassed')
            assert set(captured) == set(layers) and len({v.shape[0] for v in captured.values()}) == 1
            with torch.no_grad():
                for layer, values in captured.items():
                    assert torch.isfinite(values).all()
                    moments[layer].add_(values.T @ values)
            counts['forwards'] += 1
            counts['documents'] += 1
            counts['tokens'] += ids.shape[1]
            captured.clear()
            if counts['documents'] % 100 == 0:
                write('PROGRESS.json', dict(phase='COLLECT', counts=counts,
                    elapsed_seconds=time.time()-START, time=time.time()))
            assert counts['documents'] <= 100000
    finally:
        for handle in handles:
            handle.remove()
    assert counts['documents'] == counts['forwards'] == 100000
    del runtime, tokenizer, ids
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    matrices = []
    for layer in layers:
        budget()
        write('PROGRESS.json', dict(phase='SAVE_MOMENT', layer=layer, counts=counts, time=time.time()))
        covariance = moments.pop(layer).div_(counts['tokens'])
        covariance = .5 * (covariance + covariance.T)
        assert torch.isfinite(covariance).all()
        torch.save(covariance.cpu(), OUT / f'covariance_{layer}.pt')
        matrices.append(dict(layer=layer,width=14336,dtype='float32',tokens=counts['tokens']))
        del covariance
        gc.collect()
        torch.cuda.empty_cache()
    write('READY.json', dict(status='COMPLETE', counts=counts, matrices=matrices,
        backbone='Original LLaVA-Med FP16 values promoted to FP32; unchanged weights',
        corpus=lock['statistics'], context_cap_is_resource_adaptation=True, time=time.time()))
    write('COMPLETE.json', dict(status='COMPLETE', counts=counts, time=time.time()))


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('FAILURE.json', dict(error=traceback.format_exc(), counts=counts, time=time.time()))
        raise
    finally:
        write('WORKER_EXIT.json', dict(code=code, seconds=time.time()-START, counts=counts, time=time.time()))
