"""Disjoint batch-one inference at an immutable cumulative-weight prefix.

Editing remains serial. Manual cached forwards replay one newly produced
anchor's exact inputs/tokens for device admission; they are counted separately
from formal generation and never enter the Judge queue.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def shards(rows, count):
    assert count > 0
    result = [rows[index::count] for index in range(count)]
    assert sum(map(len, result)) == len(rows)
    return result


def helper_seconds(folder, now, strict=True, since=0.):
    """Sum helper process lifetimes, including loading, parity and idle time."""
    total, alive = 0., 0
    if not Path(folder).exists():
        return total, alive
    for start_path in Path(folder).glob('prefix*/*/START.json'):
        start = read(start_path)
        if start['time'] < since:
            continue
        end_path = start_path.with_name('END.json')
        if end_path.exists():
            ended = read(end_path)['time']
        else:
            proc = Path('/proc') / str(start['pid'])
            matched = proc.exists() and proc.joinpath('stat').read_text().split()[21] == start['start_ticks']
            assert matched or not strict, 'Unclosed helper lifetime'
            ended = now
            alive += int(matched)
        assert ended >= start['time']
        total += ended - start['time']
    return total, alive


class ParallelEvaluator:
    def __init__(self, runtime, modules, contract, out, root, lock, counts,
                 tasks, serial, budget, reserve):
        self.runtime, self.r, self.contract = runtime, modules, contract
        self.out, self.root, self.lock, self.counts = out, root, lock, counts
        self.tasks, self.serial, self.budget, self.reserve = tasks, serial, budget, reserve

    def output(self, row):
        return self.out/'outputs'/(self.r.c.digest(self.contract.consumer_id(row))+'.json')

    def evaluate(self, rows, inserted):
        existing = [row for row in rows if self.output(row).exists()]
        self.serial(existing, inserted)  # Retain the original full binding checks.
        pending = [row for row in rows if not self.output(row).exists()]
        if len(pending) < 48:
            return self.serial(pending, inserted)
        devices = []
        for gpu in json.loads(os.environ['V131_PARALLEL_DEVICES']):
            info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
            uuid, free = info.split(', ')
            if int(free) >= 45*1024:
                devices.append((gpu, uuid))
        if not devices:
            return self.serial(pending, inserted)
        import torch
        phase = self.out/'parallel'/f'prefix{inserted}'
        phase.mkdir(parents=True, exist_ok=False)
        captured = []

        def capture(_module, args, kwargs):
            assert not args, 'The frozen adapter passes named generation inputs'
            values = {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                      for key, value in kwargs.items() if key != 'past_key_values'}
            past = kwargs.get('past_key_values')
            captured.append(dict(kwargs=values, has_past='past_key_values' in kwargs,
                                 past_kind=type(past).__name__ if past is not None else None))
        handle = self.runtime.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            self.serial(pending[:1], inserted)
        finally:
            handle.remove()
        anchor = read(self.output(pending[0]))
        assert len(captured) == len(anchor['R0']['raw_token_ids'])
        anchor_path = phase/'anchor.pt'
        torch.save(dict(calls=captured, tokens=anchor['R0']['raw_token_ids'],
                        row=pending[0], binding=anchor['binding']), anchor_path)
        assignments = shards(pending[1:], len(devices)+1)
        all_ids = [self.contract.consumer_id(row) for group in assignments for row in group]
        assert len(set(all_ids)) == len(all_ids)
        helper_jobs = sum(len(group) for group in assignments[1:])
        # Full per-helper reservations keep the original global operation caps
        # conservative while the parent and helpers execute concurrently.
        self.reserve(forwards=helper_jobs*1024+len(devices)*len(captured),
                     generation_attempts=helper_jobs, formal_consumers=helper_jobs)
        children = []
        neutral = Path(os.environ['NEUTRAL_ENTRY']).parent
        for (gpu, uuid), group in zip(devices, assignments[1:]):
            job = phase/str(gpu)
            job.mkdir()
            write(job/'JOB.json', dict(rows=group, anchor=str(anchor_path), prefix=inserted,
                checkpoint=str(self.out/'sequential_resume.pt'), gpu=gpu, uuid=uuid,
                execution=self.lock['execution_identity']))
            entry = neutral/f'p{gpu}.py'
            entry.write_text('import json,os,runpy\nfrom pathlib import Path\n'
                'p=Path(__file__)\nc=json.loads(p.with_suffix(".json").read_text())\n'
                'os.environ.update(c)\nrunpy.run_path(c["V131_PARALLEL_WORKER_FILE"],run_name="__main__")\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),
                M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid,
                M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=','.join(str(v[0]) for v in devices)+','+os.environ['CUDA_VISIBLE_DEVICES'],
                V131_HELPER_JOB=str(job), V131_PARALLEL_WORKER_FILE=str(Path(__file__).resolve()),
                V131_PARALLEL_WORKER='1')
            write(entry.with_suffix('.json'), env)
            started = time.time()
            log = (job/'worker.log').open('x')
            child = subprocess.Popen([sys.executable, str(entry)], stdout=log, stderr=subprocess.STDOUT)
            log.close()
            write(job/'START.json', dict(time=started, pid=child.pid,
                start_ticks=Path('/proc', str(child.pid), 'stat').read_text().split()[21],
                gpu=gpu, uuid=uuid, parent_pid=os.getpid(), process_group=os.getpgid(child.pid)))
            children.append((child, job))
        write(phase/'ASSIGNMENTS.json', dict(prefix=inserted, parent_rows=len(assignments[0]),
            helper_rows=helper_jobs, GPUs=[gpu for gpu, _ in devices],
            generation_batch_size=1, original_hard_end=self.lock['hard_end']))
        self.serial(assignments[0], inserted)
        while any(child.poll() is None for child, _ in children):
            self.budget()
            time.sleep(1)
        additions = dict.fromkeys(self.counts, 0)
        failures = []
        for child, job in children:
            final = read(job/'COUNTS.json')
            for key in additions:
                additions[key] += final.get(key, 0)
            if child.returncode:
                failures.append(dict(gpu=read(job/'JOB.json')['gpu'], code=child.returncode,
                                     counters=final, admission=(job/'ADMISSION.json').exists()))
        self.reserve()
        for key, value in additions.items():
            self.counts[key] += value
        self.budget()
        write(phase/'MERGED.json', dict(time=time.time(), counts_added=additions, failures=failures,
                                      counters_including_prior_attempts=self.counts))
        assert all(f['counters']['generation_attempts'] == f['counters']['formal_consumers'] == 0
                   and not f['admission'] for f in failures), 'Stop on any formal generation failure; no implicit retry'
        # A card that fails admission has produced no formal consumer. The
        # untouched queries may still use the original primary path.
        missing = [row for row in pending if not self.output(row).exists()]
        self.serial(missing, inserted)
        assert all(self.output(row).exists() for row in rows)
        write(self.out/'PROGRESS.json', dict(phase='PARALLEL_PREFIX_COMPLETE', owner=inserted,
                                            counts=self.counts, time=time.time()))
        anchor_path.unlink()  # All registered hardware checks have consumed it.


def worker():
    import torch
    from types import SimpleNamespace
    root = Path(os.environ['V131_ROOT'])
    parent = Path(os.environ['PARENT_RUN'])
    out = root/'MEMIT_MM'
    jobdir = Path(os.environ['V131_HELPER_JOB'])
    job = read(jobdir/'JOB.json')
    lock = read(os.environ['V131_PROTOCOL_FILE'])
    counts = dict(forwards=0, backwards=0, updates=0, edits=0, generation_attempts=0,
                  formal_consumers=0, inserted=0, vision_forwards=0, LM_input_positions=0)
    started, code = time.time(), 0

    def budget():
        assert time.time() < min(lock['hard_end'], lock['method_original_deadline']['MEMIT_MM'])-20
        assert not (out/'STOP.json').exists()
        assert time.time()-read(out/'GUARD_HEALTH.json')['time'] < 60
        assert counts['forwards'] <= len(job['rows'])*1024+len(read_anchor['calls'])
        assert counts['generation_attempts'] <= len(job['rows'])

    try:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('Guard stop')))
        sys.path[:0] = [str(root/'tools'), str(parent/'private/tools'), str(parent/'private/source'),
                       str(parent/'private/official_llava')]
        import module_run as r
        import contract
        from scripts.medtrace.run_realmodel_core import load_real_runtime
        from m3bench_repro.editors.llava_runtime import seed_everything
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        runtime = load_real_runtime(SimpleNamespace(cpu_gate=parent/'private/cpu_gate'))
        runtime.model.eval()
        r.old.fp32(runtime)
        bindings = read(parent/'private/EVAL_BINDINGS.json')
        assert runtime.generation_config == next(iter(bindings.values()))['generation']
        saved = torch.load(job['checkpoint'], map_location='cpu', weights_only=True)
        assert saved['mode'] == 'sequential' and saved['owner'] == job['prefix']
        assert saved['source'] in (lock['execution_identity'], lock['resume_source_execution_identity'])
        with torch.no_grad():
            for layer, value in saved['weights'].items():
                target = runtime.get_module(f'model.layers.{layer}.mlp.down_proj').weight
                assert target.dtype == value.dtype == torch.float32 and target.shape == value.shape
                target.copy_(value.to(runtime.device))
                assert torch.equal(target.detach().cpu(), value)
        del saved
        read_anchor = torch.load(job['anchor'], map_location='cpu', weights_only=True)
        seed_everything(read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][job['prefix']-1]['seed'])

        def count(_module, args, kwargs):
            budget()
            counts['forwards'] += 1
            x = kwargs.get('inputs_embeds')
            if x is None:
                x = kwargs.get('input_ids')
            if isinstance(x, torch.Tensor):
                counts['LM_input_positions'] += x.shape[0]*x.shape[1]
            if counts['forwards'] % 32 == 0:
                write(jobdir/'PROGRESS.json', dict(time=time.time(), counts=counts))
        runtime.model.register_forward_pre_hook(count, with_kwargs=True)
        runtime.model.get_vision_tower().register_forward_pre_hook(
            lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards']+1))
        input_modules = [runtime.model.model.embed_tokens, runtime.model.model.mm_projector,
                         runtime.model.get_vision_tower()]
        for module in input_modules:
            module.half()
        try:
            anchor_row = read_anchor['row']
            anchor_tasks = read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks']
            anchor_item = r.s.prepare(runtime, bindings, anchor_tasks[anchor_row['owner']-1],
                                      [anchor_row['query']])[0]
        finally:
            for module in input_modules:
                module.float()
        assert anchor_item[2]['input_ids'][0].tolist() == read_anchor['binding']['judge_input']['prompt_ids']
        assert torch.equal(anchor_item[3][4].float(), read_anchor['calls'][0]['kwargs']['inputs_embeds'])
        del anchor_item
        # Replay the exact named arguments seen by the production forward,
        # replacing only the device and the freshly constructed local KV cache.
        past = None
        with torch.inference_mode():
            for index, call in enumerate(read_anchor['calls']):
                kwargs = {key: value.to(runtime.device) if isinstance(value, torch.Tensor) else value
                          for key, value in call['kwargs'].items()}
                if index == 0 and call['past_kind'] == 'DynamicCache':
                    from transformers.cache_utils import DynamicCache
                    past = DynamicCache()
                if call['has_past']:
                    kwargs['past_key_values'] = past
                output = runtime.model(**kwargs)
                assert int(output.logits[0, -1].argmax()) == read_anchor['tokens'][index], ('Token parity', index)
                past = output.past_key_values
        del past, output, kwargs
        write(jobdir/'ADMISSION.json', dict(status='PASS', time=time.time(),
            exact_cached_token_steps=len(read_anchor['tokens']), physical_FP32_weights_exact=True,
            no_formal_generation_or_Judge_retry=True))
        tasks = read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks']
        for row in job['rows']:
            budget()
            query = dict(row['query'], audit_role=row['query'].get('audit_role', 'EVAL'))
            path = out/'outputs'/(r.c.digest(contract.consumer_id(row))+'.json')
            assert not path.exists(), 'Disjoint jobs cannot overwrite a completed consumer'
            modules = [runtime.model.model.embed_tokens, runtime.model.model.mm_projector,
                       runtime.model.get_vision_tower()]
            for module in modules:
                module.half()
            try:
                item = r.s.prepare(runtime, bindings, tasks[row['owner']-1], [query])[0]
            finally:
                for module in modules:
                    module.float()
            counts['generation_attempts'] += 1
            answer, trace = r.s.fit.short.generate(runtime, None, item[2], item[3])
            assert not path.exists()
            write(path, dict(method='MEMIT_MM', mode=row['mode'], prefix=row['prefix'], owner=row['owner'],
                consumer_id=contract.consumer_id(row), binding=dict(input=query, judge_input=item[4]),
                R0=dict(raw_answer=answer.decoded_text, raw_token_ids=list(answer.raw_token_ids)),
                at_cap=len(answer.raw_token_ids)>=1024, generation_trace=trace,
                execution=job['execution'], cumulative_parameter_state=True,
                actual_inserted_prefix=job['prefix'], no_external_router=True,
                parallel_gpu=job['gpu'], time=time.time()))
            counts['formal_consumers'] += 1
            write(jobdir/'PROGRESS.json', dict(time=time.time(), counts=counts))
        assert counts['formal_consumers'] == counts['generation_attempts'] == len(job['rows'])
    except BaseException:
        code = 1
        write(jobdir/'FAILURE.json', dict(time=time.time(), error=traceback.format_exc(), counts=counts))
        raise
    finally:
        write(jobdir/'COUNTS.json', counts)
        write(jobdir/'END.json', dict(time=time.time(), code=code, seconds=time.time()-started))


if __name__ == '__main__':
    assert os.environ.get('V131_PARALLEL_WORKER') == '1'
    worker()
