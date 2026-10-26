"""Independent singles, then a separate truly cumulative parameter trajectory.

Single weights are restored to Base after their consumers finish. Sequential
editing recomputes the latent target and key under the current edited model;
neither independent single deltas nor independent latent targets are reused.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V131_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
METHOD = os.environ['V131_METHOD']
OUT = ROOT / METHOD
lock = json.loads(Path(os.environ.get('V131_PROTOCOL_FILE', ROOT / 'PROTOCOL_LOCK.json')).read_text())
START = time.time()
counts = dict(forwards=0, backwards=0, updates=0, edits=0, generation_attempts=0,
              formal_consumers=0, inserted=0, vision_forwards=0, LM_input_positions=0)
parallel_reservation = dict(forwards=0, generation_attempts=0, formal_consumers=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def budget():
    assert time.time() < lock['hard_end'] - 20 and not (OUT / 'STOP.json').exists()
    health = read(OUT / 'GUARD_HEALTH.json')
    assert time.time() - health['time'] < 60
    for key, maximum in lock['maximum_counts'][METHOD].items():
        actual = counts[key] + parallel_reservation.get(key, 0)
        assert actual <= maximum, (key, actual, maximum)


def main():
    assert METHOD == 'MEMIT_MM', 'Completed methods cannot enter this worker'
    with (OUT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original wall')))
    signal.alarm(max(1, int(lock['hard_end'] - time.time() - 20)))
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'), str(PARENT / 'private/source'),
                   str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import trajectory
    import contract
    import algebra
    import parameter_edit as editor
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    from m3bench_repro.editors.llava_runtime import seed_everything
    resume = None
    if os.environ.get('V131_MEMIT_RESUME_FILE'):
        resume = torch.load(os.environ['V131_MEMIT_RESUME_FILE'], map_location='cpu', weights_only=True)
        assert resume['mode'] == 'sequential' and 1 <= resume['owner'] <= 146
        assert resume['source'] == lock['resume_source_execution_identity']
        assert read(OUT/'SINGLE_COMPLETE.json')['consumers'] == 2956
        prior_counts = lock['prior_counts'][METHOD]
        for key in ('updates', 'backwards', 'edits', 'inserted'):
            assert prior_counts[key] == resume['counts'][key], ('Resume counter binding', key)
        assert prior_counts['generation_attempts'] == prior_counts['formal_consumers']
        assert prior_counts['formal_consumers'] >= resume['counts']['formal_consumers']
        assert all(prior_counts[key] >= resume['counts'][key] for key in resume['counts'])
        counts.update(prior_counts)
    r.c.budget = budget
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    write(OUT / 'ALGEBRA_CHECK.json', algebra.selfcheck())
    stats_admission = read(ROOT / 'STATISTICS/READY.json')
    assert stats_admission['counts']['documents'] == 100000
    packet = read(PARENT / 'private/M13_SOURCE_PACKET.json')
    assert contract.source_admission(packet)['status'] == 'PASS'
    tasks = packet['original_tasks']
    plan = [v for v in read(ROOT / 'EVAL_PLAN.json') if v['method'] == METHOD]
    assert len(plan) == 7494 and len(tasks) == 146
    if resume is not None:
        before = {r.c.digest(contract.consumer_id(row))+'.json' for row in plan
            if row['mode'] == 'single' or row['prefix'] < resume['owner']}
        through = before | {r.c.digest(contract.consumer_id(row))+'.json' for row in plan
            if row['mode'] != 'single' and row['prefix'] == resume['owner']}
        actual = {path.name for path in (OUT/'outputs').glob('*.json')}
        assert before <= actual <= through and len(actual) == counts['formal_consumers']
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    if resume is not None:
        assert not runtime.generation_config.get('do_sample', False), 'Resume requires frozen greedy generation'
    layers = lock['parameter_edit_layers']
    assert layers == [4, 5, 6, 7, 8]
    weights = {layer: runtime.get_module(f'model.layers.{layer}.mlp.down_proj').weight for layer in layers}
    base = {layer: value.detach().cpu().clone() for layer, value in weights.items()}
    statistics = {}
    for layer in layers:
        budget()
        covariance = torch.load(ROOT / 'STATISTICS' / f'covariance_{layer}.pt', map_location=runtime.device, weights_only=True).double()
        write(OUT / 'PROGRESS.json', dict(phase='COVARIANCE_INVERSE', layer=layer, time=time.time()))
        inverse = torch.linalg.inv(lock['MEMIT_MM']['mom2_update_weight'] * covariance)
        assert torch.isfinite(inverse).all()
        statistics[layer] = inverse.cpu()
        del covariance, inverse
    torch.cuda.empty_cache()
    hp = lock[METHOD]
    def count(_module, args, kwargs):
        budget()
        counts['forwards'] += 1
        x = kwargs.get('inputs_embeds', kwargs.get('input_ids'))
        if x is None and args:
            x = args[0]
        if isinstance(x, torch.Tensor):
            counts['LM_input_positions'] += int(x.shape[0] * x.shape[1])
    fh = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards']+1))

    def prepare(task, rows, training=False):
        # Converting the WHOLE edited model to FP16 would round away physical
        # writes. Only the untouched multimodal input modules enter FP16.
        modules = [runtime.model.model.embed_tokens, runtime.model.model.mm_projector,
                   runtime.model.get_vision_tower()]
        for module in modules:
            module.half()
        try:
            return r.s.prepare(runtime, bindings, task, rows, training)
        finally:
            for module in modules:
                module.float()
            assert all(value.dtype == torch.float32 for value in weights.values())

    def restore_base():
        with torch.no_grad():
            for layer, value in weights.items():
                value.copy_(base[layer].to(runtime.device))
        assert all(torch.equal(value.detach().cpu(), base[layer]) for layer, value in weights.items())

    # Admission checks the actual prefill path with a deliberately non-FP16
    # representable write, then restores Base before any scientific edit.
    value = weights[layers[0]]
    original = float(value[0, 0])
    with torch.no_grad():
        value[0, 0] += 1e-5
    sentinel = value[0, 0].clone()
    rows0 = trajectory.source_rows(packet, 1)
    prepare(tasks[0], rows0[:1], True)
    assert torch.equal(value[0, 0], sentinel), 'Prefill rounded a physical edit'
    restore_base()
    write(OUT / 'PREFILL_ADMISSION.json', dict(
        FP16_input_only=True, physical_FP32_write_preserved=True, original_value=original,
        Base_restored=True, no_whole_model_half_after_edit=True, time=time.time()))

    def inputs(task):
        rows = trajectory.source_rows(packet, task['order'])
        items = prepare(task, rows, True)
        batches = [r.old.batch_on(runtime, item[1]) for item in items]
        kl_row = dict(rows[0], question='Describe the image.', query_id=f"V131_KL_{task['order']}")
        item = prepare(task, [kl_row])[0]
        expanded = item[3]
        kwargs = dict(inputs_embeds=expanded[4].to(runtime.device, dtype=torch.float32),
            attention_mask=expanded[2].to(runtime.device) if expanded[2] is not None else None,
            position_ids=expanded[1].to(runtime.device) if expanded[1] is not None else None,
            use_cache=False, return_dict=True)
        return batches, (kwargs, kwargs['inputs_embeds'].shape[1]-1)

    # One actual image-conditioned key per production layer verifies the
    # Woodbury result against the original matrix equation, without another LU.
    batches0, kl0 = inputs(tasks[0])
    keys0, _ = editor.capture(runtime, batches0, layers, layers[-1])
    matrix_checks = []
    from algebra import memit_key
    for layer in layers:
        covariance = torch.load(ROOT/'STATISTICS'/f'covariance_{layer}.pt',map_location='cpu',weights_only=True).double()
        key = keys0[layer]
        adjusted = memit_key(statistics[layer],key)
        error = (hp['mom2_update_weight']*(covariance@adjusted)+key*torch.dot(key,adjusted)-key).norm()/key.norm()
        assert torch.isfinite(error) and float(error) < 1e-5, ('Native MEMIT solve residual',layer,float(error))
        matrix_checks.append(dict(layer=layer,relative_equation_residual=float(error)))
        del covariance
    write(OUT/'PRODUCTION_MATRIX_ADMISSION.json',dict(status='PASS',rows=matrix_checks,
        image_conditioned_expanded_prompt_keys=True,extra_regularization_or_tuning=False,time=time.time()))
    del batches0,kl0,keys0

    def save_reload(mode, owner):
        path = OUT / (mode + '_resume.pt')
        saved = dict(weights={layer: value.detach().cpu().clone() for layer, value in weights.items()},
            mode=mode, owner=owner, counts=dict(counts), source=lock['execution_identity'])
        torch.save(saved, path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        assert loaded['mode'] == mode and loaded['owner'] == owner
        assert all(torch.equal(value, loaded['weights'][layer]) for layer, value in saved['weights'].items())
        # Load the state used by the next consumer, not merely compare a file.
        with torch.no_grad():
            for layer, value in weights.items():
                value.copy_(loaded['weights'][layer].to(runtime.device))
        return path

    def evaluate_serial(rows, inserted):
        for begin in range(0, len(rows), 24):
            pending = []
            for row in rows[begin:begin+24]:
                task = tasks[row['owner']-1]
                query = dict(row['query'], audit_role=row['query'].get('audit_role', 'EVAL'))
                path = OUT/'outputs'/(r.c.digest(contract.consumer_id(row))+'.json')
                if resume is not None and path.exists():
                    saved = read(path)
                    assert saved['consumer_id'] == contract.consumer_id(row)
                    assert all(saved['binding']['input'][key] == query[key]
                        for key in ('query_id', 'question', 'reference', 'image_sha256'))
                    continue
                item = prepare(task, [query])[0]
                pending.append((row, query, item))
            for row, query, item in pending:
                budget()
                path = OUT / 'outputs' / (r.c.digest(contract.consumer_id(row)) + '.json')
                assert not path.exists(), 'No implicit inference retry'
                counts['generation_attempts'] += 1
                answer, trace = r.s.fit.short.generate(runtime, None, item[2], item[3])
                write(path, dict(method=METHOD, mode=row['mode'], prefix=row['prefix'], owner=row['owner'],
                    consumer_id=contract.consumer_id(row), binding=dict(input=query, judge_input=item[4]),
                    R0=dict(raw_answer=answer.decoded_text, raw_token_ids=list(answer.raw_token_ids)),
                    at_cap=len(answer.raw_token_ids) >= 1024, generation_trace=trace,
                    execution=lock['execution_identity'], cumulative_parameter_state=row['mode'] != 'single',
                    actual_inserted_prefix=inserted, no_external_router=True, time=time.time()))
                counts['formal_consumers'] += 1
                write(OUT / 'PROGRESS.json', dict(phase='GENERATE', last_consumer=contract.consumer_id(row),
                    counts=counts, time=time.time()))

    def reserve(forwards=0, generation_attempts=0, formal_consumers=0):
        parallel_reservation.update(forwards=forwards, generation_attempts=generation_attempts,
                                    formal_consumers=formal_consumers)
        budget()

    if os.environ.get('V131_PARALLEL_DEVICES'):
        from parallel_inference import ParallelEvaluator
        parallel = ParallelEvaluator(runtime, r, contract, OUT, ROOT, lock, counts, tasks,
                                     evaluate_serial, budget, reserve)
        evaluate = parallel.evaluate
    else:
        evaluate = evaluate_serial

    # First all independent singles, always restoring exactly the original Base.
    for task in ([] if resume is not None else tasks):
        owner = task['order']
        seed_everything(task['seed'])
        restore_base()
        batches, kl_batch = inputs(task)
        audit = editor.edit(runtime, batches, kl_batch, layers, hp, statistics, counts, budget)
        path = save_reload('single', owner)
        write(OUT / 'training' / f'single_{owner}.json', dict(audit, save_load_exact=True, time=time.time()))
        evaluate([row for row in plan if row['mode'] == 'single' and row['owner'] == owner], 1)
        # Registered single consumers finished. These states cannot substitute
        # for editing under the sequential trajectory and have no later user.
        size = path.stat().st_size
        assert path.resolve().is_relative_to(OUT.resolve()) and not path.is_symlink()
        path.unlink()
        write(OUT / 'retirement' / f'single_{owner}.json', dict(files=1, bytes=size,
            last_consumer='single generation plus save-load', retained_copies=0, time=time.time()))
        del batches, kl_batch
    if resume is None:
        write(OUT / 'SINGLE_COMPLETE.json', dict(consumers=2956, counts=counts, time=time.time()))
        restore_base()
        remaining_tasks = tasks
    else:
        path = OUT/'sequential_resume.pt'
        assert path.resolve() == Path(os.environ['V131_MEMIT_RESUME_FILE']).resolve() and not path.is_symlink()
        assert set(resume['weights']) == set(layers)
        with torch.no_grad():
            for layer, value in weights.items():
                saved = resume['weights'][layer]
                assert saved.dtype == value.dtype and saved.shape == value.shape and torch.isfinite(saved).all()
                value.copy_(saved.to(runtime.device))
                assert torch.equal(value.detach().cpu(), saved)
        assert counts['inserted'] == resume['owner'] and counts['edits'] == 146+resume['owner']
        write(OUT/'RESUME_ADMISSION.json', dict(status='PASS', owner=resume['owner'],
            physical_cumulative_weights_restored_exactly=True, repeated_single_edits=0,
            repeated_sequential_edits=0, prior_counts=lock['prior_counts'][METHOD], time=time.time()))
        seed_everything(tasks[resume['owner']-1]['seed'])
        evaluate([row for row in plan if row['mode'] != 'single' and row['prefix'] == resume['owner']], resume['owner'])
        remaining_tasks = tasks[resume['owner']:]
    for task in remaining_tasks:
        owner = task['order']
        seed_everything(task['seed'])
        batches, kl_batch = inputs(task)
        audit = editor.edit(runtime, batches, kl_batch, layers, hp, statistics, counts, budget)
        counts['inserted'] += 1
        assert counts['inserted'] == owner
        path = save_reload('sequential', owner)
        write(OUT / 'training' / f'sequential_{owner}.json', dict(audit, actual_inserted=owner,
            latent_recomputed_under_current_weights=True, save_load_exact=True, time=time.time()))
        evaluate([row for row in plan if row['mode'] != 'single' and row['prefix'] == owner], owner)
        write(OUT / 'PROGRESS.json', dict(phase='INSERTION_COMPLETE', owner=owner, counts=counts, time=time.time()))
        del batches, kl_batch
    assert counts['formal_consumers'] == counts['generation_attempts'] == 7494
    assert counts['inserted'] == 146 and counts['edits'] == 292
    write(OUT / 'GENERATION_COMPLETE.json', dict(status='ALL_CONSUMERS_DURABLE_JUDGE_PENDING',
        counts=counts, time=time.time()))
    assert path == OUT / 'sequential_resume.pt' and path.is_file() and not path.is_symlink()
    size = path.stat().st_size
    write(OUT / 'SEQUENTIAL_RETIREMENT_PLAN.json', dict(path=str(path), bytes=size,
        last_consumers='all146 insertions and4actualprefixes; full bindings/tokens durable'))
    path.unlink()
    write(OUT / 'SEQUENTIAL_RETIREMENT_RECEIPT.json', dict(files=1, bytes=size,
        retained_copies=0, reconstruction='Rerun sequential editing from Base', time=time.time()))
    write(OUT / 'COMPLETE.json', dict(status='GPU_COMPLETE_JUDGE_PENDING', counts=counts, time=time.time()))
    fh.remove()
    vh.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(OUT / 'FAILURE.json', dict(error=traceback.format_exc(), counts=counts, time=time.time()))
        raise
    finally:
        write(OUT / 'WORKER_EXIT.json', dict(code=code, seconds=time.time()-START, counts=counts, time=time.time()))
