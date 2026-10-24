"""Finite source-only target generation and frozen four-arm M13P comparison."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V81_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
PART = int(os.environ.get('PARTITION', 0))
PHASE = os.environ['PHASE']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


counts = dict(forwards=0, non_generation_forwards=0, backwards=0,
              operator_backwards=0, updates=0, teacher_generations=0,
              evaluation_generations=0)
generating = False
limits = read(ROOT / 'RESOURCE_LIMITS.json')
allocations = None


def progress(**extra):
    write(ROOT / f'PROGRESS_{PHASE}_{PART}.json', dict(
        time=time.time(), phase=PHASE, partition=PART, counts=counts, **extra))


def budget():
    global allocations
    assert time.time() < limits['hard_end'] - 300, 'Original wall deadline'
    assert not (ROOT / 'STOP').exists(), 'Guard stop'
    health = read(ROOT / 'RESOURCE_GUARD_HEALTH.json')
    assert time.time() - health['time'] < 120, 'Guard stale'
    if allocations is None:
        allocations = {k:v//2 for k,v in limits['counts'].items()}
        owners = {x['order'] for x in read(ROOT / 'TASKS.json')[PART::2]}
        allocations['evaluation_generations'] = 4 * sum(x['owner'] in owners for x in read(ROOT / 'EVAL_PLAN.json')) + 10
    for key, allocated in allocations.items():
        assert counts[key] <= allocated, (key, counts[key], allocated)


def contexts(fact, arm):
    common = ('Training-only source information follows. It applies only to the '
              'given source image and question. Answer the original question '
              'concisely in your own words, cover all supplied answer facts, '
              'retain qualifiers, and add no unsupported facts.\n')
    if arm == 'S1':
        body = fact['prose']
    elif arm == 'S2':
        body = json.dumps({f'item_{i+1}': s for i, s in enumerate(fact['prose_parts'])},
                          ensure_ascii=False)
    elif arm == 'S3':
        body = json.dumps(fact['medical_record'], ensure_ascii=False)
        body += ('\nUse the entity-relation-value bindings to answer only the '
                 'relation requested by the original question; preserve its '
                 'laterality, polarity, uncertainty, and scope when supplied.')
    else:
        raise ValueError(arm)
    return common + body


def setup():
    global torch, r, runtime, bindings, cache_class, training_math
    assert read(ROOT / 'CPU_ADMISSION.json')['status']=='PASS'
    gpu = int(os.environ['GPU'])
    uuid = read(ROOT / 'HARDWARE.json')[str(gpu)]
    os.environ.update(M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),
        M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu), M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid)
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'),
                   str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import training_math
    from prefix_cache import PrefixCache
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    cache_class = PrefixCache
    r.c.budget = budget
    r.COUNTS = counts
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    import subprocess
    observed = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert observed[0] == read(ROOT / 'HARDWARE.json')[str(gpu)]
    assert int(observed[1]) >= 45 * 1024, '45GiB load reserve'
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.run_root = ROOT / 'work' / f'{PHASE}_{PART}'
    runtime.run_root.mkdir(parents=True, exist_ok=True)
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    def count(*_):
        budget()
        counts['forwards'] += 1
        if not generating:
            counts['non_generation_forwards'] += 1
    runtime.model.register_forward_pre_hook(count)
    # Only this process family is audited; unrelated processes are untouched.
    argv = subprocess.check_output(['ps', '-ww', '-p', str(os.getpid()), '-o', 'args='], text=True).strip()
    assert not any(s in argv for s in ('wangbomin', 'Knowledge_editing', 'medtrace', 'medical_condition'))
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True).splitlines()
    own = [s for s in apps if s.split(',')[0].strip() == str(os.getpid())]
    assert own and all(s.split(',')[1].strip() == observed[0] for s in own)
    write(ROOT / f'PROCESS_AUDIT_{PHASE}_{PART}.json', dict(argv=argv, gpu_processes=own, neutral=True))


def prepare(task, rows, training=False):
    runtime.model.half()
    try:
        return r.s.prepare(runtime, bindings, task, rows, training)
    finally:
        r.old.fp32(runtime)


def device_tuple(expanded):
    return tuple(v.to(device=runtime.device, dtype=torch.float32 if v.is_floating_point() else v.dtype)
                 if isinstance(v, torch.Tensor) else v for v in expanded)


def pref(expanded):
    e = device_tuple(expanded)
    kw = dict(inputs_embeds=e[4], attention_mask=e[2], position_ids=e[1],
              use_cache=True, return_dict=True)
    return {k: v for k, v in kw.items() if v is not None}


def teacher(task):
    global generating
    owner = task['order']
    fact = read(ROOT / 'FACTS.json')[str(owner)]
    source = read(ROOT / 'SOURCE_ROWS.json')[str(owner)]
    for view, row in enumerate(source):
        for arm in ('S1', 'S2', 'S3'):
            dest = ROOT / 'teacher' / arm / str(owner) / f'{view}.json'
            assert not dest.exists(), 'No implicit target resampling'
            attempt = ROOT / 'attempts' / arm / str(owner) / f'{view}.json'
            assert not attempt.exists(), 'Interrupted generation requires explicit audit'
            write(attempt, dict(start=time.time(), owner=owner, view=view, arm=arm))
            conditioned = dict(row, question=row['question'] + '\n\n' + contexts(fact, arm))
            items = prepare(task, [row, conditioned])
            tokens = []
            g = torch.Generator(device='cpu').manual_seed((task['seed'] + 1009 * view) % (2**63))
            statistics = []
            generating = True
            counts['teacher_generations'] += 1
            with torch.inference_mode():
                out0 = runtime.model(**pref(items[0][3]))
                outc = runtime.model(**pref(items[1][3]))
                for t in range(256):
                    log0 = out0.logits[0, -1].float().log_softmax(-1)
                    logc = outc.logits[0, -1].float().log_softmax(-1)
                    p0, pc = log0.exp(), logc.exp()
                    assert torch.isfinite(log0).all() and torch.isfinite(logc).all()
                    if view == 0 and arm == 'S1' and t == 1:
                        prefix = pref(items[0][3])['inputs_embeds']
                        last = runtime.model.get_input_embeddings()(torch.tensor([tokens], device=runtime.device))
                        full = runtime.model(inputs_embeds=torch.cat([prefix, last], 1), use_cache=False,
                                             return_dict=True)
                        torch.testing.assert_close(out0.logits[0,-1], full.logits[0,-1], rtol=1e-5, atol=5e-4)
                        del full, prefix, last
                    probs = (.5 * p0 + .5 * pc).detach().cpu()
                    token = int(torch.multinomial(probs, 1, generator=g))
                    statistics.append(dict(KL_condition_to_Base=float((pc * (logc-log0)).sum()),
                        Base_entropy=float(-(p0*log0).sum()), conditional_entropy=float(-(pc*logc).sum()),
                        token_Base_NLL=float(-log0[token])))
                    tokens.append(token)
                    if token == runtime.adapter.tokenizer.eos_token_id:
                        break
                    pk0, pkc = out0.past_key_values, outc.past_key_values
                    del out0, outc
                    ids = torch.tensor([[token]], device=runtime.device)
                    out0 = runtime.model(input_ids=ids, past_key_values=pk0, use_cache=True, return_dict=True)
                    outc = runtime.model(input_ids=ids, past_key_values=pkc, use_cache=True, return_dict=True)
                    del pk0, pkc
                del out0, outc
            generating = False
            answer = runtime.adapter.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            write(dest, dict(owner=owner, view=view, arm=arm, question=row['question'],
                reference=row['reference'], raw_answer=answer, raw_token_ids=tokens,
                EOS=tokens[-1] == runtime.adapter.tokenizer.eos_token_id, at_cap=len(tokens)==256,
                token_statistics=statistics, same_prefix=True, teacher_frozen=True,
                source_only=True, sampling=dict(temperature=1., top_p=1., fact_weight=.5,
                                               seed=(task['seed']+1009*view) % (2**63))))
            del items
            progress(owner=owner, view=view, arm=arm)


def source_gradient(expert, hook, batches, step):
    expert.zero_grad(set_to_none=True)
    losses = []
    for idx, weight in ((0, .5), (1+(step-1)%4, .5)):
        hook.set_teacher_routing(batches[idx].labels)
        loss = runtime.compute_loss(batches[idx])
        assert torch.isfinite(loss)
        (weight * loss).backward()
        counts['backwards'] += 1
        losses.append(float(loss.detach()))
    return {k: v.grad.detach().clone() for k, v in expert.named_parameters()}, losses


def sampled_batch(value, token_ids):
    """Keep sampled tokenization, including leading whitespace and EOS."""
    if token_ids is None:
        return value
    kwargs, labels, _ = value
    assert token_ids and token_ids[-1] == runtime.adapter.tokenizer.eos_token_id
    assert token_ids.count(runtime.adapter.tokenizer.eos_token_id) == 1
    start = int(torch.where(labels[0] != -100)[0][0])
    ids = torch.tensor([token_ids], device=runtime.device)
    with torch.no_grad():
        embeds = runtime.model.get_input_embeddings()(ids).detach().half().cpu()
    kw = dict(kwargs)
    kw['inputs_embeds'] = torch.cat([kwargs['inputs_embeds'][:,:start], embeds], 1)
    target = torch.tensor([[-100]*start+token_ids], dtype=labels.dtype)
    kw['labels'] = target
    if kwargs.get('attention_mask') is not None:
        kw['attention_mask'] = torch.ones(target.shape, dtype=kwargs['attention_mask'].dtype)
    if kwargs.get('position_ids') is not None:
        kw['position_ids'] = torch.arange(target.shape[1], dtype=kwargs['position_ids'].dtype)[None]
    assert kw['inputs_embeds'].shape[:2] == target.shape
    assert target[0,start:].tolist() == token_ids
    return kw, target, tuple(token_ids)


def train(task, fisher):
    from m3bench_repro.editors.llava_runtime import seed_everything
    owner = task['order']
    source = read(ROOT / 'SOURCE_ROWS.json')[str(owner)]
    target = read(ROOT / 'TARGETS_FROZEN.json')[str(owner)]
    target_tokens = read(ROOT / 'TARGET_TOKENS_FROZEN.json')[str(owner)]
    reference = read(PARENT / 'private/results/C' / f'{owner}.json')
    stepsizes = [x['operator_step_norm'] for x in reference['curve']]
    regularization = read(PARENT / 'private/calibration146' / f'{owner}.json')['coefficient']
    original = r.p.load_state(PARENT / 'private/candidates/C' / f'{owner}.pt')['states']['BASE']
    assert sum(x.numel() for x in original.values()) == 7168
    for arm in ('S0', 'S1', 'S2', 'S3'):
        point = ROOT / 'candidates' / arm / f'{owner}.pt'
        assert not point.exists()
        seed_everything(task['seed'])
        rows = [dict(row, reference=target[arm][v]) for v, row in enumerate(source)]
        items = prepare(task, rows, True)
        frozen_batches = [sampled_batch(x[1], target_tokens[arm][v]) for v,x in enumerate(items)]
        batches = [r.old.batch_on(runtime, b) for b in frozen_batches]
        expert = r.p.expert(original, task['seed'], runtime.device)
        expert.load_state_dict(r.mm.canonicalize(original)[0])
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        curve = []
        scalar = dict(step=0, second_moments=[0., 0.])
        try:
            if arm == 'S0':
                # A nonzero state checks the actual teacher-forced/cache path
                # without consuming an additional scientific training step.
                test = r.old.snapshot(expert)
                test['G1'].fill_(1e-5)
                expert.load_state_dict(test)
                check = r.s.fit.short.mechanical(runtime, hook, batches[0])
                write(ROOT / 'mechanical' / f'{owner}.json', check)
                expert.load_state_dict(r.mm.canonicalize(original)[0])
            # Original production PrefixCache has mathematical parity checks;
            # retain its original five distinct source-prefix behavior.
            with cache_class(runtime, batches) as cache:
                for step in range(1, 161):
                    budget()
                    before = r.old.snapshot(expert)
                    grads, losses = source_gradient(expert, hook, batches, step)
                    selected, scalar, audit = training_math.step(r, before, grads, fisher,
                        regularization, scalar, stepsizes[step-1], 'M13P')
                    expert.load_state_dict(selected)
                    counts['operator_backwards'] += audit['operator_backwards']
                    counts['updates'] += 1
                    curve.append(dict(step=step, source_NLL=losses,
                        operator_step_norm=r.fm.map_norm(before, selected), **audit))
                    if step % 20 == 0:
                        save_state(ROOT / 'resume' / arm / f'{owner}.pt', dict(
                            state=r.old.snapshot(expert), scalar=scalar, step=step,
                            curve=curve, counts=dict(counts), rng=r.c.rng()))
                        progress(owner=owner, arm=arm, step=step)
                assert cache.misses == 5 and cache.hits == 315
            final = r.old.snapshot(expert)
            save_state(point, dict(state=final, scalar=scalar, owner=owner, arm=arm, seed=task['seed']))
            loaded = r.p.load_state(point)['state']
            assert all(torch.equal(loaded[k].cpu(), final[k].cpu()) for k in final)
            # S0 is the unmodified old algorithm. Match every training step,
            # not its final score alone. Original curves remain immutable.
            if arm == 'S0':
                old = read(PARENT / 'private/results/M13P' / f'{owner}.json')['curve']
                assert len(old) == len(curve) == 160
                for a, b in zip(old, curve):
                    assert a['source_NLL'] == b['source_NLL'], 'S0 trajectory drift'
                    assert a['operator_step_norm'] == b['operator_step_norm'], 'S0 operator drift'
            write(ROOT / 'training' / arm / f'{owner}.json', dict(status='PASS', owner=owner,
                arm=arm, steps=160, curve=curve, source_only=True, save_load_exact=True,
                original_S0_curve_exact=arm=='S0', regularization=regularization))
        finally:
            hook.detach()
        del batches, items, expert


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def evaluate(task):
    global generating
    owner = task['order']
    plans = [x for x in read(ROOT / 'EVAL_PLAN.json') if x['owner']==owner]
    items = prepare(task, [x['query'] for x in plans])
    for arm in ('S0', 'S1', 'S2', 'S3'):
        saved = r.p.load_state(ROOT / 'candidates' / arm / f'{owner}.pt')
        expert = r.p.expert(saved['state'], task['seed'], runtime.device)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        try:
            for index, (plan, item) in enumerate(zip(plans, items)):
                query = plan['query']
                dest = ROOT / 'outputs' / arm / str(owner) / f'{index}.json'
                assert not dest.exists(), 'No implicit evaluation regeneration'
                generating = True
                counts['evaluation_generations'] += 1
                with torch.inference_mode():
                    answer, trace = r.s.fit.short.generate(runtime, hook, item[2], item[3])
                generating = False
                tokens = list(answer.raw_token_ids)
                out = dict(owner=owner, arm=arm, query=query, binding=item[4],
                    raw_answer=answer.decoded_text, raw_token_ids=tokens,
                    EOS=tokens[-1]==runtime.adapter.tokenizer.eos_token_id, at_cap=len(tokens)>=1024)
                if arm=='S0':
                    historical = read(Path(plan['historical_path']))
                    assert historical['R0']['raw_token_ids']==tokens, 'S0 historical generation drift'
                    out['historical_token_parity'] = True
                write(dest, out)
                progress(owner=owner, arm=arm, evaluation_index=index)
        finally:
            hook.detach()
        del expert


def main():
    with (ROOT / f'ONCE_{PHASE}_{PART}').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original deadline')))
    signal.alarm(max(1, int(limits['hard_end']-time.time()-300)))
    setup()
    tasks = read(ROOT / 'TASKS.json')[PART::2]
    if PHASE=='teacher':
        for task in tasks:
            teacher(task)
    elif PHASE=='train':
        assert read(ROOT / 'TARGET_REVIEW_COMPLETE.json')['status']=='PASS'
        fisher = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v
                  for k, v in r.p.load_state(PARENT / 'private/OPERATOR_FISHER.pt').items()}
        for task in tasks:
            train(task, fisher)
    elif PHASE=='evaluate':
        assert read(ROOT / 'ALL_CANDIDATES_FROZEN.json')['count']==32
        for task in tasks:
            evaluate(task)
    else:
        raise ValueError(PHASE)
    write(ROOT / f'COMPLETE_{PHASE}_{PART}.json', dict(status='PASS', counts=counts, time=time.time()))


if __name__=='__main__':
    code=0
    try:
        main()
    except BaseException as error:
        code=1
        write(ROOT / f'FAILURE_{PHASE}_{PART}.json', dict(error=repr(error), traceback=traceback.format_exc(),
              counts=counts, time=time.time(), no_automatic_retry=True))
        raise
    finally:
        progress(exit_code=code)
        write(ROOT / f'EXIT_{PHASE}_{PART}.json', dict(exit_code=code, counts=counts, time=time.time()))
