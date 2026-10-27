"""Sequential matched control then fixed source-supported Base-token replay."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V130_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
Q_ROOT = ROOT / 'Q'
CONTROL = Path(os.environ['V126_PARENT'])
START = time.time()
ARMS = ('A_SHORT_REFERENCE', 'B_BASE_SEQUENCE')
counts = dict(forwards=0, backwards=0, target_backwards=0, replay_backwards=0,
              operator_backwards=0, updates=0, generations=0, vision_forwards=0,
              path_positions=0, LM_input_positions=0, replay_supervised_tokens=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


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
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'),
                   str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import kernel
    import module_run as r
    import priority_step
    import completion
    import replay_batch
    from prefix_cache import PrefixCache
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget = budget
    torch.manual_seed(lock['seed'])
    torch.cuda.manual_seed_all(lock['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tasks = read(ROOT / 'TASKS.json')['tasks']
    assert len(tasks) == 14
    assert sum(len(t['replay_protect']) for t in tasks) == 84
    assert sum(len(t['protect_new']) for t in tasks) == 56
    write(ROOT / 'REPLAY_BATCH_SELF_CHECK.json', replay_batch.selfcheck())
    write(ROOT / 'KERNEL_SELF_CHECK.json', kernel.selfcheck())
    write(ROOT / 'PRIORITY_SELF_CHECK.json', priority_step.selfcheck())
    write(ROOT / 'COMPLETION_SELF_CHECK.json', completion.selfcheck())
    contexts = {(v['task_id'], v['query_id']): v for v in read(ROOT / 'PREFIX_CONTEXTS_FROZEN.json')['contexts']}
    assert len(contexts) == 70
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    template = read(PARENT / 'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    fisher = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v
              for k, v in r.p.load_state(PARENT / 'private/OPERATOR_FISHER.pt').items()}
    def count(_module, args, kwargs):
        budget()
        counts['forwards'] += 1
        inputs = kwargs.get('inputs_embeds')
        if inputs is None:
            inputs = kwargs.get('input_ids')
        if inputs is None and args:
            inputs = args[0]
        if isinstance(inputs, torch.Tensor):
            counts['LM_input_positions'] += int(inputs.shape[0] * inputs.shape[1])
    fh = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards'] + 1))

    def prepare(rows, training=False):
        runtime.model.half()
        try:
            return r.s.prepare(runtime, bindings, template, rows, training)
        finally:
            r.old.fp32(runtime)

    def fresh(q):
        state = r.mm.canonicalize(r.p.tr.TT4(lock['seed'], 8, 8).state_dict())[0]
        expert = r.p.expert(state, lock['seed'], runtime.device)
        kernel.install(expert, q)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        return expert, hook

    def emit(item, hook, path, smoke=False):
        budget()
        counts['generations'] += 1
        if smoke:
            with hook.generation_request():
                answer, _ = r.s.fit.short.generate(runtime, None, item[2], item[3])
            assert hook.last_generation_trace and all(v['active_residual_norm'] == 0 for v in hook.last_generation_trace)
        else:
            answer, _ = r.s.fit.short.generate(runtime, hook, item[2], item[3])
        tokens = list(answer.raw_token_ids)
        value = dict(binding=item[4], raw_answer=answer.decoded_text, raw_token_ids=tokens,
                     EOS=bool(tokens) and tokens[-1] == runtime.adapter.tokenizer.eos_token_id,
                     at_cap=len(tokens) >= 1024)
        write(path, value)
        return value

    frozen_rows = read(ROOT / 'SUPERVISION_FROZEN.json')['rows']
    supervision = {(v['task_id'], v['query_id']): v for v in frozen_rows}
    assert len(supervision) == 84
    assert sum(v['selection'] == 'FROZEN_FULL_SUPPORTED_BASE' for v in frozen_rows) == 80
    inherited = read(ROOT / 'Q_INHERITANCE.json')
    assert set(inherited) == {'T07', 'T12', 'T13'}
    basis_audits = {}
    for task in tasks:
        tid = task['task_id']
        dest = Q_ROOT / tid
        dest.mkdir(parents=True)
        original = read(ROOT / 'reference' / tid / 'BASIS_ADMISSION.json')
        if tid in inherited:
            source = Path(inherited[tid]['path'])
            assert source.is_file() and not source.is_symlink()
            dest.joinpath('basis.pt').symlink_to(source)
            actual = read(source.parent / 'BASIS_ADMISSION.json')
            assert actual['rank'] == original['rank'] and actual['path_positions'] == original['path_positions']
            audit = dict(actual, inherited_V129=True, no_checkpoint_copy=True)
        else:
            vectors = []
            for item in prepare(task['protect']):
                captured = []
                def capture(_module, args):
                    assert args[0].shape[0] == 1
                    captured.append(args[0][0, -1:].detach().cpu().double())
                handle = runtime.get_module(r.c.LAYER).register_forward_pre_hook(capture)
                try:
                    answer = emit(item, None, dest / 'Base_paths' / (item[0]['query_id'] + '.json'))
                finally:
                    handle.remove()
                old = task['Base_protect'][item[0]['query_id']]
                assert answer['raw_answer'] == old['raw_answer'] and answer['raw_token_ids'] == old['raw_token_ids']
                assert answer['EOS'] and not answer['at_cap'] and len(captured) == len(answer['raw_token_ids'])
                vectors.extend(captured)
            counts['path_positions'] += len(vectors)
            q, audit = kernel.basis(vectors)
            assert audit['rank'] == original['rank'] and audit['path_positions'] == original['path_positions']
            assert max(abs(a - b) for a, b in zip(audit['singular_values'], original['singular_values'])) <= 1e-7
            torch.save(q, dest / 'basis.pt')
            del vectors, q
        basis_audits[tid] = audit
        write(dest / 'BASIS_ADMISSION.json', dict(audit, counts=dict(counts), time=time.time()))
        write(ROOT / 'PROGRESS.json', dict(phase='Q_REBUILD', task=tid, completed=len(basis_audits), counts=dict(counts), time=time.time()))
    assert counts['generations'] == 22 and sum(v['rank'] for v in basis_audits.values()) == 746
    write(ROOT / 'ALL_BASES_FROZEN.json', dict(audits=basis_audits, new_Base_generations=22,
        inherited_Base_generations=6, counts=dict(counts), time=time.time()))

    def train_arm(arm):
        completed = []
        for task in tasks:
            task_root = ROOT / task['task_id']
            task_root.mkdir(exist_ok=True)
            q = torch.load(Q_ROOT / task['task_id'] / 'basis.pt', map_location=runtime.device, weights_only=True)
            prepared = prepare(task['train'] + task['replay_protect'], True)
            batches = [r.old.batch_on(runtime, v[1]) for v in prepared]
            replay_admission = []
            for offset, item in enumerate(prepared[5:]):
                frozen = supervision[(task['task_id'], item[0]['query_id'])]
                assert item[4] == frozen['binding']
                assert batches[5 + offset].labels[batches[5 + offset].labels != -100].tolist() == frozen['short_reference_token_ids']
                if arm == 'B_BASE_SEQUENCE' and frozen['selection'] == 'FROZEN_FULL_SUPPORTED_BASE':
                    batches[5 + offset] = replay_batch.make_batch(runtime, item[3], frozen['selected_token_ids'])
                chosen = frozen['selected_token_ids'] if arm == 'B_BASE_SEQUENCE' else frozen['short_reference_token_ids']
                assert batches[5 + offset].labels[batches[5 + offset].labels != -100].tolist() == chosen
                assert batches[5 + offset].labels.shape[1] < runtime.model.config.max_position_embeddings
                replay_admission.append(dict(query_id=item[0]['query_id'], selection=frozen['selection'],
                    actual_label_tokens=len(chosen), all_prompt_and_token_checks=True))
            write(task_root / (arm + '_REPLAY_ADMISSION.json'), replay_admission)
            assert len(batches) == 5 + len(task['replay_protect'])
            completion_batches, completion_masks = [], []
            for item in prepared[:5]:
                context = contexts[(task['task_id'], item[0]['query_id'])]
                assert context['original_binding'] == item[4]
                batch, mask = completion.make_batch(runtime, item[3], context['prefix_token_ids'])
                assert batch.labels.shape[1] < runtime.model.config.max_position_embeddings
                completion_batches.append(batch)
                completion_masks.append(mask)
            write(task_root / 'COMPLETION_INPUT_ADMISSION.json', dict(rows=5, prefix_lengths=[
                contexts[(task['task_id'], v['query_id'])]['audit_prefix_tokens'] for v in task['train']],
                EOS_only_labels=True, full_previous_answer_predictor_mask=True, time=time.time()))
            smoke_item = prepare([task['target']])[0]
            calibration = read(ROOT / 'reference' / task['task_id'] / 'CALIBRATION.json')
            dest = task_root / arm
            dest.mkdir()
            expert, hook = fresh(q)
            scalar = dict(step=0, second_moments=[0., 0.])
            curve = []
            old_curve = read(CONTROL / task['task_id'] / 'REFERENCE_PRIORITY_COMPLETION/TRAIN_COMPLETE.json')['curve']
            try:
                smoke = emit(smoke_item, hook, dest / 'ZERO_SMOKE.json', True)
                assert smoke['binding'] == task['Base_target']['binding']
                assert smoke['raw_answer'] == task['Base_target']['raw_answer']
                assert smoke['raw_token_ids'] == task['Base_target']['raw_token_ids']
                with PrefixCache(runtime, batches + completion_batches) as prefix:
                    for step in range(1, 161):
                        budget()
                        before = r.old.snapshot(expert)
                        expert.zero_grad(set_to_none=True)
                        replay_index = (step - 1) % len(task['replay_protect'])
                        values = []
                        for index, weight in ((0, .25), (1 + (step - 1) % 4, .25)):
                            hook.set_teacher_routing(batches[index].labels)
                            loss = runtime.compute_loss(batches[index])
                            assert torch.isfinite(loss)
                            (weight * loss).backward()
                            counts['backwards'] += 1
                            counts['target_backwards'] += 1
                            values.append(float(loss.detach()))
                        source = {k: v.grad.detach().clone() for k, v in expert.named_parameters()}
                        expert.zero_grad(set_to_none=True)
                        completion_values = []
                        for index in (0, 1 + (step - 1) % 4):
                            hook.set_request_routing(completion_masks[index])
                            loss = runtime.compute_loss(completion_batches[index])
                            assert torch.isfinite(loss)
                            (.25 * loss).backward()
                            counts['backwards'] += 1
                            counts['target_backwards'] += 1
                            completion_values.append(float(loss.detach()))
                        eos_gradient = {k: v.grad.detach().clone() for k, v in expert.named_parameters()}
                        expert.zero_grad(set_to_none=True)
                        hook.set_teacher_routing(batches[5 + replay_index].labels)
                        loss = runtime.compute_loss(batches[5 + replay_index])
                        assert torch.isfinite(loss)
                        loss.backward()
                        counts['backwards'] += 1
                        counts['replay_backwards'] += 1
                        counts['replay_supervised_tokens'] += int((batches[5 + replay_index].labels != -100).sum())
                        values.append(float(loss.detach()))
                        replay = {k: v.grad.detach().clone() for k, v in expert.named_parameters()}
                        # Fisher is added exactly once inside step; this argument supplies EOS plus replay.
                        auxiliary = {k: eos_gradient[k] + replay[k] for k in source}
                        state, scalar, audit = priority_step.step(r, before, source, auxiliary, fisher,
                            calibration['coefficient'], scalar, calibration['targets'][step - 1])
                        audit['primary_gradient'] = 'Full reference CE only; unchanged .25 native + .25 FIT'
                        audit['auxiliary_gradient'] = 'Unchanged .25 native EOS + .25 FIT EOS + RR replay1 + Fisher once'
                        expert.load_state_dict(state)
                        counts['updates'] += 1
                        counts['operator_backwards'] += audit['operator_backwards']
                        curve.append(dict(step=step, source_NLL=values[:2], replay_NLL=values[2],
                            completion_EOS_NLL=completion_values,
                            replay_query_id=task['replay_protect'][replay_index]['query_id'],
                            latent_operator_step_norm=r.fm.map_norm(before, state), **audit))
                        curve[-1]['replay_label_tokens'] = int((batches[5 + replay_index].labels != -100).sum())
                        curve[-1]['replay_gradient_norm'] = sum(float(v.double().square().sum()) for v in replay.values()) ** .5
                        curve[-1]['auxiliary_cap_norm_means_EOS_replay_Fisher_combined'] = True
                        if arm == 'A_SHORT_REFERENCE':
                            previous = old_curve[step - 1]
                            for key in ('source_NLL', 'completion_EOS_NLL'):
                                assert max(abs(a - b) for a, b in zip(curve[-1][key], previous[key])) <= 1e-6, (task['task_id'], step, key)
                            assert abs(curve[-1]['replay_NLL'] - previous['replay_NLL']) <= 1e-6, (task['task_id'], step, 'replay')
                        if step % 20 == 0:
                            curve[-1]['effective_operator_step_norm'] = kernel.map_norm(r, before, state,
                                q)
                            saved = dict(state=r.old.snapshot(expert), scalar=scalar, rng=r.c.rng(), curve=curve,
                                counts=dict(counts), source_commit=lock['source_commit'], task_id=task['task_id'],
                                arm=arm, fixed_basis=str(Q_ROOT / task['task_id'] / 'basis.pt'),
                                training_recipe='V126 unchanged; only B uses own frozen full-supported Base replay tokens')
                            torch.save(saved, dest / 'resume.tmp')
                            (dest / 'resume.tmp').replace(dest / 'resume.pt')
                            write(ROOT / 'PROGRESS.json', dict(phase='TRAIN', task=task['task_id'], arm=arm,
                                step=step, completed=len(completed), counts=dict(counts), time=time.time()))
                    assert prefix.misses == len(batches) + 5 and prefix.hits == 800 - len(batches) - 5
                torch.save(r.old.snapshot(expert), dest / 'delta.pt')
                loaded = torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True)
                assert all(torch.equal(v, loaded[k]) for k, v in r.old.snapshot(expert).items())
                completed.append(task['task_id'] + '/' + arm)
                write(dest / 'TRAIN_COMPLETE.json', dict(step=160, fixed_steps=True, curve=curve,
                    fixed_basis_rank=q.shape[1],
                    replay_rows=len(task['replay_protect']), no_evaluation_selection=True,
                    save_load_exact=True, counts=dict(counts), time=time.time()))
            finally:
                hook.detach()
            del q, batches, completion_batches, completion_masks, prepared, smoke_item
        expected = 2240 if arm == 'A_SHORT_REFERENCE' else 4480
        assert len(completed) == 14 and counts['updates'] == expected
        assert counts['backwards'] == expected * 5 and counts['operator_backwards'] == expected
        write(ROOT / ('ALL_TRAINING_FROZEN_' + arm + '.json'), dict(completed=completed, counts=dict(counts), time=time.time()))

    def evaluate_arm(arm):
        mechanism, comparisons = [], []
        for task in tasks:
            task_root = ROOT / task['task_id']
            q = torch.load(Q_ROOT / task['task_id'] / 'basis.pt', map_location=runtime.device, weights_only=True)
            evaluation = prepare(task['train'] + task['heldout'] + task['protect'] + task['protect_extra'] + task['protect_new'])
            dest = task_root / arm
            expert, hook = fresh(q)
            expert.load_state_dict(torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True))
            try:
                for item in evaluation:
                    answer = emit(item, hook, dest / 'outputs' / (item[0]['query_id'] + '.json'))
                    if arm == 'A_SHORT_REFERENCE':
                        previous = read(CONTROL / task['task_id'] / 'REFERENCE_PRIORITY_COMPLETION/outputs' / (item[0]['query_id'] + '.json'))
                        comparisons.append(dict(task_id=task['task_id'], query_id=item[0]['query_id'],
                            binding_exact=answer['binding'] == previous['binding'],
                            complete_text_exact=answer['raw_answer'] == previous['raw_answer'],
                            complete_tokens_exact=answer['raw_token_ids'] == previous['raw_token_ids']))
                    if item[0]['query_id'] in task['Base_protect']:
                        old = task['Base_protect'][item[0]['query_id']]
                        mechanism.append(dict(task_id=task['task_id'], query_id=item[0]['query_id'],
                            Base_tokens_exact=answer['raw_token_ids'] == old['raw_token_ids'],
                            Base_fulltext_exact=answer['raw_answer'] == old['raw_answer'],
                            maximum_active_residual_norm=max(v['active_residual_norm'] for v in hook.last_generation_trace)))
                write(dest / 'GENERATION_COMPLETE.json', dict(rows=len(evaluation), time=time.time()))
            finally:
                hook.detach()
        assert len(mechanism) == 28
        write(ROOT / ('PATH_MECHANISM_AUDIT_' + arm + '.json'), mechanism)
        write(ROOT / ('GENERATION_COMPLETE_' + arm + '.json'), dict(formal_rows=238, counts=dict(counts), time=time.time()))
        return comparisons

    def retire_control():
        paths = [ROOT / task['task_id'] / 'A_SHORT_REFERENCE' / name
                 for task in tasks for name in ('delta.pt', 'resume.pt')]
        manifest = []
        for path in paths:
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve())
            manifest.append(dict(path=str(path), bytes=path.stat().st_size))
        assert len(manifest) == 28
        write(ROOT / 'A_RETIREMENT_PLAN.json', dict(files=manifest,
            completed_consumers='All160/save-load/238 generation/exact V126 reproduction; B begins from zero'))
        for path in paths:
            path.unlink()
        assert all(not path.exists() for path in paths)
        write(ROOT / 'A_RETIREMENT_RECEIPT.json', dict(files=28, bytes=sum(v['bytes'] for v in manifest),
            no_checkpoint_copy=True, Q_retained_for_B=True, time=time.time()))

    for arm in ARMS:
        train_arm(arm)
        comparisons = evaluate_arm(arm)
        if arm == 'A_SHORT_REFERENCE':
            assert len(comparisons) == 238 and counts['generations'] == 274
            passed = all(v['binding_exact'] and v['complete_text_exact'] and v['complete_tokens_exact'] for v in comparisons)
            write(ROOT / 'A_CONTROL_REPRODUCTION.json', dict(pass_all238=passed, comparisons=comparisons,
                inherited_original_semantic_labels=passed, time=time.time()))
            if not passed:
                write(ROOT / 'COMPLETE.json', dict(status='CONTROL_REPRODUCTION_FAILED_B_NOT_STARTED',
                    counts=dict(counts), time=time.time()))
                fh.remove()
                vh.remove()
                return
            assert counts['replay_supervised_tokens'] == 10355
            retire_control()
            write(ROOT / 'B_ADMISSION.json', dict(exact_control238_reproduced=True,
                labels_frozen_before_A=True, only_replay_sequence_changes=True, time=time.time()))
    assert counts['updates'] == counts['operator_backwards'] == 4480
    assert counts['backwards'] == 22400 and counts['generations'] == 526
    assert counts['target_backwards'] == 17920 and counts['replay_backwards'] == 4480
    assert counts['replay_supervised_tokens'] == 10355 + 57945
    write(ROOT / 'COMPLETE.json', dict(status='A_REPRODUCED_B_BLIND_REVIEW_PENDING',
        counts=dict(counts), time=time.time()))
    fh.remove()
    vh.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(ROOT / 'FAILURE.json', dict(error=traceback.format_exc(), counts=counts, seconds=time.time() - START, time=time.time()))
        raise
    finally:
        write(ROOT / 'WORKER_EXIT.json', dict(code=code, counts=counts, seconds=time.time() - START, time=time.time()))
