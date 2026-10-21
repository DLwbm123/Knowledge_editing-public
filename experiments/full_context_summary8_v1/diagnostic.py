"""Independent source-six evaluator for all five arms; scalar response storage."""
from pathlib import Path
import os
import torch
from experiments.fixed_input_guard8_v1.executor import atomic_json
from experiments.fixed_input_guard8_v1.diagnostic import token_metrics, generation_metrics, full_response

@torch.no_grad()
def evaluate(arm, owner, config):
    from experiments.fixed_input_guard8_v1 import launch
    from experiments.fixed_input_guard8_v1.adapter import RuntimeAdapter, context_positions
    from m3bench_repro.editors.llava_runtime import EditorRecord
    c, root = launch.helpers()
    assert config['source6_diagnostic'] and owner in (6, 7, 8)
    folder = root/'private/fixed8'/arm
    weights = folder/'diagnostic_weights'
    output = root/'private/evaluation/source6_diagnostic'/arm
    output.mkdir(parents=True, exist_ok=True)
    Budget = launch.install_budget(arm)
    budget = Budget(folder/'diagnostic-budget.json')
    gpu = config['physical_GPU']
    with c.lease(gpu):
        with c.resources() as r:
            next(s for s in r['gpu_sessions'] if s['pid'] == os.getpid() and not s.get('ended_epoch'))['arm'] = arm
        runtime, bindings = c.load(gpu)
        adapter = RuntimeAdapter(runtime, budget)
        for p in runtime.model.parameters():
            p.requires_grad_(False)
        try:
            task = c.read(root/'private/BENCHMARK146_QUEUE.json')['tasks'][5]
            row = task['native']
            record = EditorRecord(record_id=task['edit_id'], dataset=row['dataset'], question=row['question'],
                target=row['reference'], official_rephrase=task['fit_questions'][0],
                image_path=launch.image_path(row['image_path']), relative_image_path=row['original_image_path'],
                formal_sequence_position=6, question_type='DIAGNOSTIC_ONLY')
            batch = adapter.pack(runtime.build_edit_batch(record))
            eos = int(runtime.adapter.tokenizer.eos_token_id)
            cap = int(runtime.generation_config['max_new_tokens'])
            paths = sorted(weights.glob(f'{owner}_*.pt'))
            steps = sorted((folder/'steps'/str(owner)).glob('*.json'), key=lambda p: int(p.stem))
            expected = [20] if owner == 6 else [int(p.stem) for p in steps if c.read(p)['accepted']]
            assert [int(p.stem.split('_')[1]) for p in paths] == expected
            if owner == 6:
                reference = torch.load(paths[0], weights_only=True, map_location='cpu')['native']
                previous = reference
                ref_tokens = None
            else:
                reference = torch.load(weights/'reference6.pt', weights_only=True, map_location='cpu')['native']
                previous = torch.load(weights/'previous.pt', weights_only=True, map_location='cpu')['native']
                ref_tokens = c.read(output/'6_20/generation.json')['R0']['raw_token_ids']
            results = []
            for path in paths:
                state = torch.load(path, weights_only=True, map_location='cpu')
                native = state['native']
                assert native.shape == adapter.weight.shape and native.dtype == adapter.weight.dtype
                adapter.weight.copy_(native.to(adapter.weight.device))
                captured = []
                handle = adapter.layer.register_forward_pre_hook(lambda module, args: captured.append(args[0][0].detach().float().cpu()))
                try:
                    z, y = adapter.logits(batch)
                finally:
                    handle.remove()
                assert len(captured) == 1 and len(captured[0]) == batch[1].shape[1]
                h = captured[0]
                destination = output/path.stem
                destination.mkdir()
                metrics = token_metrics(z, y, eos)
                if owner == 6:
                    fixed_competitor = metrics['competitor_ids']
                else:
                    fixed_competitor = c.read(output/'6_20/METRICS.json')['competitor_ids']
                metrics['fixed_competitor_margin'] = (z.double().gather(1,y[:,None]).squeeze(1)-z.double().gather(1,torch.tensor(fixed_competitor,device=z.device)[:,None]).squeeze(1)).cpu().tolist()
                fixed = context_positions(batch, 4, anchors=True).tolist()
                sampled = context_positions(batch, 7).tolist()
                valid = batch[0].get('attention_mask', torch.ones_like(batch[1]))[0].bool().tolist()
                start, end = batch[2]
                answer = set(((batch[1][0, 1:] != -100).nonzero().flatten()).tolist())
                responses = {}
                for name, base in [('accepted_increment', previous), ('cumulative_since6', reference)]:
                    value = full_response(native.float()-base.float(), h, adapter.weight.device)
                    norms = value.norm(dim=1)
                    responses[name] = dict(per_position_l2=norms.tolist(),
                        fixed_anchor_l2=norms[fixed].tolist(), sampled_context_l2=norms[sampled].tolist(),
                        unsampled_l2=norms[[i for i, v in enumerate(valid) if v and i not in sampled]].tolist())
                    # Persist every position's norm, not the redundant 4096-channel vector.
                if owner == 6:
                    torch.save(h, output/'reference6_full_context_h.pt')
                else:
                    h6 = torch.load(output/'reference6_full_context_h.pt', weights_only=True, map_location='cpu')
                    assert h.shape == h6.shape
                    metrics['full_context_h_max_change_since6'] = float((h-h6).abs().max())
                generated = launch.generate(runtime, bindings, row, destination/'generation.json',
                    dict(arm=arm, stage='SOURCE6_DIAGNOSTIC', owner=owner, slot=state['slot'], code=config['implementation_commit']), budget)
                ids = generated['R0']['raw_token_ids']
                if ref_tokens is None:
                    ref_tokens = ids
                metrics.update(owner=owner, slot=state['slot'], generation=generation_metrics(ids, ref_tokens, cap),
                    response=responses, context_positions=list(range(len(h))), valid_positions=valid,
                    fixed_positions=fixed, sampled_positions=sampled,
                    strata=['visual' if start <= i < end else 'answer_predictor' if i in answer else 'prompt_or_other' for i in range(len(h))],
                    diagnostic_only=True, training_feedback=False)
                atomic_json(destination/'METRICS.json', metrics)
                previous = native
                torch.save(dict(native=native), weights/'previous.tmp')
                (weights/'previous.tmp').replace(weights/'previous.pt')
                if owner == 6:
                    path.rename(weights/'reference6.pt')
                else:
                    size = path.stat().st_size
                    path.unlink()
                    atomic_json(destination/'WEIGHT_RETIRED.json', dict(bytes=size, reason='Diagnostic metrics, full responses and generation durable'))
                results.append(dict(owner=owner, slot=state['slot'], CE=metrics['CE'],
                    min_target_margin=min(metrics['target_margin']), generation=metrics['generation']))
            atomic_json(output/f'COMPLETE_{owner}.json', dict(status='PASS', snapshots=len(paths), results=results,
                zero_accepted_updates=not paths, training_feedback=False, counters=budget.data['counts']))
            if owner == 8:
                chronological = [c.read(p) for p in sorted(output.glob('*_*/METRICS.json'))]
                baseline = chronological[0]
                def first(predicate):
                    return next((dict(owner=m['owner'], slot=m['slot']) for m in chronological[1:] if predicate(m)), None)
                atomic_json(output/'FIRST_CHANGE.json', dict(
                    first_generation_divergence=first(lambda m: m['generation']['first_divergence_from_source6'] is not None),
                    first_baseline_positive_target_margin_lost=first(lambda m: any(a > 0 and b <= 0 for a, b in zip(baseline['target_margin'], m['target_margin']))),
                    first_new_cap_hit=first(lambda m: m['generation']['cap_hit'] and not baseline['generation']['cap_hit']),
                    first_increased_4gram_repetition=first(lambda m: m['generation']['most_frequent_4gram_count'] > baseline['generation']['most_frequent_4gram_count']),
                    baseline_target_margins=baseline['target_margin'], baseline_generation=baseline['generation'],
                    evaluated_states=len(chronological), semantic_correctness_not_inferred=True))
                retired = []
                for name in ('reference6.pt', 'previous.pt'):
                    path = weights/name
                    retired.append(dict(name=name, bytes=path.stat().st_size))
                    path.unlink()
                h_path = output/'reference6_full_context_h.pt'
                retired.append(dict(name=h_path.name, bytes=h_path.stat().st_size))
                h_path.unlink()
                atomic_json(output/'REFERENCE_WEIGHTS_RETIRED.json', dict(files=retired, reason='All source7-8 diagnostics durable'))
        finally:
            adapter.close()
