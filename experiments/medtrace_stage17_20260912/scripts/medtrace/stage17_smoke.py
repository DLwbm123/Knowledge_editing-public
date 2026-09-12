"""One already-exposed DEV native on GPU3; zero training and zero Judge calls."""
import argparse
import io
import json
from pathlib import Path
import sys
import time
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(config, out):
    import torch
    from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert, predictor_mask
    from m3bench_repro.editors.llava_runtime import EditorRecord
    from scripts.medtrace.run_realmodel_core import LAYER, load_real_runtime
    from scripts.medtrace.stage17_contract import base_only, prefix_router, require_binding

    if out.exists():
        raise FileExistsError('reuse the existing mechanical result; no automatic rerun')
    if config['scope'] != 'DEV_NATIVE_ONLY_MECHANICAL' or config['gpu'] != '3':
        raise ValueError('this runner authorizes no formal input or training')
    manifest = json.loads(Path(config['dev_manifest']).read_text())
    if manifest['event_count'] != 16 or len(manifest['event_ids']) != 16:
        raise ValueError('expected frozen DEV16 metadata')
    # Only the first already-exposed DEV event; never load formal or Base answers.
    with Path(config['dev_inputs']).open() as stream:
        event = json.loads(next(stream))
    if event['event_id'] != manifest['event_ids'][0]:
        raise ValueError('DEV event order mismatch')
    started = time.monotonic()
    print('LOADING_FROZEN_RUNTIME', flush=True)
    runtime = load_real_runtime(argparse.Namespace(cpu_gate=Path(config['cpu_gate'])))
    runtime.run_root = out.parent
    frozen = [(p, p._version, p.data_ptr()) for p in runtime.model.parameters()]
    if any(p.requires_grad for p, _, _ in frozen):
        raise RuntimeError('Base is not frozen')
    record = EditorRecord.from_dict(event['edit_record'])
    query = replace(record, record_id='query', target='', official_rephrase='')
    batch = runtime.build_question_batch(query)
    raw = runtime.adapter.prepare_inputs(query.image_path, query.question, None)
    layer = runtime.get_module(LAYER)
    cp = AsymmetricCPExpert(layer.in_features, layer.out_features, 4).to(runtime.device)
    expert = LowRankExpert(cp, 20260912, rank=4).to(runtime.device)
    expert.requires_grad_(False)
    hook = MedTraceLayerHook(layer, expert)
    binding = dict(input={'prompt_ids': raw['input_ids'].tolist(),
        'attention': raw['attention_mask'].tolist(), 'image_tensor': raw['image_sha256']},
        runtime=config['runtime_lock'], generation=runtime.generation_config,
        writer='zero_freeR4_mechanical', prefix=1, training='NONE')

    def key():
        with base_only([hook]), torch.no_grad():
            return runtime.extract_layer_input_key(batch,
                module_path=runtime.target_lock['balancedit']['targets'][0], pooling='mean').cpu()

    def generate(active=False):
        with torch.inference_mode():
            if active:
                with hook.generation_request():
                    result = runtime.adapter.generate_prepared_with_result(raw, runtime.generation_config)
            else:
                result = runtime.adapter.generate_prepared_with_result(raw, runtime.generation_config)
        return dict(binding=binding, tokens=list(result.raw_token_ids), text=result.decoded_text)

    print('RUNNING_ONE_DEV_NATIVE', flush=True)
    initial_key, base = key(), generate()
    hook.attach()
    try:
        zero = generate(True)
        # A nonzero resident expert must not contaminate Base routing features.
        with torch.no_grad():
            expert.B.fill_(1e-5)
        with hook.generation_request():
            runtime.adapter.generate_prepared_with_result(raw, dict(runtime.generation_config, max_new_tokens=1))
        routed_key = key()
        stream = io.BytesIO()
        torch.save(expert.state_dict(), stream); stream.seek(0)
        restored = LowRankExpert(cp, 20260912, rank=4).to(runtime.device)
        restored.load_state_dict(torch.load(stream, map_location=runtime.device, weights_only=True))
        save_load = all(torch.equal(v, restored.state_dict()[k]) for k, v in expert.state_dict().items())
        off = generate()
    finally:
        hook.detach()
    detached = generate()
    teacher = runtime.build_edit_batch(record)
    mask = predictor_mask(teacher.labels, teacher.attention_mask)
    hook.set_teacher_routing(teacher.labels)
    mask_equal = torch.equal(hook.token_mask, mask)
    hook.clear_request_routing()
    router = prefix_router([dict(logical_edit_id='first', key=initial_key, radius=0., label=[999])], 1)
    decision = router.route(routed_key)
    require_binding(zero['binding'], base['binding'])
    unchanged = all(p._version == version and p.data_ptr() == pointer and not p.requires_grad
                    for p, version, pointer in frozen)
    checks = dict(zero_effect=base == zero, disabled_and_detached=base == off == detached,
        base_features_unchanged=torch.equal(initial_key, routed_key), state_save_load=save_load,
        predictor_mask=mask_equal, eos_in_teacher=int(runtime.adapter.tokenizer.eos_token_id) in teacher.target_token_ids,
        no_gold_router=router.labels == [()], self_route=decision.logical_edit_id == 'first',
        request_cleanup=not hook.enabled and not hook.generation_routing and hook._handle is None,
        base_parameter_versions_unchanged=unchanged, free_parameters=sum(p.numel() for p in expert.parameters()) == 73728)
    result = dict(status='PASS' if all(checks.values()) else 'MECHANICAL_FAIL', checks=checks,
        dev_edits=1, training_steps=0, judge_calls=0, generation_calls=5,
        note='four canonical generations and one 1-token lifecycle probe; no semantic qualification',
        canonical_cap=runtime.generation_config['max_new_tokens'],
        cap_hit=len(base['tokens']) >= runtime.generation_config['max_new_tokens'],
        base_empty=not bool(base['text'].strip()), runtime_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent/'SMOKE_PRIVATE.json').write_text(json.dumps(dict(result, binding=binding,
        source_event=event['event_id'], outputs=dict(base=base, zero=zero, off=off, detached=detached)), indent=2)+'\n')
    with out.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result), flush=True)
    if result['status'] != 'PASS':
        raise RuntimeError('mechanical checks failed; preserve all evidence')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    run(json.loads(args.config.read_text()), args.out)
