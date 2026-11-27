"""Teacher-forced Base/MED/MDU on a frozen view and locality panel."""
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import signal
import sys
import time

import torch


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def resolve_image(path):
    if os.access(path, os.R_OK):
        return Path(path)
    image_root = os.environ.get('FROZEN_IMAGE_ROOT')
    if image_root and '/DataP/' in path:
        relocated = Path(image_root) / path.split('/DataP/', 1)[1]
        if os.access(relocated, os.R_OK):
            return relocated
    raise FileNotFoundError('frozen image unavailable')


def main():
    root = Path.cwd()
    protocol = read(root / 'protocol.json')
    remaining = int(protocol['deadline'] - time.time())
    assert remaining > 0
    signal.alarm(remaining)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    outputs = root.parent
    parent = outputs / 'medtrace-curriculum146-v1-20261017/run/private'
    stages = [outputs / name / 'run' for name in (
        'med-mdu-boundary-v1-20261105', 'med-medinit-101-v1-20261104', 'med-mdu150-v1-20261105'
    )]
    sources = [outputs / 'med-medinit-loss-v1-20261103/run', *stages[1:]]
    sys.path[:0] = [str(p) for p in (
        stages[2] / 'MDU/tools', outputs / 'med-fixed-depth-v1-20261030/run/tools',
        parent / 'tools', parent / 'source', Path(os.environ['M3BENCH_LLAVA_SOURCE'])
    )]
    from accelerate import dispatch_model
    from llava.model import LlavaMistralForCausalLM
    from m3bench_repro.editors.llava_runtime import EditorRecord, seed_everything
    from methods.medtrace.core import MedTraceLayerHook
    from scripts.editor_paperspec_formal import assert_official_llavamed_source, load_runtime
    from structures import TT4
    from train_pair import state_digest

    med_bank = load(outputs / 'med150-v1-20261027/run/retained/MED150.pt')
    task_by_id = {x['edit_id']: x for x in med_bank['tasks']}
    task_by_owner = {x['order']: x for x in med_bank['tasks']}
    bank = read(stages[2] / 'MDU/BANK_INPUTS_PRIVATE.json')
    bank_rows = {x['query']['query_id']: x for x in bank['records']}
    bank_views = defaultdict(list)
    for entry in bank['records']:
        v = entry['query']
        bank_views[v['image_sha256'], v['question']].append(v)
    assert len(bank_rows) == 2148 and len(task_by_owner) == 150
    state = {'MED': med_bank['states'], 'MDU': {}}
    reference_rows = {}
    first_step = {}
    for stage, source in zip(stages, sources):
        entries = {e['task']['order']: e for e in read(source / 'MDU/TRAIN_SOURCE_BOUND_PRIVATE.json')}
        arm = stage / 'MDU'
        receipts = {r['owner']: r for f in arm.glob('TRAIN_COMPLETE_*.json') for r in read(f)['results']}
        for owner, r in receipts.items():
            e = entries[owner]
            assert e['task']['edit_id'] == task_by_owner[owner]['edit_id']
            assert owner not in reference_rows and len(e['source_rows']) == 5
            reference_rows[owner] = e['source_rows']
            task = task_by_owner[owner]
            expert = task['edit_id']
            c = load(arm / 'weights' / f'{owner}_MDU.pt')
            assert expert not in state['MDU'] and c['steps'] == r['steps'] == 160
            assert c['module'] == 'model.layers.21.mlp.down_proj'
            assert c['initial_digest'] == state_digest(state['MED'][expert])
            assert state_digest(c['state']) == r['final_digest']
            state['MDU'][expert] = c['state']
            first_step[owner] = c['curve'][0]['ce']
    assert len(reference_rows) == len(state['MDU']) == 150

    rows = read(root / 'panel.json')
    lineage = outputs / 'm3bench_task_specific_core9_data/20260903T013245Z/base_inventory/BASE_QUERY_LINEAGE_CORE9.jsonl'
    want = {r['query'] for r in rows if r['role'] == 'T2G'}
    view_types = {}
    with lineage.open() as stream:
        for line in stream:
            e = json.loads(line)
            if e['query_id'] not in want:
                continue
            tags = {x.get('legacy_derived_probe_id', '').split('::T2G::', 1)[-1]
                    for x in e['lineage'] if '::T2G::' in x.get('legacy_derived_probe_id', '')}
            if tags:
                assert len(tags) == 1
                view_types[e['query_id']] = next(iter(tags))
    assert len(view_types) == 557
    for row in rows:
        row['variant'] = view_types.get(row['query'], 'unknown') if row['role'] == 'T2G' else 'T2L'
    for owner, pairs in reference_rows.items():
        assert [p['audit_role'] for p in pairs] == ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']
        for i, p in enumerate(pairs):
            candidates = bank_views[p['image_sha256'], p['question']]
            assert len(candidates) == 1
            query = candidates[0]
            assert (query['query_id'] == task_by_owner[owner]['edit_id'] if i == 0
                    else query['audit_role'] == 'FIT')
            rows.append({'owner': owner, 'query': query['query_id'], 'role': p['audit_role'],
                         'question': p['question'], 'reference': p['reference'],
                         'variant': p['audit_role'], 'view': i})
    assert len(rows) == protocol['events'] == 1375
    assert Counter(r['role'] for r in rows) == {'NATIVE':150, 'FIT':600, 'T2G':571, 'T2L':54}
    routes = Counter()
    for row in rows:
        item = bank_rows[row['query']]
        q, route = item['query'], item['route']
        assert q['question'] == row['question'] == item['judge_input']['question'], ('question', row['role'])
        # Bank owner zero marks a global query. The frozen roles bind its edit.
        assert item['owner'] in (0, row['owner']), ('owner', row['role'])
        assert item['judge_input']['image_sha256'] == q['image_sha256']
        assert route['activated'] and route['logical_edit_id'] == item['effective_expert']
        row['image'] = str(resolve_image(q['image_path']))
        row['image_sha256'] = q['image_sha256']
        row['selected'] = route['logical_edit_id']
        row['prompt_ids'] = item['judge_input']['prompt_ids']
        assert row['selected'] in task_by_id
        routes[row['role'], row['selected'] == task_by_owner[row['owner']]['edit_id']] += 1
        if row['role'] in ('NATIVE', 'FIT', 'T2G', 'T2L'):
            assert q['image_sha256'] == reference_rows[row['owner']][0]['image_sha256'], ('same image', row['role'])
    write(root / 'admission.json', {'events': len(rows), 'roles': dict(Counter(r['role'] for r in rows)),
                                     'route_counts': {str(k): v for k,v in routes.items()}})
    print('INPUT_BINDINGS_PASS', flush=True)
    if os.environ.get('PREFLIGHT_ONLY') == '1':
        return

    assert torch.cuda.device_count() == 2
    for i, limit in enumerate(protocol['gpu_limit_bytes']):
        torch.cuda.set_per_process_memory_fraction(
            limit / torch.cuda.get_device_properties(i).total_memory, i
        )

    assert_official_llavamed_source()
    original = LlavaMistralForCausalLM.from_pretrained
    attribute = LlavaMistralForCausalLM.__dict__.get('from_pretrained')

    @classmethod
    def load_on_cpu(cls, name, *args, **kwargs):
        kwargs.update(low_cpu_mem_usage=True, device_map={'': 'cpu'})
        return original(name, *args, **kwargs)

    LlavaMistralForCausalLM.from_pretrained = load_on_cpu
    try:
        runtime = load_runtime(parent / 'cpu_gate', 'cuda:0')
    finally:
        if attribute is None:
            del LlavaMistralForCausalLM.from_pretrained
        else:
            LlavaMistralForCausalLM.from_pretrained = attribute
    placement = {}
    for name, module in runtime.model.named_children():
        if name != 'model':
            placement[name] = 0
            continue
        for key, child in module.named_children():
            if key == 'layers':
                assert len(child) == 32
                placement.update({f'model.layers.{i}': int(i >= 15) for i in range(32)})
            else:
                placement[f'model.{key}'] = 0
    runtime.adapter.model = dispatch_model(runtime.model, device_map=placement, force_hooks=True)
    runtime.model.eval().float()
    runtime.run_root = root
    assert not any(p.requires_grad for p in runtime.model.parameters())
    layer = runtime.get_module('model.layers.21.mlp.down_proj')
    prefill = (runtime.model.model.embed_tokens, runtime.model.model.mm_projector,
               runtime.model.get_vision_tower())
    eos = runtime.adapter.tokenizer.eos_token_id
    num_forwards = 0

    def capture(_module, args):
        current.append(args[0][0, position].detach().float())

    tap = layer.register_forward_pre_hook(capture)
    print('MODEL_READY', flush=True)
    rows.sort(key=lambda v: (v['owner'], v['role'] != 'NATIVE', v.get('view', 9), v['role'], v['query']))
    try:
        with (root / 'metrics.jsonl').open('x', buffering=1) as stream, torch.no_grad():
            for index, row in enumerate(rows):
                if time.time() >= protocol['deadline'] or num_forwards >= protocol['forward_cap']:
                    raise RuntimeError('frozen resource cap reached')
                task = task_by_owner[row['owner']]
                native = task['native']
                seed_everything(task['seed'])
                rec = EditorRecord(task['edit_id'], native['dataset'], row['question'], row['reference'],
                                   task['fit_questions'][0], Path(row['image']), native['original_image_path'],
                                   task['order'], 'FROZEN_REFERENCE_TEACHER_PREFIX')
                for module in prefill:
                    module.half()
                try:
                    batch = runtime.build_edit_batch(rec)
                finally:
                    for module in prefill:
                        module.float()
                assert batch.image_sha256 == row['image_sha256'] and batch.inputs_embeds.dtype == torch.float16
                prefix = batch.raw_input_ids[0][batch.raw_labels[0] == -100].cpu()
                assert torch.equal(prefix, torch.tensor(row['prompt_ids']).reshape(-1))
                kwargs = batch.forward_kwargs()
                decode = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                          for k,v in kwargs.items() if k != 'labels'}
                labels = batch.labels[0, 1:]
                mask = labels != -100
                answer_mask = mask & (labels != eos)
                assert bool(answer_mask.any())
                position = batch.key_token_index
                assert position == int(torch.where(mask)[0][0])
                current = []
                answer_ids = labels[answer_mask]
                result = {}
                for arm in ('Base', 'MED', 'MDU'):
                    hook = None
                    if arm != 'Base':
                        selected = row['selected']
                        expert = TT4(task_by_id[selected]['seed'], 8, 8)
                        expert.load_state_dict(state[arm][selected], strict=True)
                        expert.to(layer.weight.device).eval().requires_grad_(False)
                        hook = MedTraceLayerHook(layer, expert)
                        hook.attach()
                        hook.set_teacher_routing(batch.labels)
                    try:
                        out = runtime.model(**decode)
                        num_forwards += 1
                        logits = out.logits[0, :-1].float()
                        active = logits[mask.to(logits.device)]
                        targets = labels[mask].to(active.device)
                        logp = active.log_softmax(-1)
                        loss = -logp.gather(1, targets[:,None]).squeeze(1)
                        token_loss = loss[targets != eos]
                        info = {'nll':float(token_loss.mean()), 'first_nll':float(token_loss[0]),
                                'ce_with_eos':float(loss.mean())}
                        if arm == 'Base':
                            baseline = logp[targets != eos]
                        else:
                            b,a = expert.factors()
                            qa = torch.linalg.qr(a.T).Q
                            x = current[-1]
                            x = x / (x.square().mean().sqrt() + expert.epsilon)
                            info['projection'] = float(torch.linalg.vector_norm(x @ qa))
                            info['residual'] = float(torch.linalg.vector_norm((x@a.T)@b.T))
                            kl = (baseline.exp() * (baseline-logp[targets != eos])).sum(-1).mean()
                            assert float(kl) >= -2e-6
                            info['kl'] = max(0.0, float(kl))
                        result[arm] = info
                        del out, logits, active, logp
                    finally:
                        if hook is not None:
                            hook.detach()
                assert len(current) == 3 and all(torch.equal(current[0], x) for x in current[1:])
                if row.get('view') in (0,1) and row['selected'] == task['edit_id']:
                    assert math.isclose(result['MED']['ce_with_eos'], first_step[row['owner']][row['view']],
                                        rel_tol=0, abs_tol=5e-4), 'historical source CE parity'
                data = {k:row[k] for k in ('owner','role','query','variant','selected')}
                data.update(tokens=int(answer_ids.numel()), audit=row.get('audit'), status=row.get('status'),
                            routed_to_owner=row['selected']==task['edit_id'], arms=result)
                print(json.dumps(data, allow_nan=False), file=stream)
                if index < 5 or (index+1) % 25 == 0:
                    write(root / 'progress.json', {'events':index+1, 'expected':len(rows),
                                                     'forwards':num_forwards})
    finally:
        tap.remove()
    assert num_forwards == protocol['forward_cap']
    write(root/'done.json', {'events':len(rows), 'forwards':num_forwards,
                             'training':0, 'generation':0, 'score':0})


if __name__ == '__main__':
    main()
