"""Train and measure ordinary CE TT experts on admitted Qwen source errors."""
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from methods.medtrace.core import MedTraceLayerHook
from qwen_expert import QwenExpert


def main(root):
    plan = json.loads((root / 'protocol.json').read_text())
    lane = int(os.environ['Q_LANE'])
    preflight = plan.get('preflight', False)
    steps = 4 if preflight else 160
    all_entries = json.loads((root / 'panel.json').read_text())
    entries = all_entries[lane::2]
    assert all(r['base_core'] is False and r['base_eos'] for r in entries)
    assert len(entries) and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(plan['limit_bytes'][lane] /
                                               torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(1)
    device = 'cuda:0'
    model_path = os.environ['QWEN_MODEL_CHECK']
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True, device_map={'': device})
    model.eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28,
                                             max_pixels=1280 * 28 * 28, local_files_only=True)
    layer = model.model.layers[18].mlp.down_proj
    assert (layer.in_features, layer.out_features) == (18944, 3584)
    eos = processor.tokenizer.eos_token_id
    counts = dict(updates=0, forwards=0, generations=0, events=0)

    def budget():
        if time.time() >= plan['deadline'] - 60:
            raise TimeoutError('Fixed mechanism deadline')

    def encode(image, question):
        message = [dict(role='user', content=[dict(type='image', image=image),
                                             dict(type='text', text=question)])]
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return processor(text=[prompt], images=[image], return_tensors='pt').to(device)

    @torch.no_grad()
    def teacher_inputs(raw, answer):
        token_ids = processor.tokenizer(answer, add_special_tokens=False).input_ids + [eos]
        target = torch.tensor([token_ids], dtype=torch.long, device=device)
        assert target.numel() > 1
        prompt = raw['input_ids']
        full = torch.cat((prompt, target), dim=-1)
        valid = torch.ones_like(full)
        grid = raw['image_grid_thw']
        positions, _ = model.get_rope_index(full, image_grid_thw=grid, attention_mask=valid)
        embeds = model.model.embed_tokens(full)
        pixels = raw['pixel_values'].to(model.visual.dtype)
        vision = model.visual(pixels, grid_thw=grid).to(embeds.dtype)
        mask = full == model.config.image_token_id
        assert vision.size(0) == int(mask.sum())
        embeds[mask] = vision
        labels = torch.cat((torch.full_like(prompt, -100), target), dim=-1)
        return full.size(-1) - target.size(-1), target, labels, dict(
            inputs_embeds=embeds, attention_mask=valid, position_ids=positions, use_cache=False,
            return_dict=True)

    def teacher(batch, hook=None):
        budget()
        pos, target, labels, kwargs = batch
        if hook is not None:
            hook.set_teacher_routing(labels)
        hidden = model.model(**kwargs).last_hidden_state[:, pos - 1:-1]
        assert hidden.shape[:2] == target.shape
        counts['forwards'] += 1
        return model.lm_head(hidden).float()

    def greedy(raw, hook=None):
        budget()
        with (hook.generation_request() if hook else nullcontext()):
            out = model.generate(**raw, do_sample=False, num_beams=1, max_new_tokens=1024,
                                 repetition_penalty=1.05, temperature=None, top_p=None, top_k=None,
                                 use_cache=True)
        continuation = out[0, raw['input_ids'].size(-1):].tolist()
        stops = model.generation_config.eos_token_id
        stops = [stops] if isinstance(stops, int) else stops
        counts['generations'] += 1
        return dict(answer=processor.decode(continuation, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False),
                    tokens=continuation, eos=bool(continuation and continuation[-1] in stops),
                    at_cap=len(continuation) >= 1024)

    print('MODEL_READY', flush=True)
    if not preflight:
        (root / 'weights').mkdir(exist_ok=True)
    with (root / f'metrics{lane}.jsonl').open('x', buffering=1) as out:
        for entry in entries:
            budget()
            source = entry['source']
            owner, seed = entry['owner'], source['seed']
            torch.manual_seed(seed)
            image_file = source['source_rows'][0]['image_path']
            path = Path(image_file)
            if not path.is_file():
                path = Path(os.environ['FROZEN_IMAGE_ROOT']) / image_file.split('/DataP/', 1)[1]
            with Image.open(path) as pixels:
                image = pixels.convert('RGB')
            refs = source['source_rows']
            assert [r['audit_role'] for r in refs] == ['NATIVE', 'FIT', 'FIT', 'FIT', 'FIT']
            raw = [encode(image, r['question']) for r in refs]
            train = [teacher_inputs(r, b['reference']) for r, b in zip(raw, refs, strict=True)]
            expert = QwenExpert(seed).to(device)
            with torch.no_grad():
                pos, _, _, _ = train[0]
                plain = model(**raw[0], use_cache=False).logits[:, pos - 1].float()
                reconstructed = teacher(train[0])[:, 0]
                assert torch.allclose(plain, reconstructed, rtol=0.002, atol=0.02)
            a, b = expert.parameter_groups()
            opt = torch.optim.Adam([dict(params=a, lr=1e-4), dict(params=b, lr=1e-3)])
            hook = MedTraceLayerHook(layer, expert)
            hook.attach()
            try:
                for step in range(steps):
                    opt.zero_grad(set_to_none=True)
                    for idx in (0, 1 + step % 4):
                        logits = teacher(train[idx], hook)
                        (0.5 * F.cross_entropy(logits[0], train[idx][1][0])).backward()
                    torch.nn.utils.clip_grad_norm_(expert.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    counts['updates'] += 1
                    if step in (0, 39, 79, 119, 159):
                        print('STEP', counts['events'], step + 1, flush=True)
                state = {k: v.detach().cpu().clone() for k, v in expert.state_dict().items()}
                assert all(torch.isfinite(v).all() for v in state.values())
                assert torch.count_nonzero(state['G1']) > 0
                if preflight:
                    continue
                with (root / 'weights' / f'{owner}.pt').open('xb') as f:
                    torch.save(dict(state=state, seed=seed, steps=160), f)
                positives = [v for v in entry['views'] if v['role'].startswith('holdout_')]
                negatives = [dict(question=v['question'], reference=v['gold_answer'],
                                  role='independent') for v in entry['negative']['holdout']]
                network_rows = [dict(v, role=v['audit_role'].lower()) for v in refs]
                network_rows.extend(positives + negatives + entry.get('official_t2g', []))
                measures = []
                for row in network_rows:
                    with torch.no_grad():
                        inputs = encode(image, row['question'])
                        batch = teacher_inputs(inputs, row['reference'])
                        pos, target, _, _ = batch
                        inputs_at_layer = []
                        hook.clear_request_routing()
                        base = teacher(batch).log_softmax(-1)
                        monitor = layer.register_forward_pre_hook(
                            lambda _layer, args: inputs_at_layer.append(args[0][0, pos - 1].float()))
                        try:
                            changed = teacher(batch, hook).log_softmax(-1)
                        finally:
                            monitor.remove()
                        x = inputs_at_layer[0]
                        x = x / (x.square().mean().sqrt() + expert.epsilon)
                        b, a = expert.factors()
                        active = target[0] != eos
                        assert active.any()
                        kl = (base.exp() * (base - changed)).sum(-1)[0, active].mean()
                        tokens = target.unsqueeze(-1)
                        reference_nll = -changed.gather(-1, tokens)[0, active, 0]
                        metric = dict(role=row['role'], question=row['question'], reference=row['reference'],
                                      projection=float((x @ torch.linalg.qr(a.T).Q).norm()),
                                      residual=float((x @ a.T @ b.T).norm()),
                                      kl=max(0., float(kl)), first_nll=float(reference_nll[0]),
                                      nll=float(reference_nll.mean()))
                        assert all(math.isfinite(v) for v in metric.values() if isinstance(v, float))
                        if row['role'] != 'fit':
                            hook.clear_request_routing()
                            metric['answer'] = greedy(inputs, hook)
                        measures.append(metric)
                    counts['events'] += 1
                print(json.dumps(dict(owner=owner, measurements=measures), ensure_ascii=False,
                                 allow_nan=False), file=out)
            finally:
                hook.detach()
    assert counts['updates'] == len(entries) * steps
    (root / f'done{lane}.json').write_text(json.dumps(dict(counts, owners=len(entries),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())))


if __name__ == '__main__':
    main(Path.cwd())
