"""Measure TT gradient scales and disposable optimizer throughput on training-side data."""
import json
import statistics
import time

import torch
import torch.nn.functional as F


def gradient_preflight(root, plan, lane, items, runtime, layer, med, mdu, tt,
                       hook_cls, prepare, record, seed):
    eos = runtime.adapter.tokenizer.eos_token_id
    device = layer.weight.device
    torch.cuda.reset_peak_memory_stats()
    rows = []
    timings = []
    updates = 0
    calibration = plan.get('kl_calibration', False)

    def logits(batch):
        labels = batch['labels']
        args = {k: v for k, v in batch.items() if k != 'labels'}
        mask = labels[0, 1:] != -100
        scores = runtime.model(**args).logits[0, :-1][mask].float()
        return scores, labels[0, 1:][mask]

    for index, item in enumerate(items):
        assert time.time() < plan['deadline'] - 30
        source = item['source']
        edit = source['edit_id']
        seed(source['seed'])
        image = source['source_rows'][0]
        row = dict(source=source, image=image['image_path'], image_sha=image['image_sha256'],
                   owner=item['owner'])

        def batch(view, answer, tokens=None):
            q = record(dict(row, question=view['question']), answer)
            with torch.no_grad():
                return prepare(row, q, tokens)

        native = batch(image, image['reference'])
        def negative_batches(role):
            result = []
            for entry in item['negative'][role][:2]:
                assert entry['label'] == 'independent' and entry['role'] == role
                ids = entry['base_continuation']['raw_token_ids']
                b = batch(entry, entry['gold_answer'], ids)
                with torch.no_grad():
                    score, target = logits(b)
                    active = target != eos
                    assert active.any()
                    log_p = score[active].log_softmax(-1)
                    result.append((b, active, log_p, log_p.exp()))
            assert result
            return result

        neg = negative_batches('train')
        validation_neg = negative_batches('holdout') if calibration else neg

        def make_expert(state):
            expert = tt(source['seed'], 8, 8).to(device)
            expert.load_state_dict(state)
            return expert

        expert = make_expert(med[edit])
        params = tuple(expert.parameters())
        assert sum(v.numel() for v in params) == 7168
        hook = hook_cls(layer, expert)
        hook.attach()

        def edited(b):
            hook.set_teacher_routing(b['labels'])
            return logits(b)

        def positive(b, with_eos=False):
            score, target = edited(b)
            losses = F.cross_entropy(score, target, reduction='none')
            if with_eos:
                return losses.mean()
            losses = losses[target != eos]
            assert losses.numel()
            if losses.numel() == 1:
                return losses.mean()
            first, second = losses.tensor_split([losses.numel() // 2])
            return (first.mean() + second.mean()) / 2

        def locality(b, active, log_p, p):
            score, _ = edited(b)
            log_q = score[active].log_softmax(-1)
            return (p * (log_p - log_q)).sum(-1).mean()

        def grad_norm(loss):
            gradients = torch.autograd.grad(loss, params)
            norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
            assert torch.isfinite(norm)
            return float(norm)

        def optimize(views, beta, lr_a, steps, kind, log_timing=False):
            nonlocal updates
            expert.load_state_dict(med[edit])
            a, b = expert.parameter_groups()
            optimizer = torch.optim.Adam([dict(params=a, lr=lr_a), dict(params=b, lr=0.001)])
            for step in range(steps):
                assert time.time() < plan['deadline'] - 30
                torch.cuda.synchronize()
                before = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                positive(native, with_eos=True).backward()
                for view in views[step % 4:] + views[:step % 4]:
                    (positive(view) / 4).backward()
                for n in neg:
                    (beta * locality(*n) / len(neg)).backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                updates += 1
                assert float(norm) > 0
                if log_timing:
                    timings.append(dict(views=kind, step=step, seconds=time.monotonic() - before,
                                        negatives=len(neg)))
            assert any(not torch.equal(val.detach().cpu(), med[edit][name])
                       for name, val in expert.named_parameters())

        try:
            with torch.no_grad():
                med_kl = [float(locality(*n)) for n in validation_neg]
                expert.load_state_dict(mdu[edit])
                mdu_kl = [float(locality(*n)) for n in validation_neg]
                expert.load_state_dict(med[edit])
            result = dict(med_kl=med_kl, mdu_kl=mdu_kl)
            if not calibration:
                dev = [batch(v, v['reference']) for v in item['views'] if v['role'] == 'validation']
                assert len(dev) == 3
                result['ce_grad'] = [grad_norm(positive(b)) for b in dev]
                result['kl_grad'] = [grad_norm(locality(*n)) for n in neg]
            if calibration or index < plan['timed_owners_per_lane']:
                groups = [batch(v, v['reference']) for v in item['views'] if v['role'] == 'train']
                fits = [batch(v, v['reference']) for v in source['source_rows'][1:]]
                assert len(groups) == len(fits) == 4
                if calibration:
                    result['candidates'] = {}
                    for factor in (0.3, 1., 3.):
                        beta = plan['beta0'] * factor
                        arms = {}
                        for arm, views, lr_a in (('P1', fits, 0.0001), ('P2', fits, 0.001),
                                                 ('P3', groups, 0.0001), ('P4', groups, 0.001)):
                            optimize(views, beta, lr_a, plan['calibration_steps'], arm)
                            with torch.no_grad():
                                arms[arm] = [float(locality(*n)) for n in validation_neg]
                        result['candidates'][str(factor)] = arms
                        print('GRID', index + 1, factor, flush=True)
                else:
                    beta = statistics.median(result['ce_grad']) / statistics.median(result['kl_grad'])
                    assert beta > 0
                    for kind, views in (('original', fits), ('expanded', groups)):
                        optimize(views, beta, 0.001, plan['timed_steps'], kind, log_timing=True)
            rows.append(result)
        finally:
            hook.detach()
        print('PROGRESS', index + 1, len(items), flush=True)

    payload = dict(owners=len(rows), gradients=rows, timings=timings,
                   disposable_updates=updates, generation=0, judge=0,
                   cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                   cuda_peak_reserved=torch.cuda.max_memory_reserved())
    (root / f'done{lane}.json').write_text(json.dumps(payload, allow_nan=False))
