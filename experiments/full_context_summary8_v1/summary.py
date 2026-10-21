"""Current-source activations and per-token VJPs; no optimizer or historical input."""
import time
import torch
from experiments.fixed_input_guard8_v1.adapter import context_positions
from experiments.fixed_input_guard_v1.guard import normalized_rows, right_basis
from .guard import compress

MAX_CONTEXT, MAX_TARGET = 4096, 1024
VJP_SCALE = 128


def layout(batch):
    kwargs, labels, (start,end) = batch
    valid = kwargs.get('attention_mask', torch.ones_like(labels))[0].bool().cpu()
    if len(valid) > MAX_CONTEXT or not 0 < int((labels[:,1:] != -100).sum()) <= MAX_TARGET:
        raise ValueError('Frozen summary context/token bound exceeded')
    strata = torch.full((len(valid),), 2, dtype=torch.int64)
    answer = ((labels[0,1:] != -100) & valid[1:] & valid[:-1]).nonzero().flatten()
    strata[answer] = 1
    strata[start:end] = 0
    return valid, strata


def collect(adapter, batch, sensitive):
    valid, strata = layout(batch)
    captured = []
    def hook(module, args, output):
        h = args[0]
        if h.ndim != 3 or h.shape[:2] != batch[1].shape or output.shape[:2] != h.shape[:2]:
            raise ValueError('Summary expanded positions mismatch')
        a = output.detach().requires_grad_(sensitive)
        captured.append((h[0].detach().float().cpu(), a))
        return a
    handle = adapter.layer.register_forward_hook(hook)
    enabled = adapter.weight.requires_grad
    adapter.weight.requires_grad_(False)
    started = time.monotonic()
    if adapter.weight.is_cuda:
        torch.cuda.reset_peak_memory_stats(adapter.weight.device)
    try:
        adapter.budget.consume('summary_forwards')
        with torch.set_grad_enabled(sensitive):
            z,y = adapter.logits(batch)
            if len(captured) != 1 or not torch.isfinite(captured[0][0]).all():
                raise ValueError('Invalid complete summary capture')
            h,a = captured[0]
            other = z.detach().clone(); other.scatter_(1,y[:,None],-torch.inf)
            competitors = other.argmax(1)
            margin = z.gather(1,y[:,None]).squeeze(1)-z.gather(1,competitors[:,None]).squeeze(1)
            energy = torch.zeros(len(h), dtype=torch.float64)
            nonzero_gradients = []
            if sensitive:
                for t in range(len(y)):
                    adapter.budget.consume('summary_vjps')
                    g = torch.autograd.grad(margin[t]*VJP_SCALE, a, retain_graph=t+1<len(y))[0]
                    if not torch.isfinite(g).all():
                        raise ValueError('Nonfinite native margin VJP')
                    g = g[0].detach().double()/VJP_SCALE
                    energy += g.square().sum(1).cpu()/len(y)
                    nonzero_gradients.append(int(g.count_nonzero()))
            metrics = dict(supervised_tokens=len(y), margins=margin.detach().double().cpu().tolist(),
                gradient_nonzero_elements=nonzero_gradients, vjps=len(y) if sensitive else 0,
                elapsed_seconds=time.monotonic()-started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(adapter.weight.device) if adapter.weight.is_cuda else 0,
                vjp_scale=VJP_SCALE, sensitive=sensitive)
        return h,energy,valid,strata,metrics
    finally:
        handle.remove()
        adapter.weight.requires_grad_(enabled)


def weighted_rows(h, energy, valid, strata, sampled, arm, seed):
    if arm not in ('B','C','N','D','E') or not torch.isfinite(h).all() or not torch.isfinite(energy).all() or (energy < 0).any():
        raise ValueError('Invalid weighting inputs')
    unit = normalized_rows(h)
    nonzero = h.abs().amax(1) > 0
    ids = (valid & nonzero).nonzero().flatten()
    reference = unit[sampled].double().square().sum()
    info = dict(valid_positions=int(valid.sum()), nonzero_positions=len(ids), sampled_positions=len(sampled),
                sampled_nonzero=int(nonzero[sampled].sum()), reference_energy=float(reference),
                zero_positions=int((valid & ~nonzero).sum()), zero_reference=bool(reference == 0),
                sensitive_fallback=False, singleton_strata=0)
    if arm == 'B':
        rows = unit[sampled]
    elif not len(ids):
        rows = unit[:0]
    else:
        # Scaling before squaring preserves ratios without overflow or underflow.
        norm2 = (h[ids].double()/h[ids].double().abs().max()).square().sum(1)
        score = norm2 if arm == 'N' else norm2*energy[ids]
        p = torch.full((len(ids),), 1/len(ids), dtype=torch.float64)
        if arm != 'C' and score.sum() > 0:
            p = .5*p+.5*score/score.sum()
        elif arm in ('D','E'):
            info['sensitive_fallback'] = True
        if arm == 'E':
            gen = torch.Generator(device='cpu').manual_seed(seed)
            for group in (0,1,2):
                where = (strata[ids] == group).nonzero().flatten()
                info['singleton_strata'] += int(len(where) == 1)
                p[where] = p[where[torch.randperm(len(where),generator=gen)]]
        rows = (unit[ids].double()*(reference*p).sqrt()[:,None]).float()
    info['weighted_input_energy'] = float(rows.double().square().sum())
    if abs(info['weighted_input_energy']-float(reference)) > 2e-6*max(1.,float(reference)):
        raise ValueError('Weighted input energy mismatch')
    return rows,info


@torch.no_grad()
def same_state_diagnostic(h,energy,valid,strata,sampled,fixed,seed):
    # Only the first view of current source1; never construct a historical oracle.
    if len(h) > 1024:
        return dict(status='UNAVAILABLE_PREDECLARED_1024_ROW_CAP')
    q0 = torch.linalg.qr(fixed.double(),mode='reduced').Q
    matrices = {}
    for arm in ('C','N','D','E'):
        rows,_ = weighted_rows(h,energy,valid,strata,sampled,arm,seed)
        rows = rows.double(); matrices[arm] = rows-(rows@q0)@q0.T
    common = matrices['C']
    result = {}
    for arm,m in matrices.items():
        if not len(m):
            result[arm] = dict(rows=0,rank=0,exact_best64=0.,stream_error=0.,common_uniform_error=0.)
            continue
        _,s,vh = torch.linalg.svd(m,full_matrices=False)
        q = right_basis(m,64).double()
        exact = float(s[64:].square().sum())
        error = float((m-(m@q)@q.T).square().sum())
        result[arm] = dict(rows=len(m),rank=q.shape[1],exact_best64=exact,stream_error=error,
            common_uniform_error=float((common-(common@q)@q.T).square().sum()),
            stream_compressions=0, note='At most 1024 rows: empty sketch has not triggered compression; rank extraction only')
    return dict(status='PASS',scope='CURRENT_SOURCE1_ORIGINAL_VIEW_ONLY',arms=result,
                historical_oracle_available=False,training_feedback=False)
