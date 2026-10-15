"""Opt-in, one-correction reuse; never installed by a production entry point."""
import copy
from contextlib import contextmanager
import torch


def equal_state(a, b):
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and a[k].device == b[k].device
        and a[k].shape == b[k].shape
        and torch.equal(a[k].detach().contiguous().reshape(-1).view(torch.uint8),
                        b[k].detach().contiguous().reshape(-1).view(torch.uint8)) for k in a)


def snapshot(state):
    return {k: v.detach().clone() for k, v in state.items()}


@contextmanager
def correction_reuse(original, pure):
    """Caller must own the worker exclusively; globals are restored on failure."""
    p, g, r, c = original.p, original.g, original.r, original.c
    old_losses, old_canonical = r.losses_at, r.mm.canonicalize
    old_rows, old_penalty = g.gm.root_rows, r.mm.operator_penalty
    canonical, rows, penalties = [], [], {}
    pure_values, chosen_ids = None, None
    pure_versions = tuple((id(v), v._version) for v in pure.values())
    stats = dict(pure_loss_hits=0, canonical_hits=0, root_rows_hits=0, penalty_hits=0)

    def losses(runtime, hook, expert, state, batches):
        nonlocal pure_values, chosen_ids
        if state is not pure:
            return old_losses(runtime, hook, expert, state, batches)
        assert tuple((id(v), v._version) for v in pure.values()) == pure_versions
        identity = tuple(id(b) for b in batches)
        if pure_values is None:
            before = c.rng()
            pure_values = old_losses(runtime, hook, expert, state, batches)
            after = c.rng()
            assert before['python_rng'] == after['python_rng']
            assert all((before[k] is None and after[k] is None) or
                       torch.equal(before[k], after[k]) for k in ('torch_rng', 'cuda_rng'))
            chosen_ids = identity
        else:
            assert identity == chosen_ids and not runtime.model.training
            # Preserve losses_at's state load and final teacher-routing side effects.
            expert.load_state_dict(state)
            for batch in batches:
                hook.set_teacher_routing(batch.labels)
            stats['pure_loss_hits'] += 1
        return list(pure_values)

    def canonicalize(state):
        for saved, result, pivots in canonical:
            if equal_state(state, saved):
                stats['canonical_hits'] += 1
                return snapshot(result), copy.deepcopy(pivots)
        result, pivots = old_canonical(state)
        canonical.append((snapshot(state), snapshot(result), copy.deepcopy(pivots)))
        return result, pivots

    def root_rows(state, cache):
        binding = (id(cache), tuple((id(cache[k]), cache[k]._version) for k in ('x', 'y')),
                   tuple(cache['token_counts']))
        for saved, previous_binding, result in rows:
            if binding == previous_binding and equal_state(state, saved):
                stats['root_rows_hits'] += 1
                return result.clone()
        result = old_rows(state, cache)
        rows.append((snapshot(state), binding, result.detach().clone()))
        return result

    def penalty(state, cache):
        # Autograd's live Parameter penalty must always retain its original graph.
        if any(v.requires_grad for v in state.values()):
            return old_penalty(state, cache)
        key = (id(state), id(cache), tuple((id(v), v._version) for v in state.values()),
               tuple((id(cache[k]), cache[k]._version) for k in ('x', 'y')), tuple(cache['token_counts']))
        if key in penalties:
            stats['penalty_hits'] += 1
            return penalties[key][2].clone()
        result = old_penalty(state, cache)
        penalties[key] = (tuple(state.values()), cache, result.detach().clone())
        return result

    r.losses_at, r.mm.canonicalize = losses, canonicalize
    g.gm.root_rows, r.mm.operator_penalty = root_rows, penalty
    try:
        yield stats
    finally:
        r.losses_at, r.mm.canonicalize = old_losses, old_canonical
        g.gm.root_rows, r.mm.operator_penalty = old_rows, old_penalty
