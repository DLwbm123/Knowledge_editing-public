"""CPU trust-boundary checks for the opt-in correction cache."""
import random
from types import SimpleNamespace as NS
import torch
from equivalent_reuse import correction_reuse, equal_state


def main():
    assert not equal_state({'x': torch.tensor([0.])}, {'x': torch.tensor([-0.])})
    calls = dict(loss=0, canonical=0, rows=0, penalty=0)
    expert = torch.nn.Module()
    expert.register_parameter('w', torch.nn.Parameter(torch.ones(1, 1)))
    pure = {'w': torch.tensor([[2.]])}
    cache = dict(x=torch.ones(1, 1), y=torch.ones(1, 1), token_counts=[1])
    batch = NS(labels=torch.tensor([[1]]))
    hook = NS(set_teacher_routing=lambda labels: None)
    runtime = NS(model=expert.eval())
    def rng(): return dict(python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=None)
    def losses(*args):
        calls['loss'] += 1
        return [float(args[3]['w'].square().sum())]
    def canonical(state):
        calls['canonical'] += 1
        return {k: v.clone() for k, v in state.items()}, [[0]]
    def rows(state, cache):
        calls['rows'] += 1
        return state['w'] @ cache['x']
    def penalty(state, cache):
        calls['penalty'] += 1
        return (state['w'] @ cache['x']).square().sum()
    mm = NS(canonicalize=canonical, operator_penalty=penalty)
    gm = NS(root_rows=rows)
    r = NS(losses_at=losses, mm=mm)
    original = NS(p=None, g=NS(gm=gm), r=r, c=NS(rng=rng))
    with correction_reuse(original, pure) as hits:
        assert r.losses_at(runtime, hook, expert, pure, [batch]) == [4.]
        assert r.losses_at(runtime, hook, expert, pure, [batch]) == [4.]
        a, _ = mm.canonicalize(pure); a['w'].zero_()
        b, _ = mm.canonicalize(pure); assert b['w'].item() == 2.
        gm.root_rows(pure, cache); assert gm.root_rows(pure, cache).item() == 2.
        cache['x'].add_(1); assert gm.root_rows(pure, cache).item() == 4.
        mm.operator_penalty(pure, cache); assert mm.operator_penalty(pure, cache).item() == 16.
        parameter = torch.nn.Parameter(torch.tensor([[2.]]))
        for _ in range(2):
            assert torch.autograd.grad(mm.operator_penalty({'w': parameter}, cache), parameter)[0].item() == 16.
        pure['w'].add_(1)
        try: r.losses_at(runtime, hook, expert, pure, [batch])
        except AssertionError: pass
        else: raise AssertionError('Mutated pure state reused')
    assert hits == dict(pure_loss_hits=1, canonical_hits=1, root_rows_hits=1, penalty_hits=1)
    assert r.losses_at is losses and mm.canonicalize is canonical and gm.root_rows is rows and mm.operator_penalty is penalty
    try:
        with correction_reuse(original, pure): raise RuntimeError('fixture')
    except RuntimeError: pass
    assert r.losses_at is losses and mm.operator_penalty is penalty
    print(dict(status='PASS', mutation_rejected=True, autograd_not_cached=True, exception_restoration=True, hits=hits, calls=calls))


if __name__ == '__main__': main()
