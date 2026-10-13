"""Invertible TT coordinate changes and source-only diagnostic measurements."""
import torch
from scoped_math import factors
from mechanism_math import map_delta, lowrank_inner

MODES = ('IDENTITY', 'PERMUTATION', 'DIAGONAL')


def gauge(state, mode):
    assert mode in MODES
    out = {k: v.detach().clone() for k, v in state.items()}
    for i in range(1, 4):
        left, right = f'G{i}', f'G{i+1}'
        rank = out[left].shape[-1]
        if mode == 'PERMUTATION':
            ix = torch.arange(rank-1, -1, -1, device=out[left].device)
            out[left] = out[left].index_select(-1, ix)
            out[right] = out[right].index_select(0, ix)
        elif mode == 'DIAGONAL':
            scale = torch.tensor([.5, 2.]*(rank//2), device=out[left].device, dtype=out[left].dtype)
            out[left] = out[left]*scale
            out[right] = out[right]/scale[:, None, None]
    return out


def delta(before, after):
    return map_delta({k:v.detach().cpu() for k,v in before.items()},
                     {k:v.detach().cpu() for k,v in after.items()})


def compare_maps(value, reference):
    norm = max(0., lowrank_inner(*value, *value))**.5
    refnorm = max(0., lowrank_inner(*reference, *reference))**.5
    inner = lowrank_inner(*value, *reference)
    return dict(norm=norm, reference_norm=refnorm,
                norm_ratio=norm/refnorm if refnorm else None,
                cosine=inner/(norm*refnorm) if norm*refnorm else None,
                difference_norm=max(0., norm*norm+refnorm*refnorm-2*inner)**.5)


def metrics(reference, candidate, target):
    assert reference.shape == candidate.shape and len(target) == len(reference)
    lp, lq = reference.double().log_softmax(-1), candidate.double().log_softmax(-1)
    assert torch.isfinite(lq).all()
    ix = torch.arange(len(target)); correct = lq.argmax(-1) == target
    other = lq[0].clone(); other[target[0]] = -torch.inf
    kl = (lp.exp()*(lp-lq)).sum(-1)
    assert float(kl.min()) >= -1e-10
    return dict(tokens=len(target), NLL=float(-lq[ix,target].mean()),
                delta_NLL=float((lp[ix,target]-lq[ix,target]).mean()),
                first_target_logprob=float(lq[0,target[0]]),
                first_target_margin=float(lq[0,target[0]]-other.max()),
                first_delta_NLL=float(lp[0,target[0]]-lq[0,target[0]]),
                first_KL=max(0.,float(kl[0])), mean_KL=max(0.,float(kl.mean())),
                first_argmax_changed=bool(lp[0].argmax()!=lq[0].argmax()),
                first_target_top1=bool(correct[0]), all_target_top1=bool(correct.all()),
                changed_tokens=int((lp.argmax(-1)!=lq.argmax(-1)).sum()),
                maximum_logit_difference=float((candidate-reference).abs().max()))


def visual_input(value, span, mode, donor=None):
    kwargs, labels, tokens = value
    kwargs = dict(kwargs); x = kwargs['inputs_embeds'].clone(); a,b = span
    assert 0 <= a < b <= x.shape[1] and b-a == 576
    if mode == 'ZERO_VISUAL': x[:,a:b] = 0
    else:
        assert mode == 'SWAPPED_VISUAL' and donor.shape == x[:,a:b].shape
        x[:,a:b] = donor
    kwargs['inputs_embeds'] = x
    return kwargs, labels, tokens


def selfcheck():
    from structures import TT4
    torch.manual_seed(20261011)
    e = TT4(19,8,8).double()
    x = torch.randn(3,14336,dtype=torch.float64)
    target = torch.randn(3,4096,dtype=torch.float64)
    for zero in (True,False):
        if not zero:
            with torch.no_grad(): e.G1.normal_(0,.02)
        state = {k:v.detach().clone() for k,v in e.state_dict().items()}
        y = e.residual(x).detach(); gradients={}
        for mode in MODES:
            e.load_state_dict(gauge(state,mode)); e.zero_grad()
            torch.testing.assert_close(e.residual(x),y,rtol=1e-11,atol=1e-11)
            (e.residual(x)-target).square().mean().backward()
            gradients[mode] = {k:v.grad.detach().clone() for k,v in e.named_parameters()}
        # Permutations preserve the Euclidean gradient; diagonal gauges do not.
        for k,v in gauge(gradients['IDENTITY'],'PERMUTATION').items():
            torch.testing.assert_close(v,gradients['PERMUTATION'][k],rtol=1e-10,atol=1e-12)
        assert any(not torch.allclose(gradients['IDENTITY'][k],gradients['DIAGONAL'][k]) for k in state)
        e.load_state_dict(state)
    a=torch.randn(2,5,dtype=torch.float64); t=torch.tensor([1,2])
    same=metrics(a,a,t); assert same['first_KL']==0 and same['delta_NLL']==0 and not same['first_argmax_changed']
    emb=torch.ones(1,580,3); value=({'inputs_embeds':emb,'attention_mask':torch.ones(1,580)},t,(1,2))
    changed=visual_input(value,(2,578),'ZERO_VISUAL')[0]['inputs_embeds']
    assert torch.equal(changed[:,:2],emb[:,:2]) and torch.equal(changed[:,578:],emb[:,578:])
    assert changed[:,2:578].count_nonzero()==0 and emb.count_nonzero()==emb.numel()
    return dict(status='PASS',TT88_function_equivalence=True,permutation_gradient_equivariance=True,
                diagonal_changes_gradient=True,metrics_identity=True,visual_span_only=True)
