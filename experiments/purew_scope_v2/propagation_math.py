"""ScopeEdit equations 14--24 in editing-only coordinates; inference uses W."""
import torch


@torch.no_grad()
def make_basis(weight, rank_private=448, rank_shared=64, seed=20261010):
    from purew_math import row_basis
    q = row_basis(weight)
    rng = torch.Generator(device=weight.device).manual_seed(seed)
    samples = torch.randn(weight.shape[1], rank_private + rank_shared,
                          generator=rng, device=weight.device, dtype=weight.dtype)
    samples -= q @ (q.T @ samples)
    a = torch.linalg.qr(samples, mode='reduced')[0].T.contiguous()
    eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    orthogonal_error = float((a @ a.T - eye).abs().max())
    null_error = float((weight @ a.T).norm() / weight.norm().clamp_min(1e-12))
    if orthogonal_error > 2e-5 or null_error > 2e-5:
        raise ValueError('orthogonal/nullspace basis admission failed')
    return (a[:rank_private], a[rank_private:]), dict(
        orthogonal_max_error=orthogonal_error, relative_null_error=null_error,
        seed=seed, fixed_initial_weight_nullspace=True)


def gate(text, image, always_on=False, tau=.2, beta=10., eps=1e-6):
    if not torch.isfinite(text).all() or not torch.isfinite(image).all():
        raise ValueError('nonfinite modality evidence')
    nt, nv = text.norm(), image.norm()
    cosine = (text @ image) / (nt * nv + eps)
    support = torch.minimum(nt, nv) / (torch.maximum(nt, nv) + eps)
    valid = (nt > eps) & (nv > eps)
    gamma = torch.ones_like(cosine) if always_on else torch.sigmoid(beta * (cosine - tau)) * support
    gamma = torch.where(valid, gamma, torch.zeros_like(gamma)).clamp(0., 1.)
    return float(gamma), dict(gamma=float(gamma), cosine=float(cosine), support=float(support),
                              text_norm=float(nt), image_norm=float(nv))


@torch.no_grad()
def pool_keys(features, labels, image_span=None):
    if features.ndim != 3 or features.shape[:2] != labels.shape or labels.shape[0] != 1:
        raise ValueError('feature/causal-label shape mismatch')
    predictors = torch.zeros_like(labels, dtype=torch.bool)
    predictors[:, :-1] = labels[:, 1:] != -100
    if not predictors.any():
        raise ValueError('empty answer-predictor key')
    text = features[predictors].float().mean(0)
    if image_span is None:
        return text, None, None
    start, end = image_span
    if not 0 <= start < end <= labels.shape[1] or predictors[:, start:end].any():
        raise ValueError('image/predictor spans overlap or are invalid')
    visual = features[0, start:end].float().mean(0)
    # A fixed pooled multimodal key, including both admitted evidence modalities.
    multimodal = (text * predictors.sum() + visual * (end-start)) / (predictors.sum() + end-start)
    return text, visual, multimodal


@torch.no_grad()
def recurse(inverse, key, rho=1.):
    if rho < 0 or not torch.isfinite(key).all():
        raise ValueError('invalid historical key weight')
    z = key.to(dtype=inverse.dtype) * rho**.5
    pz = inverse @ z
    updated = inverse - torch.outer(pz, pz) / (1. + z @ pz)
    updated = (updated + updated.T) * .5
    if not torch.isfinite(updated).all():
        raise ValueError('nonfinite recursive preconditioner')
    return updated


@torch.no_grad()
def coordinates(gradient, bases, inverses, gamma):
    ap, ash = bases
    pp, ps = inverses
    return (gradient @ ap.T) @ pp.to(gradient.dtype), gamma * ((gradient @ ash.T) @ ps.to(gradient.dtype))


@torch.no_grad()
def clipped_step(directions, lr=.01, cap=5.):
    norm = torch.sqrt(sum(v.square().sum() for v in directions))
    scale = -lr * cap / norm.clamp_min(cap)
    delta = tuple(scale * v for v in directions)
    if not all(torch.isfinite(v).all() for v in delta):
        raise ValueError('nonfinite coordinate update')
    return delta, dict(coordinate_gradient_norm=float(norm), proposed_step_norm=float(norm * scale.abs()))


@torch.no_grad()
def merged(base, coefficients, bases):
    # No extra forward module: these temporary tensors reconstruct the native matrix.
    return base + coefficients[0] @ bases[0] + coefficients[1] @ bases[1]


def selfcheck():
    torch.manual_seed(7)
    dtype = torch.float64
    a = torch.linalg.qr(torch.randn(9, 5, dtype=dtype), mode='reduced')[0].T
    ap, ash = a[:3], a[3:]
    base = torch.randn(4, 9, dtype=dtype)
    x = torch.randn(7, 9, dtype=dtype)
    cp = torch.zeros(4, 3, dtype=dtype, requires_grad=True)
    cs = torch.zeros(4, 2, dtype=dtype, requires_grad=True)
    dense = base.clone().requires_grad_()
    dense_loss = (x @ dense.T - .3).square().mean()
    dense_loss.backward()
    branch_loss = (x @ base.T + (x @ ap.T) @ cp.T + (x @ ash.T) @ cs.T - .3).square().mean()
    branch_loss.backward()
    assert torch.allclose(cp.grad, dense.grad @ ap.T, atol=1e-12)
    assert torch.allclose(cs.grad, dense.grad @ ash.T, atol=1e-12)
    pp, ps = torch.eye(3, dtype=dtype), torch.eye(2, dtype=dtype)
    kp, ks = torch.randn(3, dtype=dtype), torch.randn(2, dtype=dtype)
    pp = recurse(pp, kp)
    ps = recurse(ps, ks, .3)
    assert torch.allclose(pp, torch.linalg.inv(torch.eye(3, dtype=dtype)+torch.outer(kp,kp)), atol=1e-12)
    assert torch.allclose(ps, torch.linalg.inv(torch.eye(2, dtype=dtype)+.3*torch.outer(ks,ks)), atol=1e-12)
    assert torch.equal(recurse(ps, ks, 0.), ps)
    g = dense.grad.detach()
    dp, ds = coordinates(g, (ap,ash), (pp,ps), .4)
    assert torch.allclose(dp, cp.grad @ pp, atol=1e-12)
    assert torch.allclose(ds, .4*cs.grad @ ps, atol=1e-12)
    steps, receipt = clipped_step((dp,ds))
    c = tuple(v.detach()+s for v,s in zip((cp,cs),steps))
    w = merged(base,c,(ap,ash))
    assert torch.allclose(x @ w.T, x @ base.T + (x@ap.T)@c[0].T + (x@ash.T)@c[1].T, atol=1e-12)
    assert torch.allclose((w-base).norm(), torch.sqrt(sum(s.square().sum() for s in steps)), atol=1e-12)
    assert (w-base).norm() <= .05+1e-12
    assert torch.equal(w, merged(base,tuple(v.clone() for v in c),(ap,ash)))
    t = torch.tensor([1.,0.], dtype=dtype)
    aligned = gate(t,t)[0]; conflict=gate(t,-t)[0]; weak=gate(t,.01*t)[0]
    assert aligned > .99 and conflict < 1e-4 and weak < .011
    assert gate(t,torch.zeros_like(t))[0] == 0. and gate(t,-t,True)[0] == 1.
    labels = torch.tensor([[-100,-100,-100,1,2]])
    features = torch.randn(1,5,9,dtype=dtype)
    tk, vk, mk = pool_keys(features,labels,(0,2))
    assert torch.allclose(tk,features[0,2:4].float().mean(0))
    assert torch.allclose(vk,features[0,:2].float().mean(0))
    return dict(status='PASS',merged_forward=True,dense_branch_gradient=True,
                recursive_inverse=True,zero_gate_skips_geometry=True,
                aligned_conflict_weak_evidence=True,causal_feature_mask=True,
                checkpoint_reconstruction_exact=True,step_cap=True)


if __name__ == '__main__':
    print(selfcheck())
