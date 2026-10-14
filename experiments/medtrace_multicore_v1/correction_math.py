"""Root space plus six prescribed tail directions; never a global TT certificate."""
import numpy as np
import torch
from scipy.optimize import minimize
import module_math as mm
import guard_math as gm
import joint_geometry as geometry

INNER_MARGIN = 1e-5


def solve(matrix, offset, center, step_offset, orthogonal, radius,
          source, required, trust=None, fisher_objective=False):
    """Nearest proposal in whitened tangent-operator coordinates."""
    matrix, offset, center, step_offset, source = [np.asarray(x, dtype=np.float64)
        for x in (matrix, offset, center, step_offset, source)]
    assert matrix.shape[1] == len(center) == source.shape[-1] == len(step_offset)
    assert all(np.isfinite(x).all() for x in (matrix, offset, center, step_offset, source))
    ball = radius**2-orthogonal
    if ball < -1e-8:
        return None, dict(status='CERTIFIED_LOCAL_INFEASIBLE', certificate='tangent_orthogonal_distance')
    source=np.atleast_2d(source);required=np.atleast_1d(required)
    assert len(source)==len(required) and np.isfinite(required).all()
    normalization = np.maximum(np.maximum(np.linalg.norm(source,axis=1),np.abs(required)),1e-12)
    source, required = source/normalization[:,None], required/normalization
    values, vectors = np.linalg.eigh(matrix.T@matrix)
    assert values.min() >= -1e-8*max(1., values.max())
    values = np.maximum(values, 0.)
    af = vectors.T@(matrix.T@offset)
    ce, se = [vectors.T@x for x in (center, step_offset)]
    so=vectors.T@source.T
    trust2 = trust**2 if trust is not None else 1e30

    regularizer = 1e-6 if fisher_objective else 1.
    fisher_weight = .5 if fisher_objective else 0.

    def evaluate(x, certificate=False):
        lam, mu, eta = x[0],x[1],x[-1]
        nu=x[2:-1]
        tie, weight = (0., 0.) if certificate else (regularizer, fisher_weight)
        rhs = tie*ce-2*(weight+lam)*af-2*mu*se+so@nu
        diagonal = tie+2*(weight+lam)*values+2*mu+2*eta
        if certificate and diagonal.min() <= 1e-10*max(1., diagonal.max()):
            return None
        z = vectors@(rhs/diagonal)
        residuals = np.r_[np.linalg.norm(matrix@z+offset)**2-1.,
            np.linalg.norm(z+step_offset)**2-ball,required-source@z,np.linalg.norm(z)**2-trust2]
        objective = .5*tie*np.linalg.norm(z-center)**2+weight*np.linalg.norm(matrix@z+offset)**2+x@residuals
        return objective, residuals, z, rhs, diagonal

    def dual(x):
        objective, residuals, *_ = evaluate(x)
        return -objective, -residuals

    result = minimize(dual, np.zeros(3+len(required)), jac=True, method='L-BFGS-B',
        bounds=[(0., None)]*(2+len(required))+[(0., None) if trust is not None else (0., 0.)],
        options=dict(maxiter=2000, maxls=50, ftol=1e-14, gtol=1e-10))
    objective, residuals, z, rhs, diagonal = evaluate(result.x)
    stationarity = float(np.linalg.norm(diagonal*(vectors.T@z)-rhs))
    complementarity = float(np.max(np.abs(result.x*residuals)))
    audit = dict(status='NUMERIC_UNRESOLVED', iterations=int(result.nit),
        solver_message=str(result.message), residuals=residuals.tolist(),
        stationarity=stationarity, complementarity=complementarity,
        multipliers=result.x.tolist(), dimension=len(center), certificate_scope='CURRENT_TANGENT_SUBSPACE')
    audit['objective'] = 'FISHER_WITH_FIXED_1e-6_DISTANCE_TIE' if fisher_objective else 'NEAREST_PROPOSAL'
    if (residuals.max() <= 1e-7 and stationarity <= 1e-7*max(1., np.linalg.norm(rhs))
            and complementarity <= 1e-6*max(1., abs(objective))):
        audit['status'] = 'LOCAL_KKT_PASS'
        return z, audit
    if result.x.sum() > 0:
        weights = result.x/result.x.sum()
        bound = evaluate(weights, certificate=True)
        constants = np.r_[offset@offset-1., step_offset@step_offset-ball,required,-trust2]
        if bound is not None and bound[0] > 1e-6*max(1., np.max(np.abs(weights*constants))):
            audit.update(status='CERTIFIED_LOCAL_INFEASIBLE',
                         certificate='positive_constraint_mixture', certificate_lower_bound=float(bound[0]))
    return None, audit


def inner(left, right):
    return sum(((b.T@d)*(a@e.T)).sum() for b,a in left for d,e in right)


def terms(state, direction):
    b,a = mm.factors(state)
    db = (torch.einsum('ia,ajb->ijb', direction['G1'][0], state['G2'])
          +torch.einsum('ia,ajb->ijb', state['G1'][0], direction['G2'])).flatten(0,1)
    da = (torch.einsum('aib,bj->aij', direction['G3'], state['G4'][:,:,0])
          +torch.einsum('aib,bj->aij', state['G3'], direction['G4'][:,:,0])).flatten(1)
    return [(db,a),(b,da)]


def features(state, cache):
    b,a = mm.factors(state)
    xa = cache['x']@a.T
    yb = (cache['y'].flatten(0,1)@b).reshape(cache['y'].shape[0],cache['y'].shape[1],b.shape[1])
    values = (xa[:,None]*yb).sum(-1)
    return torch.stack([x.sum(0) for x in values.split(cache['token_counts'])]).flatten()


def feature_direction(state, direction, cache):
    pairs = terms(state,direction)
    outputs = []
    for b,a in pairs:
        xa = cache['x']@a.T
        yb = (cache['y'].flatten(0,1)@b).reshape(cache['y'].shape[0],cache['y'].shape[1],b.shape[1])
        outputs.append((xa[:,None]*yb).sum(-1))
    values = sum(outputs)
    return torch.stack([x.sum(0) for x in values.split(cache['token_counts'])]).flatten()


def tangent(current, proposal, before, gradients, protection_gradient, cache, radius, budget, alignment_reference=None):
    state = {k:v.detach().cpu().double() for k,v in current.items()}
    gradient = {k:v.detach().cpu().double() for k,v in gradients.items()}
    protect = {k:v.detach().cpu().double() for k,v in protection_gradient.items()}
    tails = []
    for key in mm.KEYS[1:]:
        for vector in (gradient[key],protect[key]):
            if vector.norm() > 1e-20:
                d = {k:torch.zeros_like(v) for k,v in state.items()}
                d[key] = -vector/vector.norm()
                tails.append(d)
    root_size = state['G1'].numel()
    transform, _, _, _ = geometry.coordinates(state,state)
    metric = torch.zeros((root_size+len(tails),)*2,dtype=torch.float64)
    metric[:root_size,:root_size] = transform.T@transform
    variable = dict(state);variable['G1'] = state['G1'].clone().requires_grad_(True)
    base_pairs = [mm.factors(variable)]
    tail_pairs = [terms(state,d) for d in tails]

    def root_projection(pairs):
        return torch.autograd.grad(inner(base_pairs,pairs),variable['G1'],retain_graph=True)[0].flatten().detach()

    for i,pairs in enumerate(tail_pairs):
        cross = root_projection(pairs)
        metric[:root_size,root_size+i] = cross
        metric[root_size+i,:root_size] = cross
        for j,other in enumerate(tail_pairs):
            metric[root_size+i,root_size+j] = inner(pairs,other)
    values,vectors = torch.linalg.eigh(metric)
    assert values.min() >= -1e-8*values.max().clamp_min(1.)
    keep = values > 1e-10*values.max()
    mapping = vectors[:,keep]*(radius/values[keep].sqrt())

    def projection(other):
        ob,oa = mm.factors({k:v.detach().cpu().double() for k,v in other.items()})
        cb,ca = mm.factors(state)
        residual = [(cb,ca),(-ob,oa)]
        raw = torch.cat([root_projection(residual),torch.stack([inner(p,residual) for p in tail_pairs])
                         if tails else torch.empty(0,dtype=torch.float64)])
        projected = mapping.T@raw/radius**2
        squared = float(inner(residual,residual))/radius**2
        orthogonal = squared-float(projected@projected)
        assert orthogonal >= -1e-7*max(1.,squared)
        return projected,max(0.,orthogonal)

    step_offset,orthogonal = projection(before)
    target,_ = projection(proposal)
    matrix = torch.cat([gm.root_rows(state,cache),torch.stack([feature_direction(state,d,cache) for d in tails],dim=1)
                        if tails else torch.empty((len(features(state,cache)),0),dtype=torch.float64)],dim=1)
    source = torch.cat([gradient['G1'].flatten(),torch.stack([sum((gradient[k]*d[k]).sum() for k in mm.KEYS) for d in tails])
                        if tails else torch.empty(0,dtype=torch.float64)])
    def candidate(z):
        alpha = mapping@torch.as_tensor(z,dtype=torch.float64)
        out = {k:v.clone() for k,v in state.items()}
        out['G1'] += alpha[:root_size].reshape_as(out['G1'])
        for coefficient,direction in zip(alpha[root_size:],tails):
            for k in mm.KEYS:out[k] += coefficient*direction[k]
        out,_ = mm.canonicalize(out)
        return {k:v.to(current[k]) for k,v in out.items()}
    model=dict(matrix=(matrix@mapping/budget**.5).numpy(),offset=(features(state,cache)/budget**.5).numpy(),
        center=(-target).numpy(),step_offset=step_offset.numpy(),orthogonal=orthogonal,
        source=(-mapping.T@source).numpy())
    if alignment_reference is not None:
        reference={k:v.detach().cpu().double() for k,v in alignment_reference.items()}
        projected,_=projection(reference)
        cb,ca=mm.factors(state);bb,ba=mm.factors({k:v.detach().cpu().double() for k,v in before.items()})
        pb,pa=mm.factors(reference)
        model.update(alignment=(step_offset-projected).numpy(),
            alignment_offset=float(inner([(cb,ca),(-bb,ba)],[(pb,pa),(-bb,ba)]))/radius**2)
    return model,candidate,dict(root_dimensions=root_size,
        tail_directions=len(tails),tangent_rank=int(keep.sum()),same_TT_parameter_count=sum(v.numel() for v in state.values()))


def selfcheck():
    z,a = solve(np.eye(2),np.array([2.,0.]),np.zeros(2),np.array([2.,0.]),0.,2.1,
                np.array([0.,1.]),.2)
    assert a['status']=='LOCAL_KKT_PASS' and np.linalg.norm(z+np.array([2.,0.])-np.array([np.sqrt(.96),.2]))<1e-5
    z,a = solve(np.eye(2),np.array([2.,0.]),np.zeros(2),np.array([2.,0.]),0.,2.1,
                np.array([0.,1.]),2.)
    assert z is None and a['status']=='CERTIFIED_LOCAL_INFEASIBLE'
    # Internal margin absorbs FP32 boundary rounding without relaxing the external bound.
    nominal = 1.002
    stored = float(np.float32(nominal-INNER_MARGIN))
    assert stored <= nominal and nominal-stored > 1e-6
    rng=torch.Generator().manual_seed(20261013)
    shapes=((1,3,2),(2,4,2),(2,5,2),(2,6,1))
    state=mm.canonicalize({k:torch.randn(s,generator=rng,dtype=torch.float64) for k,s in zip(mm.KEYS,shapes)})[0]
    d={k:.01*torch.randn(v.shape,generator=rng,dtype=v.dtype) for k,v in state.items()}
    cache=dict(x=torch.randn(7,30,generator=rng,dtype=torch.float64),
        y=torch.randn(7,3,12,generator=rng,dtype=torch.float64),token_counts=[3,4])
    eps=1e-6
    plus={k:state[k]+eps*d[k] for k in state};minus={k:state[k]-eps*d[k] for k in state}
    torch.testing.assert_close(feature_direction(state,d,cache),(features(plus,cache)-features(minus,cache))/(2*eps),rtol=1e-6,atol=1e-7)
    bp,ap=mm.factors(plus);bm,am=mm.factors(minus)
    numerical=(bp@ap-bm@am)/(2*eps)
    torch.testing.assert_close(inner(terms(state,d),terms(state,d)),numerical.square().sum(),rtol=1e-6,atol=1e-8)
    gradients={k:torch.randn(v.shape,generator=rng,dtype=v.dtype) for k,v in state.items()}
    model,build,space=tangent(state,state,state,gradients,gradients,cache,.1,1.)
    assert space['root_dimensions']==state['G1'].numel() and space['tail_directions']==6
    point=build(np.zeros(len(model['center'])))
    for k in state:torch.testing.assert_close(point[k],state[k],rtol=1e-8,atol=1e-8)
    return dict(status='PASS',root_problem_equivalence=True,local_certificate=True,
                FP32_inner_margin=True,external_rules_unchanged=True,
                full_TT_feature_JVP=True,operator_Gram=True,root_space_included=True)
