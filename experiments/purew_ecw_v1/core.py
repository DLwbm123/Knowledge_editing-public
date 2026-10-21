"""Fixed-coordinate sampled functional risk with a native residual envelope."""
import math
import torch


def inner(a, b):
    return float((a.double() * b.double()).sum())


class Memory:
    def __init__(self, left, right, shape):
        self.left, self.right = left.double(), right.double()
        self.shape = tuple(shape)
        self.m = left.shape[1] * right.shape[1]
        device = left.device
        self.HA = torch.zeros((self.m, self.m), dtype=torch.float64, device=device)
        self.HE = torch.zeros_like(self.HA)
        self.N = torch.zeros_like(self.HA)
        self.x = torch.zeros(self.m, dtype=torch.float64, device=device)
        self.v = torch.zeros_like(self.x)
        self.q = 0.
        self.residual = torch.zeros(shape, dtype=torch.float64, device=device)
        self.mean = torch.zeros_like(self.residual)
        self.mass = self.scatter = 0.
        self.prepared = None

    def project(self, full):
        return (self.left.T @ full.reshape(self.shape).double() @ self.right).flatten()

    def embed(self, coordinate):
        return self.left @ coordinate.reshape(self.left.shape[1], self.right.shape[1]) @ self.right.T

    def value(self, coordinate):
        h = self.HA + self.HE
        return self.q + inner(self.v, coordinate) + .5 * inner(coordinate, h @ coordinate)

    def tail(self, residual):
        return self.scatter + self.mass * inner(residual-self.mean, residual-self.mean)

    def inspect(self, actual):
        dx = self.project(actual)
        e = self.residual + actual.double() - self.embed(dx)
        q, tail = self.value(dx), self.tail(e)
        assert q >= -1e-9 and tail >= 0 and math.isfinite(q+tail)
        bound = .5 * (math.sqrt(max(0., 2*q)) + math.sqrt(tail))**2
        return dx, e, dict(coordinate_risk=max(0., q), residual_energy=tail,
            native_risk_bound=bound, residual_norm=math.sqrt(inner(e,e)),
            residual_budget_fraction=tail/.02, cross_term=math.sqrt(max(0.,2*q)*tail))

    def advance(self, actual, gradient):
        dx, e, event = self.inspect(actual)
        self.q = self.value(dx)
        self.v += (self.HA+self.HE) @ dx
        self.x += dx
        self.residual = e
        return event

    def consolidate(self, full_rows, block='answer'):
        # Only the cumulative covariance and residual birth moments survive this call.
        h = self.HA if block == 'answer' else self.HE
        mass = 0.
        for row in full_rows:
            z = self.project(row)
            h += torch.outer(z,z)
            mass += inner(row,row)
        if mass:
            total = self.mass + mass
            delta = self.residual-self.mean
            self.scatter += self.mass*mass/total * inner(delta,delta)
            self.mean += mass/total * delta
            self.mass = total
        self.prepared = None
        return dict(projected_trace=float(torch.trace(h)), full_probe_mass=mass,
            cumulative_mass=self.mass, residual_scatter=self.scatter)

    def prepare(self, whiten=True):
        metric = torch.eye(self.m, dtype=torch.float64, device=self.x.device)
        if whiten:
            blocks = [h for h in (self.HA, self.HE, self.N) if float(torch.trace(h)) > 0]
            if blocks:
                metric = sum(self.m*h/torch.trace(h) for h in blocks)/len(blocks)
                vals, vectors = torch.linalg.eigh(metric)
                metric = (vectors * vals.clamp(.1,10.)) @ vectors.T
        vals, vectors = torch.linalg.eigh(metric)
        invroot = (vectors * vals.rsqrt()) @ vectors.T
        spectrum, axes = torch.linalg.eigh(invroot @ (self.HA+self.HE) @ invroot)
        self.prepared = (metric, invroot @ axes, spectrum.clamp_min(0))

    def solve(self, gradient, eta=.01, epsilon=.01):
        assert self.prepared is not None
        metric, transform, spectrum = self.prepared
        current_tail = self.tail(self.residual)
        if current_tail > 2*epsilon:
            return torch.zeros_like(self.x), dict(solver_reason='RESIDUAL_ENVELOPE_INFEASIBLE')
        cap = .5*(math.sqrt(2*epsilon)-math.sqrt(current_tail))**2
        g, v = self.project(gradient), self.v
        gt, vt = transform.T@g, transform.T@v
        def proposal(multiplier):
            return -eta * (transform @ ((gt+multiplier*vt)/(1+eta*multiplier*spectrum)))
        lo, hi, iterations = 0., 1., 0
        if self.value(proposal(0)) <= cap:
            delta, multiplier = proposal(0), 0.
        else:
            while iterations < 24 and self.value(proposal(hi)) > cap:
                lo, hi, iterations = hi, 2*hi, iterations+1
            if self.value(proposal(hi)) > cap:
                return torch.zeros_like(self.x), dict(solver_reason='FINITE_ROOT_REJECT')
            while iterations < 48:
                mid = (lo+hi)/2
                if self.value(proposal(mid)) > cap:lo = mid
                else:hi = mid
                iterations += 1
            delta, multiplier = proposal(hi), hi
        norm = math.sqrt(inner(delta,delta))
        if norm > .05:delta *= .05/norm
        return delta, dict(multiplier=multiplier, root_iterations=iterations,
            coordinate_cap=cap, proposal_norm=math.sqrt(inner(delta,delta)),
            current_residual_energy=current_tail)

    def state(self):
        result={k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v)
            for k,v in self.__dict__.items() if k != 'prepared'}
        result['prepared']=tuple(v.detach().cpu() for v in self.prepared) if self.prepared is not None else None
        return result

    @classmethod
    def restore(cls, state, device):
        result = cls(state['left'].to(device),state['right'].to(device),state['shape'])
        for k,v in state.items():
            if k=='prepared' and v is not None:v=tuple(x.to(device) for x in v)
            elif isinstance(v,torch.Tensor):v=v.to(device)
            setattr(result,k,v)
        assert result.q >= -1e-9 and result.mass >= 0 and result.scatter >= 0
        return result


def basis(weight, rank=32):
    # The frozen source-free randomized SVD uses only the original model weight.
    with torch.random.fork_rng(devices=[weight.device.index] if weight.is_cuda else []):
        torch.manual_seed(20260912)
        left, _, right = torch.svd_lowrank(weight.detach().float(),q=rank,niter=2)
    left = torch.linalg.qr(left.double(),mode='reduced').Q
    right = torch.linalg.qr(right.double(),mode='reduced').Q
    eye = torch.eye(rank,dtype=torch.float64,device=weight.device)
    assert float((left.T@left-eye).abs().max()) < 1e-10
    assert float((right.T@right-eye).abs().max()) < 1e-10
    return left, right
