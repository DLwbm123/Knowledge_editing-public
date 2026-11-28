"""Contrastive generalized eigensolver for the TT input factors."""
import copy
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Moment:
    # X contains one weighted RMS-normalized feature per row.
    rows: torch.Tensor

    @classmethod
    def from_queries(cls, queries, device='cpu'):
        if not queries or any(x.ndim != 2 or not x.size(0) for x in queries):
            raise ValueError('Moment needs nonempty query feature matrices')
        rows = torch.cat([q.to(device=device, dtype=torch.float64) /
                          (len(queries) * len(q)) ** .5 for q in queries])
        if not torch.isfinite(rows).all():
            raise ValueError('Nonfinite moment')
        return cls(rows)

    def gram(self, right):
        z = self.rows @ right
        return z.T @ z

    def tt_gram(self, g4, modes):
        x = (self.rows.reshape(-1, modes, g4.size(1)) @ g4.T).flatten(1)
        return x.T @ x

    def compress_g3(self, g3):
        rank, modes, bond = g3.shape
        # Keep the low-rank factor for small cohorts.
        rows = torch.einsum('nij,ris->nrsj', self.rows.reshape(len(self.rows), modes, -1), g3)
        rows = rows.reshape(len(rows), rank, -1)

        def gram(g4):
            projected = rows @ g4.flatten()
            return projected.T @ projected

        return gram


@dataclass(frozen=True)
class CovarianceMoment:
    """Exact Gram for the large shared cohort, cached once on the target device."""
    matrix: torch.Tensor

    def gram(self, right):
        return right.T @ (self.matrix @ right)

    def tt_gram(self, g4, modes):
        width = g4.size(1)
        bond = g4.size(0)
        blocks = self.matrix.reshape(modes,width,modes,width) @ g4.T
        left = blocks.permute(0,2,3,1).reshape(modes,modes*bond,width)
        return (left @ g4.T).transpose(1,2).reshape(modes*bond,modes*bond)

    def compress_g3(self, g3):
        rank, modes, bond = g3.shape
        width = self.matrix.size(0) // modes
        projection = g3.transpose(1, 2).reshape(rank * bond, modes)
        # Apply G3 along each TT mode of the background covariance before Adam.
        blocks = self.matrix.reshape(modes, width, modes * width).transpose(0, 1)
        left = (projection @ blocks).reshape(width, rank * bond, modes, width)
        right = left.transpose(2, 3).reshape(width * rank * bond, width, modes) @ projection.T
        reduced = right.reshape(width, rank * bond, width, rank * bond)
        reduced = reduced.permute(1, 0, 3, 2).contiguous().reshape(rank, bond * width, rank, bond * width)

        def gram(g4):
            flat = g4.flatten()
            return ((reduced @ flat).transpose(1, 2)) @ flat

        return gram


def input_matrix(g3, g4):
    return (g3 @ g4).flatten(1)


def row_basis(a):
    if not torch.isfinite(a).all() or int(torch.linalg.matrix_rank(a)) != a.size(0):
        raise ValueError('Degenerate read-in rows')
    return torch.linalg.qr(a.T, mode='reduced').Q.T.contiguous()


def orthogonalize(g3, g4):
    _, r = torch.linalg.qr(input_matrix(g3, g4).T)
    return torch.linalg.solve_triangular(
        r.T, g3.flatten(1), upper=False).reshape_as(g3)


def generalized_top(signal, noise, rank):
    chol = torch.linalg.cholesky((noise + noise.T) * .5)
    left = torch.linalg.solve_triangular(chol, signal, upper=False)
    white = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    values, vectors = torch.linalg.eigh((white + white.T) * .5)
    columns = torch.linalg.solve_triangular(chol.T, vectors[:, -rank:].flip(1), upper=True)
    return values[-rank:].flip(0), columns


def objective(a, positive, negatives, ridge):
    noise = ridge * (a @ a.T)
    for weight, moment in negatives:
        if weight:
            noise = noise + weight * moment.gram(a.T)
    signal = positive.gram(a.T)
    return torch.trace(torch.linalg.solve(noise, signal))


def compress_objective(g3, positive, negatives, ridge):
    signal = positive.compress_g3(g3)
    terms = [(weight, moment.compress_g3(g3)) for weight, moment in negatives if weight]

    def evaluate(g4):
        a = input_matrix(g3, g4)
        noise = ridge * (a @ a.T)
        for weight, gram in terms:
            noise = noise + weight * gram(g4)
        return torch.trace(torch.linalg.solve(noise, signal(g4)))

    return evaluate


def g3_step(g4, positive, negatives, ridge, modes, rank=4):
    def projected(moment):
        return moment.tt_gram(g4,modes)

    noise = ridge * torch.kron(torch.eye(modes, device=g4.device, dtype=g4.dtype),
                               (g4 @ g4.T).contiguous())
    for weight, moment in negatives:
        if weight:
            noise.add_(projected(moment), alpha=weight)
    eigenvalues, columns = generalized_top(projected(positive), noise, rank)
    return eigenvalues, columns.T.reshape(rank, modes, g4.size(0)).contiguous()


class BackgroundInverse:
    """Shared exact scalar-ridge inverse, factored once per background weight."""
    def __init__(self, bg, gamma, ridge, dim, background_gram=None):
        if ridge <= 0 or gamma < 0:
            raise ValueError('Invalid regularization')
        self.ridge = ridge
        self.rows = None
        self.chol = None
        if gamma:
            if background_gram is not None:
                if background_gram.shape!=(dim,dim) or background_gram.dtype!=torch.float64:
                    raise ValueError('Invalid shared background Gram')
                gram = background_gram.to('cpu') * gamma
                gram.diagonal().add_(ridge)
            else:
                y = bg.rows.to('cpu', torch.float64) * gamma ** .5
                if y.size(0) < dim:
                    self.rows = y
                    gram = (y @ y.T) / ridge
                    gram.diagonal().add_(1)
                else:
                    gram = y.T @ y
                    gram.diagonal().add_(ridge)
            self.chol = torch.linalg.cholesky(gram)

    def apply(self, rhs):
        if self.chol is None:
            return rhs / self.ridge
        if self.rows is None:
            return torch.cholesky_solve(rhs, self.chol)
        z = self.rows @ rhs / self.ridge
        return (rhs - self.rows.T @ torch.cholesky_solve(z, self.chol)) / self.ridge


def unrestricted(positive, negatives, inverse, rank=4):
    p = positive.rows.to('cpu', torch.float64)
    rhs = p.T
    z = inverse.apply(rhs)
    parts = [m.rows.to('cpu', torch.float64) * w ** .5 for w, m in negatives if w]
    if parts:
        u = torch.cat(parts)
        y = inverse.apply(u.T)
        gram = u @ y
        gram.diagonal().add_(1)
        z = z - y @ torch.cholesky_solve(u @ z, torch.linalg.cholesky(gram))
    k = p @ z
    values, cols = torch.linalg.eigh((k + k.T) * .5)
    if len(values) < rank or values[-rank] <= 0:
        raise ValueError('Insufficient positive rank')
    return values[-rank:].flip(0), row_basis((z @ cols[:, -rank:].flip(1)).T)


def tt_svd(a, modes=112, bond=8):
    flattened = a.reshape(-1, a.size(1) // modes)
    _, _, vh = torch.linalg.svd(flattened, full_matrices=False)
    if vh.size(0) < bond:
        raise ValueError('TT-SVD cannot fill the bond')
    return vh[:bond].contiguous()


def fit(positive, negatives, ridge, initial_g4, modes=112, rank=4, rounds=10, steps=50, *, profile=None):
    begin = time.monotonic()
    g4 = row_basis(initial_g4.to(positive.rows)).detach().requires_grad_()
    adam = torch.optim.Adam([g4], lr=.01)
    trace = []
    for iteration in range(rounds):
        if profile is not None:
            phase_started = time.monotonic()
        with torch.no_grad():
            if iteration:
                g4.copy_(row_basis(g4))
                adam.state.clear()
            vals, g3 = g3_step(g4, positive, negatives, ridge, modes, rank)
            current = float(objective(input_matrix(g3, g4), positive, negatives, ridge))
            if trace and current < trace[-1] - 1e-8 * abs(trace[-1]):
                raise RuntimeError('G3 exact step decreased objective')
            trace.append(current)
            if profile is not None:
                profile['g3'] += time.monotonic() - phase_started
                phase_started = time.monotonic()
            g4_objective = compress_objective(g3, positive, negatives, ridge)
            if profile is not None:
                if g4.is_cuda:
                    torch.cuda.synchronize(g4.device)
                profile['compression'] += time.monotonic() - phase_started
                phase_started = time.monotonic()
        for _ in range(steps):
            adam.zero_grad(set_to_none=True)
            (-g4_objective(g4)).backward()
            saved = g4.detach().clone()
            state = copy.deepcopy(adam.state_dict())
            adam.step()
            with torch.no_grad():
                try:
                    proposal = float(g4_objective(g4))
                except torch.linalg.LinAlgError:
                    proposal = float('nan')
                if not (abs(proposal) < float('inf') and proposal >= current - 1e-8 * abs(current)):
                    g4.copy_(saved)
                    adam.load_state_dict(state)
                    adam.param_groups[0]['lr'] *= .5
                else:
                    current = proposal
                trace.append(current)
        if profile is not None:
            profile['adam'] += time.monotonic() - phase_started
        del g4_objective
    with torch.no_grad():
        # End at a G3 optimum so the reported eigenvalues and final objective agree.
        vals, g3 = g3_step(g4, positive, negatives, ridge, modes, rank)
        g3 = orthogonalize(g3, g4)
        final = float(objective(input_matrix(g3, g4), positive, negatives, ridge))
        if final < trace[-1] - 1e-8 * abs(trace[-1]):
            raise RuntimeError('Final G3 step decreased objective')
        trace.append(final)
    return dict(G3=g3.cpu(), G4=g4.detach().cpu().unsqueeze(-1),
                eigenvalues=vals.tolist(), objective=final, trace=trace, seconds=time.monotonic()-begin)
