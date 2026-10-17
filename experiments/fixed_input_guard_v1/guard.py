"""Fixed-capacity input protection. No model loader or experiment entry point."""
import math
import torch

CAP_BYTES = 16 * 1024**2
BASE_RANK, HISTORY_RANK, SKETCH_ROWS, MAX_INPUTS = 32, 64, 128, 32
ATOL, RTOL = 1e-7, 1e-6
RANK_ATOL = 1e-6
REPAIR_FINE, REPAIR_COARSE, REPAIR_ROWS, REPAIR_STEPS = 128, 512, 32, 4
MAX_RELATIVE_DISTORTION = .05


def finite_cpu(x, ndim):
    if not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or x.ndim != ndim:
        raise ValueError('Expected CPU tensor of the stated dimension')
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Expected finite floating point tensor')


def right_basis(rows, limit):
    if not len(rows):
        return rows.new_empty((rows.shape[1], 0))
    _, s, vh = torch.linalg.svd(rows.double(), full_matrices=False)
    rank = min(limit, int((s > max(RANK_ATOL, max(rows.shape) * torch.finfo(s.dtype).eps * float(s[0]))).sum()))
    return vh[:rank].T.float().contiguous()


def normalized_rows(inputs):
    rows = inputs.detach().double().clone()
    rows /= rows.abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    rows /= rows.norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    return rows.float()


def select_inputs(views):
    """Select up to 32 evenly spaced token positions in caller-fixed view order."""
    if not 1 <= len(views) <= 5:
        raise ValueError('Expected 1..5 current-edit views')
    selected = []
    for i, view in enumerate(views):
        finite_cpu(view, 2)
        if not len(view) or view.shape[1] != views[0].shape[1]:
            raise ValueError('Empty view or mismatched width')
        quota = MAX_INPUTS // len(views) + (i < MAX_INPUTS % len(views))
        indices = torch.linspace(0, len(view)-1, min(quota, len(view))).round().long()
        selected.append(view.detach()[indices].float())
    return torch.cat(selected)


@torch.no_grad()
def repair_native(candidate, before, projected, q):
    """Two bounded native correction banks; no persistent compensation parameters."""
    columns = REPAIR_FINE + REPAIR_COARSE
    corrected_rows = 0
    if candidate.shape[1] < columns:
        return dict(corrected_rows=0, reason='INSUFFICIENT_CORRECTION_COLUMNS')
    for first in range(0, len(candidate), REPAIR_ROWS):
        x = candidate[first:first+REPAIR_ROWS]
        residual = (x.double()-before[first:first+REPAIR_ROWS].double()) @ q
        goal = .5 * (ATOL / math.sqrt(len(candidate)) + RTOL * projected[first:first+REPAIR_ROWS].norm(dim=1))
        if (residual.norm(dim=1) <= goal).all():
            continue
        indices = x.abs().float().topk(columns, largest=False, dim=1).indices
        directions = q[indices]
        values = x.gather(1, indices).double()
        # Free the finest bank near zero, then compensate its displacement with the coarse bank.
        residual -= (values[:, :REPAIR_FINE].unsqueeze(1) @ directions[:, :REPAIR_FINE]).squeeze(1)
        values[:, :REPAIR_FINE] = 0
        for bank in (slice(REPAIR_FINE, None), slice(None, REPAIR_FINE)):
            a = directions[:, bank]
            inverse = torch.linalg.pinv(a.transpose(1, 2) @ a, hermitian=True, rtol=1e-10)
            for _ in range(REPAIR_STEPS):
                correction = (a @ (inverse @ -residual.unsqueeze(-1))).squeeze(-1)
                trial = (values[:, bank]+correction).to(x.dtype).double()
                trial_residual = residual + ((trial-values[:, bank]).unsqueeze(1) @ a).squeeze(1)
                better = torch.isfinite(trial).all(1) & (trial_residual.norm(dim=1) < residual.norm(dim=1))
                values[:, bank] = torch.where(better[:, None], trial, values[:, bank])
                residual = torch.where(better[:, None], trial_residual, residual)
        x.scatter_(1, indices, values.to(x.dtype))
        corrected_rows += len(x)
    return dict(corrected_rows=corrected_rows, reason='BOUNDED_TWO_BANK_REPAIR')


class InputGuard:
    def __init__(self, base_inputs):
        finite_cpu(base_inputs, 2)
        d = base_inputs.shape[1]
        if not 96 <= d <= 14336 or not 1 <= len(base_inputs) <= MAX_INPUTS:
            raise ValueError('Expected 1..32 calibration inputs, width 96..14336')
        self.base = torch.zeros((d, BASE_RANK), dtype=torch.float32)
        q = right_basis(normalized_rows(base_inputs), BASE_RANK)
        self.base[:, :q.shape[1]] = q
        self.history = torch.zeros((d, HISTORY_RANK), dtype=torch.float32)
        self.sketch = torch.zeros((SKETCH_ROWS, d), dtype=torch.float32)
        # Fixed-size metadata: base rank, history rank, used rows, accepted edits.
        self.counts = torch.tensor([q.shape[1], 0, 0, 0], dtype=torch.int64)
        self.discarded = torch.zeros((), dtype=torch.float64)
        self.check_storage()

    def state(self):
        # Fixed schema only; never include per-edit lists or activation packets.
        return {k: getattr(self, k).clone() for k in ('base', 'history', 'sketch', 'counts', 'discarded')}

    def check_storage(self):
        size = sum(getattr(self, k).untyped_storage().nbytes()
                   for k in ('base', 'history', 'sketch', 'counts', 'discarded'))
        if size > CAP_BYTES:
            raise RuntimeError('16 MiB protection-state limit exceeded')
        return size

    @classmethod
    def restore(cls, state):
        if set(state) != {'base', 'history', 'sketch', 'counts', 'discarded'}:
            raise ValueError('Unexpected state schema')
        d = state['base'].shape[0]
        if not 96 <= d <= 14336:
            raise ValueError('Invalid input dimension')
        shapes = {'base': (d, 32), 'history': (d, 64), 'sketch': (128, d),
                  'counts': (4,), 'discarded': ()}
        obj = object.__new__(cls)
        for key, shape in shapes.items():
            value = state[key]
            dtype = torch.int64 if key == 'counts' else torch.float64 if key == 'discarded' else torch.float32
            if value.device.type != 'cpu' or value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid state tensor: ' + key)
            setattr(obj, key, value.clone())
        b, h, used, edits = obj.counts.tolist()
        if not (0 <= b <= 32 and 0 <= h <= 64 and 0 <= used <= 128 and edits >= 0 and obj.discarded >= 0):
            raise ValueError('Invalid metadata')
        q = torch.cat((obj.base[:, :b], obj.history[:, :h]), dim=1)
        if q.numel() and not torch.allclose(q.T @ q, torch.eye(b+h), atol=2e-5, rtol=0):
            raise ValueError('Nonorthogonal restored basis')
        obj.check_storage()
        return obj

    def basis(self):
        b, h = self.counts[:2].tolist()
        q = torch.cat((self.base[:, :b], self.history[:, :h]), dim=1)
        # Reorthogonalize in double precision for the native-write certificate.
        return torch.linalg.qr(q.double(), mode='reduced').Q

    @torch.no_grad()
    def admit(self, inputs, *, successful):
        finite_cpu(inputs, 2)
        if inputs.shape[1] != self.base.shape[0] or not 1 <= len(inputs) <= MAX_INPUTS:
            raise ValueError('Expected at most 32 selected inputs at the calibrated width')
        if not isinstance(successful, bool):
            raise ValueError('Explicit success decision required')
        if not successful:
            return False
        if self.counts[3] == torch.iinfo(torch.int64).max:
            raise OverflowError('Edit counter exhausted')
        q0 = torch.linalg.qr(self.base[:, :int(self.counts[0])].double(), mode='reduced').Q
        rows = normalized_rows(inputs).double()
        rows -= (rows @ q0) @ q0.T
        # Do not turn numerical residue of already-protected inputs into history.
        rows[rows.norm(dim=1) <= RANK_ATOL] = 0
        for row in rows:
            used = int(self.counts[2])
            if used == SKETCH_ROWS:
                _, s, vh = torch.linalg.svd(self.sketch, full_matrices=False)
                shrink = s[HISTORY_RANK].square()
                self.discarded += shrink.double()
                self.sketch.zero_()
                self.sketch[:HISTORY_RANK] = (s[:HISTORY_RANK].square()-shrink).clamp_min(0).sqrt()[:, None] * vh[:HISTORY_RANK]
                used = HISTORY_RANK
            self.sketch[used].copy_(row)
            self.counts[2] = used + 1
        rows = self.sketch[:int(self.counts[2])].double()
        # Remove numerical base leakage before extracting the historical basis.
        qh = right_basis(rows - (rows @ q0) @ q0.T, HISTORY_RANK)
        self.history.zero_()
        self.history[:, :qh.shape[1]] = qh
        self.counts[1] = qh.shape[1]
        self.counts[3] += 1
        self.check_storage()
        return True

    @torch.no_grad()
    def write(self, weight, optimizer_delta):
        """Project final optimizer delta, write native dtype, certify or roll back."""
        finite_cpu(weight, 2)
        finite_cpu(optimizer_delta, 2)
        if weight.shape != optimizer_delta.shape or weight.shape[1] != self.base.shape[0]:
            raise ValueError('Weight/update shape mismatch')
        if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('Unsupported native dtype')
        q = self.basis()
        d = optimizer_delta.detach().double()
        projected = d - (d @ q) @ q.T
        before = weight.detach().clone()
        candidate = (before.double() + projected).to(weight.dtype)
        if not torch.isfinite(candidate).all():
            return dict(accepted=False, reason='NONFINITE_NATIVE', residual=None)
        initial = candidate.double()-before.double()
        initial_residual = float((initial @ q).norm())
        repair = dict(corrected_rows=0, reason='NOT_NEEDED')
        if weight.dtype in (torch.float16, torch.bfloat16) and initial_residual > ATOL + RTOL * float(initial.norm()):
            repair = repair_native(candidate, before, projected, q)
        del initial
        try:
            weight.copy_(candidate)
            actual = weight.double() - before.double()
            norm = float(actual.norm())
            residual = float((actual @ q).norm())
            limit = ATOL + RTOL * norm
            projected_norm = float(projected.norm())
            distortion = float((actual-projected).norm()) / projected_norm if projected_norm else (0. if norm == 0 else math.inf)
            accepted = norm > 0 and math.isfinite(residual) and residual <= limit and distortion <= MAX_RELATIVE_DISTORTION
            result = dict(accepted=accepted, reason='ACCEPTED' if accepted else 'ZERO_NATIVE' if norm == 0 else 'NATIVE_PROJECTION_REJECTED' if residual > limit else 'DIRECTION_DISTORTION_REJECTED',
                          residual=residual, limit=limit, update_norm=norm,
                          relative_residual=residual / norm if norm else 0.,
                          ideal_residual=float((projected @ q).norm()),
                          initial_native_residual=initial_residual,
                          relative_direction_distortion=distortion, repair=repair)
            if not accepted:
                weight.copy_(before)
            return result
        except BaseException:
            weight.copy_(before)
            raise
