"""Bounded full-context history plus eight immutable native anchor packets."""
import torch
from experiments.fixed_input_guard_v1.guard import InputGuard as SketchGuard, right_basis, normalized_rows, finite_cpu

PIN_COLUMNS = 32


class InputGuard(SketchGuard):
    def __init__(self, base_inputs):
        # Parent construction checks storage before the extra bank exists.
        super().__init__(base_inputs)
        self.pinned = torch.zeros((self.base.shape[0], PIN_COLUMNS), dtype=torch.float32)
        self.pin_count = torch.zeros((), dtype=torch.int64)
        self.pin_sources = torch.zeros((), dtype=torch.int64)
        self.check_storage()

    def check_storage(self):
        size = super().check_storage()
        if hasattr(self, 'pinned'):
            size += sum(getattr(self, k).untyped_storage().nbytes() for k in ('pinned', 'pin_count', 'pin_sources'))
        if size > 16 * 1024**2:
            raise RuntimeError('16 MiB protection-state limit exceeded')
        return size

    def state(self):
        return dict(super().state(), **{k: getattr(self, k).clone() for k in ('pinned', 'pin_count', 'pin_sources')})

    @classmethod
    def restore(cls, state):
        keys = {'base', 'history', 'sketch', 'counts', 'discarded', 'pinned', 'pin_count', 'pin_sources'}
        if set(state) != keys:
            raise ValueError('Context guard schema mismatch; no implicit legacy migration')
        old = SketchGuard.restore({k: state[k] for k in keys - {'pinned', 'pin_count', 'pin_sources'}})
        obj = object.__new__(cls)
        obj.__dict__.update(old.__dict__)
        for k, shape, dtype in [('pinned', (old.base.shape[0], PIN_COLUMNS), torch.float32),
                                ('pin_count', (), torch.int64), ('pin_sources', (), torch.int64)]:
            v = state[k]
            if v.device.type != 'cpu' or v.shape != shape or v.dtype != dtype or not torch.isfinite(v).all():
                raise ValueError('Invalid pinned state')
            setattr(obj, k, v.clone())
        n = int(obj.pin_count)
        if not 0 <= n <= 32 or not 0 <= int(obj.pin_sources) <= 8 or int(obj.pin_sources) != int(obj.counts[3]) or n > 4*int(obj.pin_sources):
            raise ValueError('Invalid pinned counts')
        q = torch.cat((obj.base[:, :int(obj.counts[0])], obj.pinned[:, :n]), 1).double()
        if q.numel() and not torch.allclose(q.T @ q, torch.eye(q.shape[1], dtype=q.dtype), atol=2e-5, rtol=0):
            raise ValueError('Invalid pinned basis')
        if obj.pinned[:, n:].count_nonzero():
            raise ValueError('Nonzero unused pinned columns')
        obj.check_storage()
        return obj

    def basis(self):
        fixed = torch.cat((self.base[:, :int(self.counts[0])], self.pinned[:, :int(self.pin_count)]), 1).double()
        fixed = torch.linalg.qr(fixed, mode='reduced').Q
        history = self.history[:, :int(self.counts[1])].double()
        history -= fixed @ (fixed.T @ history)
        residual = right_basis(history.T, 64).double()
        return torch.linalg.qr(torch.cat((fixed, residual), 1), mode='reduced').Q

    def coverage(self, inputs):
        finite_cpu(inputs, 2)
        if inputs.shape[1] != self.base.shape[0] or not len(inputs):
            raise ValueError('Invalid coverage inputs')
        rows = normalized_rows(inputs).double()
        q = self.basis()
        residual = rows - (rows @ q) @ q.T
        energy = rows.square().sum(1)
        ratio = residual.square().sum(1) / energy.clamp_min(1e-30)
        return dict(rows=len(rows), nonzero_rows=int((energy > 0).sum()),
                    mean_uncovered_energy=float(ratio.mean()), max_uncovered_energy=float(ratio.max()),
                    basis_rank=q.shape[1], sketch_discarded=float(self.discarded))

    @torch.no_grad()
    def admit(self, inputs, *, successful, anchors):
        if not isinstance(successful, bool):
            raise ValueError('Explicit success decision required')
        finite_cpu(anchors, 2)
        if anchors.shape[1] != self.base.shape[0] or not 1 <= len(anchors) <= 4:
            raise ValueError('Expected 1..4 native anchor directions')
        if not successful:
            return super().admit(inputs, successful=False)
        if int(self.pin_sources) >= 8:
            raise ValueError('Eight-source anchor capacity exhausted')
        # Validate the shared input boundary before changing either bank.
        finite_cpu(inputs, 2)
        if inputs.shape[1] != self.base.shape[0] or not 1 <= len(inputs) <= 32:
            raise ValueError('Invalid historical inputs')
        fixed = torch.cat((self.base[:, :int(self.counts[0])], self.pinned[:, :int(self.pin_count)]), 1).double()
        fixed = torch.linalg.qr(fixed, mode='reduced').Q
        rows = normalized_rows(anchors).double()
        rows -= (rows @ fixed) @ fixed.T
        extra = right_basis(rows, 4)
        n = int(self.pin_count)
        self.pinned[:, n:n+extra.shape[1]] = extra
        self.pin_count += extra.shape[1]
        self.pin_sources += 1
        result = super().admit(inputs, successful=True)
        self.check_storage()
        return result
