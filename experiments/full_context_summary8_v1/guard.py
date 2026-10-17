"""Shared 1024-row sketch; native write/repair and fixed pins stay unchanged."""
import torch
from experiments.fixed_input_guard8_v1.context_guard import InputGuard as Pins
from experiments.fixed_input_guard_v1.guard import normalized_rows, right_basis, finite_cpu, RANK_ATOL

ROWS = 1024
FIELDS = ('base', 'history', 'sketch', 'counts', 'discarded', 'pinned', 'pin_count', 'pin_sources', 'compressions')


def compress(rows):
    # FP64 decomposition avoids square-root amplification of FP32 flat-spectrum noise.
    _, s, vh = torch.linalg.svd(rows.double(), full_matrices=False)
    delta = s[64].double().square() if len(s) > 64 else torch.zeros((), dtype=torch.float64)
    kept = (s[:64].double().square()-delta).clamp_min(0).sqrt()[:, None]*vh[:64].double()
    tail = float(s[64:].double().square().sum())
    return kept.float(), dict(delta=float(delta), tail_loss=tail, shrink_loss=64*float(delta),
        before_energy=float(rows.double().square().sum()), after_energy=float(kept.square().sum()),
        nonzero_rank=int((kept.norm(dim=1) > RANK_ATOL).sum()))


class InputGuard(Pins):
    def __init__(self, inputs):
        super().__init__(inputs)
        self.sketch = torch.zeros((ROWS, inputs.shape[1]), dtype=torch.float32)
        self.compressions = torch.zeros((), dtype=torch.int64)
        self.check_storage()

    def check_storage(self):
        size = sum(getattr(self, k).untyped_storage().nbytes() for k in FIELDS if hasattr(self, k))
        if size >= 1_000_000_000:
            raise RuntimeError('Persistent protection state exceeds 1 GB')
        return size

    def state(self):
        return {k: getattr(self, k).clone() for k in FIELDS}

    @classmethod
    def restore(cls, state):
        if set(state) != set(FIELDS):
            raise ValueError('New sketch schema required; no legacy migration')
        d = state['base'].shape[0]
        if not 96 <= d <= 14336:
            raise ValueError('Invalid summary width')
        shapes = dict(base=(d,32), history=(d,64), sketch=(ROWS,d), counts=(4,), discarded=(),
                      pinned=(d,32), pin_count=(), pin_sources=(), compressions=())
        obj = object.__new__(cls)
        for k, shape in shapes.items():
            value = state[k]
            dtype = torch.int64 if k in ('counts','pin_count','pin_sources','compressions') else torch.float64 if k == 'discarded' else torch.float32
            if value.device.type != 'cpu' or value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid summary field: '+k)
            setattr(obj, k, value.clone())
        b,h,used,sources = obj.counts.tolist()
        pins = int(obj.pin_count)
        if not (0 <= b <= 32 and 0 <= h <= 64 and 0 <= used <= ROWS and 0 <= sources <= 8
                and int(obj.pin_sources) == sources and 0 <= pins <= 4*sources
                and obj.compressions >= 0 and obj.discarded >= 0):
            raise ValueError('Invalid summary metadata')
        for q in (torch.cat((obj.base[:,:b], obj.history[:,:h]), 1),
                  torch.cat((obj.base[:,:b], obj.pinned[:,:pins]), 1)):
            if not torch.allclose(q.T@q, torch.eye(q.shape[1]), atol=2e-5, rtol=0):
                raise ValueError('Nonorthogonal restored protection')
        if any(x.count_nonzero() for x in (obj.base[:,b:],obj.history[:,h:],obj.pinned[:,pins:],obj.sketch[used:])):
            raise ValueError('Nonzero unused state')
        obj.check_storage()
        return obj

    @torch.no_grad()
    def begin(self, anchors):
        finite_cpu(anchors, 2)
        if anchors.shape[1] != self.base.shape[0] or not 1 <= len(anchors) <= 4 or self.pin_sources >= 8:
            raise ValueError('Invalid anchor admission')
        fixed = torch.cat((self.base[:,:int(self.counts[0])], self.pinned[:,:int(self.pin_count)]), 1).double()
        fixed = torch.linalg.qr(fixed, mode='reduced').Q
        rows = normalized_rows(anchors).double()
        extra = right_basis(rows-(rows@fixed)@fixed.T, 4)
        n = int(self.pin_count)
        self.pinned[:,n:n+extra.shape[1]] = extra
        self.pin_count += extra.shape[1]
        self.pin_sources += 1

    @torch.no_grad()
    def add(self, weighted):
        finite_cpu(weighted, 2)
        if weighted.shape[1] != self.base.shape[0] or len(weighted) > 4096:
            raise ValueError('Invalid bounded weighted view')
        q0 = torch.linalg.qr(self.base[:,:int(self.counts[0])].double(), mode='reduced').Q
        rows = weighted.double()
        rows = rows-(rows@q0)@q0.T
        norms = rows.norm(dim=1)
        cleared = norms <= RANK_ATOL
        event = dict(input_rows=len(rows), input_energy=float(weighted.double().square().sum()),
            base_residual_energy=float(rows.square().sum()), cleared_rows=int(cleared.sum()),
            cleared_energy=float(rows[cleared].square().sum()), residual_zero_threshold=RANK_ATOL, compressions=[])
        rows[cleared] = 0
        event['post_clear_energy'] = float(rows.square().sum())
        # Keep zero-row insertion behavior shared with the original sampled baseline.
        for row in rows:
            used = int(self.counts[2])
            if used == ROWS:
                kept, report = compress(self.sketch)
                self.sketch.zero_()
                self.sketch[:len(kept)] = kept
                self.discarded += report['delta']
                self.compressions += 1
                event['compressions'].append(report)
                used = 64
            self.sketch[used] = row
            self.counts[2] = used+1
        return event

    @torch.no_grad()
    def end(self):
        if int(self.pin_sources) != int(self.counts[3])+1:
            raise ValueError('Source transaction mismatch')
        q0 = torch.linalg.qr(self.base[:,:int(self.counts[0])].double(), mode='reduced').Q
        rows = self.sketch[:int(self.counts[2])].double()
        qh = right_basis(rows-(rows@q0)@q0.T, 64)
        self.history.zero_(); self.history[:,:qh.shape[1]] = qh
        self.counts[1] = qh.shape[1]; self.counts[3] += 1
        rank = self.basis().shape[1]-int(self.counts[0])-int(self.pin_count)
        return dict(history_rank=int(self.counts[1]), effective_new_history_rank=rank,
                    buffer_rows=int(self.counts[2]), compression_events=int(self.compressions),
                    cumulative_delta=float(self.discarded), protection_bytes=self.check_storage())
