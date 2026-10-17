"""Real runtime bridge: native module, prepared edit batches, original-Base H."""
from dataclasses import replace
import torch
import torch.nn.functional as F
from .configuration import LAYER


def h_loss(logits, labels, target_odds):
    z = logits.double()
    target = z.gather(1, labels[:, None]).squeeze(1)
    other = z.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    odds = target - torch.logsumexp(other, dim=1)
    return F.relu(target_odds.to(z.device)-odds).mean()


class RuntimeAdapter:
    def __init__(self, runtime, budget):
        self.runtime, self.budget = runtime, budget
        self.layer = runtime.get_module(LAYER)
        self.weight = self.layer.weight
        if self.weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('Native half precision is required')
        runtime.model.eval()
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(parameter is self.weight)
        self.handles = [runtime.model.register_forward_pre_hook(lambda *_: budget.consume('forwards'))]
        if hasattr(runtime.model, 'get_vision_tower'):
            self.handles.append(runtime.model.get_vision_tower().register_forward_pre_hook(lambda *_: budget.consume('forwards')))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def pack(self, prepared):
        labels = prepared.labels.detach().cpu()
        kwargs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                  for k, v in prepared.forward_kwargs().items()}
        span = (int(prepared.image_token_start), int(prepared.image_token_end))
        return kwargs, labels, span

    def views(self, record, fits):
        if len(fits) != 4:
            raise ValueError('Exactly four original FIT questions required')
        batches = []
        for question in [record.question] + list(fits):
            prepared = self.runtime.build_edit_batch(replace(record, question=question))
            labels = prepared.labels.detach().cpu()
            kwargs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in prepared.forward_kwargs().items()}
            if labels.ndim != 2 or labels.shape[0] != 1 or not (labels[:, 1:] != -100).any():
                raise ValueError('Invalid single-source answer positions')
            if any(isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != self.weight.dtype for v in kwargs.values()):
                raise ValueError('Prepared inputs changed native dtype')
            batches.append(self.pack(prepared))
        return batches

    def logits(self, batch):
        kwargs, labels = batch[:2]
        kwargs = {k: v.to(self.weight.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        labels = labels.to(self.weight.device)[:, 1:]
        mask = labels != -100
        z = self.runtime.model(**kwargs).logits[:, :-1].float()
        if z.shape[:2] != labels.shape or not torch.isfinite(z).all():
            raise ValueError('Invalid native forward')
        return z[mask], labels[mask]

    @torch.no_grad()
    def teacher(self, views):
        anchors = []
        for batch in views:
            z, y = self.logits(batch)
            z = z.double()
            target = z.gather(1, y[:, None]).squeeze(1)
            z.scatter_(1, y[:, None], -torch.inf)
            odds = target-torch.logsumexp(z, dim=1)
            anchors.append(odds.clamp_min(__import__('math').log(.95/.05)).cpu())
        return anchors

    def objective(self, views, anchors, slot):
        return sum(.5*h_loss(*self.logits(views[k]), anchors[k]) for k in (0, 1+(slot-1)%4))

    @torch.no_grad()
    def measure(self, views):
        value = 0.
        for coefficient, batch in zip((.5, .125, .125, .125, .125), views):
            z, y = self.logits(batch)
            value += coefficient*float(F.cross_entropy(z.double(), y))
        return value

    @torch.no_grad()
    def inputs(self, views, *, anchors=False):
        selected = []
        for i, batch in enumerate(views[:1] if anchors else views):
            positions = context_positions(batch, 4 if anchors else 32//len(views)+(i < 32 % len(views)), anchors=anchors)
            captured = []
            def capture(module, args):
                h = args[0]
                if h.ndim != 3 or h.shape[0] != 1 or h.shape[1] != batch[1].shape[1]:
                    raise ValueError('Unexpected edited-layer inputs')
                captured.append(h[0, positions.to(h.device)].detach().float().cpu())
            handle = self.layer.register_forward_pre_hook(capture)
            try:
                self.logits(batch)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise ValueError('Edited layer must be reached exactly once')
            selected.extend(captured)
        result = torch.cat(selected)
        if not 1 <= len(result) <= (4 if anchors else 32):
            raise ValueError('Input summary exceeds fixed capacity')
        return result


def context_positions(batch, quota, *, anchors=False):
    """Bounded stratified full-context sampling; no evaluation inputs or RNG."""
    kwargs, labels, (start, end) = batch
    valid = kwargs.get('attention_mask', torch.ones_like(labels))[0].bool().cpu()
    if labels.ndim != 2 or labels.shape[0] != 1 or not 0 <= start < end <= labels.shape[1]:
        raise ValueError('Invalid expanded image span')
    answer = ((labels[0, 1:] != -100) & valid[1:] & valid[:-1]).nonzero().flatten().tolist()
    if not answer or end > answer[0]+1 or not valid[start:end].all():
        raise ValueError('Invalid answer/image layout')
    visual = list(range(start, end))
    prompt = [p for p in range(answer[0]+1) if valid[p] and not start <= p < end]
    chosen = list(dict.fromkeys([answer[0], answer[-1]]))
    if anchors:
        for group in (visual, prompt):
            candidates = [p for p in group if p not in chosen]
            if candidates:chosen.append(candidates[len(candidates)//2])
        return torch.tensor(chosen, dtype=torch.long)
    groups = [[p for p in visual if p not in answer],
              [p for p in prompt if p not in answer], answer]
    counts = [0, 0, len(chosen)]
    if quota < sum(counts):
        raise ValueError('Quota cannot retain answer endpoints')
    # Allocate final stratum counts first; truncating sampled lists biases toward their starts.
    while sum(counts) < quota:
        available = [i for i, group in enumerate(groups) if counts[i] < len(group)]
        if not available:
            break
        i = min(available, key=lambda i: counts[i])
        counts[i] += 1
    chosen = []
    for group, count in zip(groups, counts):
        if count:
            indices = ([len(group)//2] if count == 1 else
                       torch.linspace(0, len(group)-1, count).round().long().tolist())
            chosen.extend(group[j] for j in indices)
    return torch.tensor(sorted(chosen), dtype=torch.long)
