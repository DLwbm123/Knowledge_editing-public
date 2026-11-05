"""All-boundary mean span losses for the preregistered two-view objective."""
import torch


def partition_means(values):
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError('expected nonempty finite token losses')
    n = values.numel()
    if n == 1:
        return values
    prefix = torch.cat((values.new_zeros(1), values.cumsum(dim=0)))
    cuts = torch.arange(1, n, device=values.device)
    left = prefix[cuts] / cuts
    right = (prefix[n] - prefix[cuts]) / (n - cuts)
    return torch.stack((left, right), dim=1).reshape(-1)
