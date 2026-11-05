"""Synthetic check: all split means, values and softmax-weighted gradients."""
import torch
from average import partition_means


def check(n):
    torch.manual_seed(n)
    x = torch.rand(2, n, dtype=torch.float64, requires_grad=True)
    p = torch.stack([partition_means(x[i]) for i in range(2)])
    spans = [(0, 1)] if n == 1 else [(a, b) for k in range(1, n) for a,b in ((0, k), (k, n))]
    q = torch.stack([torch.stack([row[a:b].mean() for a,b in spans]) for row in x])
    torch.testing.assert_close(p, q, atol=1e-12, rtol=1e-12)
    def exact(z):
        return (torch.logsumexp(z, dim=0) - torch.log(z.new_tensor(2.))).mean()
    def surrogate(z):
        return (z * torch.softmax(z.detach(), dim=0)).mean(dim=1).sum()
    torch.testing.assert_close(torch.autograd.grad(exact(p), x, retain_graph=True)[0],
                               torch.autograd.grad(surrogate(q), x)[0], atol=1e-12, rtol=1e-12)
    if n in (1, 2):
        midpoint = [(0, n)] if n == 1 else [(0, n//2), (n//2, n)]
        original = torch.stack([torch.stack([r[a:b].mean() for a,b in midpoint]) for r in x])
        torch.testing.assert_close(p, original, atol=1e-12, rtol=1e-12)


for n in (1, 2, 3, 5, 17, 101):
    check(n)
for x in (torch.tensor([]), torch.tensor([float('nan')]), torch.tensor([float('inf')]), torch.ones(2, 2)):
    try:
        partition_means(x)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid input was accepted')
print('synthetic_checks_passed')
