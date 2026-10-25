"""Exact rank-one form of the official MEMIT matrix solve.

MEMIT: KMeng01/memit @ 80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b.
The fact batch has one averaged key. Already completed AlphaEdit is reused;
this module neither implements nor launches another AlphaEdit experiment.
"""
import torch


def memit_key(inverse, key):
    value = inverse @ key
    return value / (1. + torch.dot(key, value))


def selfcheck():
    rng = torch.Generator().manual_seed(13120261025)
    x = torch.randn(11, 16, dtype=torch.float64, generator=rng)
    covariance = x @ x.T + torch.eye(11, dtype=x.dtype)
    key = torch.randn(11, dtype=x.dtype, generator=rng)
    inverse = torch.linalg.inv(15000 * covariance)
    expected = torch.linalg.solve(15000 * covariance + torch.outer(key, key), key)
    assert torch.allclose(memit_key(inverse, key), expected, atol=1e-12, rtol=1e-10)
    return dict(status='PASS', MEMIT_dense_solve_equivalent=True,
        tolerance=dict(atol=1e-12, rtol=1e-10))


if __name__ == '__main__':
    print(selfcheck())
