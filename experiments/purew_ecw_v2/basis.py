import torch

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
