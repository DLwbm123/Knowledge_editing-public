import torch
from .algebra import update


def main():
    torch.manual_seed(19)
    dtype=torch.float64
    u=torch.linalg.qr(torch.randn(17,9,dtype=dtype)).Q
    p=u@u.T;k=torch.randn(17,dtype=dtype);r=torch.randn(7,dtype=dtype)
    for n in (0,1,8,25):
        h=torch.randn(17,n,dtype=dtype)
        actual=update(k,r,u,h,10)
        expected=torch.linalg.solve(p@(torch.outer(k,k)+h@h.T)+10*torch.eye(17,dtype=dtype),
                                    p@torch.outer(k,r)).T
        assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
        assert torch.allclose(actual@(torch.eye(17,dtype=dtype)-p),torch.zeros(7,17,dtype=dtype),atol=1e-12)
    assert not update(k,r,u[:,:0],torch.zeros(17,3,dtype=dtype)).any()
    print('PASS: empty/nonempty/overcomplete history solve agrees with official dense AlphaEdit and remains in P')


if __name__=='__main__':main()
