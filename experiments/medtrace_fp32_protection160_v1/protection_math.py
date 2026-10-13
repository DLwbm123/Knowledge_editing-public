"""Full-TT optimizer-coordinate trust ball with a fixed regularized Fisher sketch."""
import math
import torch
RIDGE=1e-4
FRACTION=.9


class Geometry:
    def __init__(self,rows):
        rows=rows.cpu().double()
        assert rows.ndim==2 and torch.isfinite(rows).all()
        _,singular,self.v=torch.linalg.svd(rows,full_matrices=False)
        assert singular[0]>0 and torch.isfinite(singular).all()
        self.largest=float(singular[0].square())
        self.values=(singular/singular[0]).square()
        self.audit=dict(rows=rows.shape[0],dimensions=rows.shape[1],largest_eigenvalue=self.largest,
            ridge_relative=RIDGE,ridge_absolute=RIDGE*self.largest,rank_truncation=False)

    def solve(self,gain,raw):
        gain,raw=gain.cpu().double(),raw.cpu().double()
        radius=float(raw.norm());norm=float(gain.norm());linear=float(gain@raw)
        assert radius>0 and norm>0 and linear>0
        unit=gain/norm;required=FRACTION*linear/(norm*radius)
        assert 0<required<=FRACTION*(1+1e-12)
        coordinate=self.v@unit;perp=unit-self.v.T@coordinate;perp2=float(perp.square().sum())
        def quantities(mu):
            denom=self.values+RIDGE+mu
            dot=float((coordinate.square()/denom).sum())+perp2/(RIDGE+mu)
            square=float((coordinate.square()/denom.square()).sum())+perp2/(RIDGE+mu)**2
            return dot,required*math.sqrt(square)/dot
        mu=0.;dot,length=quantities(mu)
        if length>1:
            lo,hi=0.,1.
            for _ in range(64):
                if quantities(hi)[1]<=1:break
                hi*=2
            else:raise AssertionError('No trust-ball multiplier bracket')
            for _ in range(80):
                mid=(lo+hi)/2
                if quantities(mid)[1]>1:lo=mid
                else:hi=mid
            mu=hi;dot,length=quantities(mu)
        response=self.v.T@(coordinate/(self.values+RIDGE+mu))+perp/(RIDGE+mu)
        value=required*response/dot
        multiplier=required/dot
        applied=self.v.T@(self.values*(self.v@value))+RIDGE*value
        residual=applied+mu*value-multiplier*unit
        assert value.norm()<=1+1e-8 and abs(float(unit@value)-required)<1e-8
        assert float(residual.norm())<1e-7 and abs(mu*(float(value.square().sum())-1))<1e-7
        delta=radius*value
        return delta,dict(RAW_linear_gain=linear,protected_linear_gain=float(gain@delta),required_fraction=FRACTION,
            RAW_coordinate_norm=radius,protected_coordinate_norm=float(delta.norm()),multiplier=mu,
            KKT_residual=float(residual.norm()),normalized_objective=float(value@applied))


def selfcheck():
    for rows,gain,raw in [
        (torch.eye(3),torch.tensor([1.,0.,0.]),torch.tensor([1.,0.,0.])),
        (torch.tensor([[1.,0.,0.]]),torch.tensor([1.,1.,0.]),torch.tensor([.5,.5,0.])),
        (torch.diag(torch.tensor([.001,1.])),torch.tensor([1.,1.]),torch.tensor([.5,.5])),
    ]:
        geometry=Geometry(rows);value,audit=geometry.solve(gain,raw)
        normalized=rows.double().T@rows.double()/geometry.largest+RIDGE*torch.eye(rows.shape[1],dtype=torch.float64)
        assert value.norm()<=raw.double().norm()*(1+1e-8)
        assert torch.allclose(gain.double()@value,FRACTION*(gain.double()@raw.double()),atol=1e-9)
        assert value@normalized@value <= raw.double()@normalized@raw.double()+1e-9
    geometry=Geometry(torch.eye(2))
    value,_=geometry.solve(torch.tensor([1.,0.]),torch.tensor([1.,0.]));assert torch.allclose(value,torch.tensor([.9,0.],dtype=torch.float64))
    try:geometry.solve(torch.ones(2),-torch.ones(2))
    except AssertionError:pass
    else:raise AssertionError('Nonpositive RAW gain accepted')
    return dict(status='PASS',full_and_rank_deficient=True,trust_ball_and_gain=True,convex_objective=True,nonpositive_proposal_rejected=True)


if __name__=='__main__':print(selfcheck())
