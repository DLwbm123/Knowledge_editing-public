import io
import unittest
import torch
from components import *
from manifest import query_input

class Writer(nn.Module):
    def __init__(self):
        super().__init__();self.b=nn.Parameter(torch.randn(5,4));self.a=nn.Parameter(torch.randn(4,7));self.epsilon=1e-6
    def factors(self):return self.b,self.a
    def residual(self,x):return (x/(x.square().mean(-1,keepdim=True).sqrt()+self.epsilon)@self.a.T)@self.b.T

class Checks(unittest.TestCase):
    def test_loader_disabled_initialization(self):
        from unittest.mock import patch
        with patch.object(nn.Linear,'reset_parameters',lambda self:None),patch.object(nn.LayerNorm,'reset_parameters',lambda self:None):
            router=LDKERouter(8,4,4);x=torch.randn(2,8)
            self.assertTrue(torch.isfinite(router.objective(x,x+.01,-x)))
            self.assertTrue(torch.isfinite(EvidenceGate()(torch.randn(2,256))).all())
            torch.testing.assert_close(Conditioner()(torch.randn(2,256)),torch.ones(2,4))
    def test_radius(self):
        self.assertEqual(radius(2.,10.,0.),10.);self.assertEqual(radius(2.,10.,1.),2.)
        self.assertAlmostEqual(radius(2.,10.,.2),radius(2.,10.,.8,'paper_equation3'))
        a=torch.zeros(2,3);r=RadiusRouter(a,a+1,a+3);self.assertTrue(r(a)[0].all());self.assertFalse(r(a+100)[0].any())
    def test_evidence_padding(self):
        p=EvidenceProvider();v=torch.randn(2,3,128);q=torch.randn(2,4,128)
        a=p(v,q);vm=torch.tensor([[1,1,1,0]]*2);tm=torch.tensor([[1,1,1,1,0]]*2)
        b=p(torch.cat((v,torch.ones(2,1,128)*99),1),torch.cat((q,torch.ones(2,1,128)*99),1),vm,tm)
        torch.testing.assert_close(a.visual,b.visual);torch.testing.assert_close(a.relation,b.relation)
    def test_condition_lifecycle(self):
        torch.manual_seed(9);w=Writer();e=ConditionedExpert(w);x=torch.randn(2,3,7)
        with e.request(torch.tensor([0.,1.]),torch.ones(2,4)):
            y=e.residual(x);self.assertEqual(y[0].count_nonzero(),0);torch.testing.assert_close(y[1],w.residual(x)[1])
        self.assertIsNone(e.gate)
        with self.assertRaises(RuntimeError):e.residual(x)
        with self.assertRaises(ValueError):
            with e.request(torch.ones(2),torch.ones(2,4)*2):pass
        c=Conditioner();z=torch.randn(2,256);torch.testing.assert_close(c(z),torch.ones(2,4))
        c.net[-1].weight.data.normal_();self.assertFalse(torch.equal(c(z),c(-z)));self.assertTrue(((c(z)>=.5)&(c(z)<=1.5)).all())
        buffer=io.BytesIO();torch.save(c.state_dict(),buffer);buffer.seek(0);cc=Conditioner();cc.load_state_dict(torch.load(buffer,weights_only=True));torch.testing.assert_close(c(z),cc(z),rtol=0,atol=0)
        try:
            with e.request(torch.ones(2),torch.ones(2,4)):raise RuntimeError('abort')
        except RuntimeError:pass
        self.assertIsNone(e.gate)
    def test_ldke_and_localizer(self):
        torch.manual_seed(2);r=LDKERouter(8,4,4);a=torch.randn(2,8);p=a+.01;n=-a
        loss=r.objective(a,p,n);loss.backward();self.assertTrue(all(torch.isfinite(x.grad).all() for x in r.parameters()))
        self.assertTrue(r.route(a,a)[0].all())
        pre={0:torch.zeros(1,2),1:torch.zeros(1,2)};post={0:torch.tensor([[9.,0.]]),1:torch.tensor([[1.,0.]])}
        k,s=localize(pre,post,nn.Identity(),nn.Identity(),0,1);self.assertEqual(k,1);self.assertEqual(list(s),[1])
    def test_labels_and_medical_guard(self):
        self.assertEqual(query_input(dict(image='x',question='q',answer='secret',role='test')),dict(image='x',question='q'))
        with self.assertRaises(ValueError):evidence_increment_loss(1.,0.,0.,0.,verified_medical_pair=False)
        logits=torch.randn(1,4,9);labels=torch.tensor([[-100,2,3,8]])
        s=target_score(logits,labels,8);want=torch.stack((logits[0,0].log_softmax(0)[2],logits[0,1].log_softmax(0)[3])).mean();torch.testing.assert_close(s,want)

if __name__=='__main__':unittest.main()
