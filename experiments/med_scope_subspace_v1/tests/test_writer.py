import unittest

import torch
from torch import nn

from experiments.med_scope_subspace_v1.train_value import train


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.G1 = nn.Parameter(torch.zeros(6,2))
        self.G2 = nn.Parameter(torch.eye(2))
        self.G3 = nn.Parameter(torch.randn(2,2,2))
        self.G4 = nn.Parameter(torch.randn(2,2,1))

    def parameter_groups(self):
        return [self.G3,self.G4],[self.G1,self.G2]

    def residual(self, x):
        a = (self.G3 @ self.G4[...,0]).flatten(1)
        return (x @ a.T) @ (self.G1 @ self.G2).T


class Hook:
    def __init__(self, layer, expert):
        self.layer = layer
        self.expert = expert

    def attach(self):
        self.handle = self.layer.register_forward_hook(
            lambda _,x,y:y+self.expert.residual(x[0])*self.mask.unsqueeze(-1))

    def set_teacher_routing(self, labels):
        self.mask = torch.zeros_like(labels,dtype=torch.bool)
        self.mask[:, :-1] = labels[:,1:]!=-100

    def detach(self):
        self.handle.remove()


class Runtime:
    eos = 5
    hook_type = Hook

    def __init__(self):
        self.model = nn.Linear(4,6,bias=False).requires_grad_(False)
        self.layer = self.model

    def budget(self):
        pass

    def teacher_batch(self,row,prefix):
        return row['batch']

    def forward(self,batch):
        return self.model(batch['inputs'])


class WriterTests(unittest.TestCase):
    def test_frozen_and_joint_writers(self):
        torch.manual_seed(44)
        runtime = Runtime()
        expert = Expert()
        source = {k:v.clone() for k,v in expert.state_dict().items()}
        events = {k:dict(role=role,batch=dict(inputs=torch.randn(1,5,4),
                      labels=torch.tensor([[-100,-100,2,4,5]])))
                  for k,role in [('n','native'),('p','train_eq'),('f','fit'),('i','train_ind')]}
        schedule = [dict(native='n',views=['p','f','f','f'],negatives=['i'])]*3
        frozen, _ = train(runtime,expert,source,events,schedule,.3,False)
        self.assertTrue(torch.equal(frozen['G3'],source['G3']))
        self.assertTrue(torch.equal(frozen['G4'],source['G4']))
        self.assertGreater(float(frozen['G1'].norm()),0)
        expert = Expert()
        joint, _ = train(runtime,expert,source,events,schedule,.3,True)
        self.assertFalse(torch.equal(joint['G3'],source['G3']))
        self.assertIsNone(runtime.layer.weight.grad)
        self.assertEqual(len(runtime.layer._forward_hooks),0)


if __name__=='__main__':
    unittest.main()
