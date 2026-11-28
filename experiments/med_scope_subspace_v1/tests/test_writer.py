import unittest

import torch
from torch import nn

from experiments.med_scope_subspace_v1.dense_writer import DenseReadIn, principal_angles
from experiments.med_scope_subspace_v1.s1_dense import calibration_selection
from experiments.med_scope_subspace_v1.s1_dense_report import judge
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

    def test_dense_writer_trains_and_preserves_frozen_readin(self):
        from types import SimpleNamespace

        torch.manual_seed(44)
        a = torch.randn(2,4)
        tt = SimpleNamespace(G1=torch.zeros(1,2,2),G2=torch.randn(2,3,2),
                             epsilon=1e-6,factors=lambda: (None,a))
        runtime = Runtime()
        fit = dict(inputs=torch.randn(1,5,4),labels=torch.tensor([[-100,-100,2,4,5]]))
        events = {k:dict(role=role,batch=fit) for k,role in [('n','native'),('f','fit'),('i','train_ind')]}
        steps = [dict(native='n',views=['f']*4,negatives=['i'])]*3
        basis = torch.linalg.qr(a.double().T,mode='reduced').Q.T
        with torch.no_grad():
            self.assertEqual(DenseReadIn(tt).residual(fit['inputs']).abs().max().item(),0)
        frozen,_ = train(runtime,DenseReadIn(tt),dict(A=a),events,steps,.3,False)
        joint,trace = train(runtime,DenseReadIn(tt),dict(A=a),events,steps,.3,True,
                            read_lr=.0002,angle_basis=basis)
        self.assertTrue(torch.equal(frozen['A'],a))
        self.assertFalse(torch.equal(joint['A'],a))
        self.assertLess(max(principal_angles(a,basis)),1e-5)
        self.assertEqual(len(trace[-1]['ku_angles_deg']),2)

    def test_calibration_uses_first_joint_before_single(self):
        def candidate(beta, frozen, joint):
            return dict(beta=beta,medians={'V-U':frozen,'V-U-ft':joint})

        partial = candidate(.3,0.,2.)
        joint = candidate(1.,0.,0.)
        fail = candidate(3.,2.,2.)
        self.assertEqual(calibration_selection([partial,joint,fail],1.)['beta'],1.)
        self.assertEqual(calibration_selection([partial,fail],1.)['ku_arms'],['V-U'])
        self.assertEqual(calibration_selection([fail],1.)['training_arms'],[])

    def test_visual_floor_still_requires_kl_superiority(self):
        from collections import Counter

        p4 = {'hold_vis':dict(target_exact=1,exact_support=15)}
        source = {'native':dict(core_correct=30,medians={'answer_nll':.01}),
                  'fit':dict(core_correct=120,medians={'answer_nll':.01})}
        effects = {'exact':dict(ci95=[-.03,.04]),'base_kl':dict(ci95=[-.04,-.01])}
        report = {'groups':{'P4':p4,'V-U':source},'paired':{
            'V-U vs P4 – hold_vis':dict(effects=effects),
            'V-U vs P4 – hold_ind':dict(effects=effects),
            'V-U vs P4 – official_ind':dict(effects={'core':dict(ci95=[-.01,.07])},unknown=0)}}
        self.assertTrue(judge('V-U',report,Counter())['visual_superior'])
        effects['base_kl']['ci95'][1] = 0
        self.assertFalse(judge('V-U',report,Counter())['visual_superior'])
        effects['base_kl']['ci95'][1] = -.01
        p4['hold_vis']['target_exact'] = 2
        self.assertFalse(judge('V-U',report,Counter())['visual_superior'])


if __name__=='__main__':
    unittest.main()
