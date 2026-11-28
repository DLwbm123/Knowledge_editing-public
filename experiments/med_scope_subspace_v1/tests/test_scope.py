import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.med_scope_subspace_v1.evaluate import scope_metrics, routing_report
from experiments.med_scope_subspace_v1.features import load_manifest, normalized_positions
from experiments.med_scope_subspace_v1.router import ScopeRouter
from experiments.med_scope_subspace_v1.scope_solver import (
    BackgroundInverse, Moment, fit, g3_step, generalized_top, input_matrix, objective,
    orthogonalize, row_basis, unrestricted,
)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def randn(self,*shape):
        return torch.randn(*shape,dtype=torch.float64)

    def test_exact_tt_projection(self):
        g3,g4 = self.randn(4,5,3),self.randn(3,6)
        x = self.randn(20,30)
        p = torch.kron(torch.eye(5,dtype=torch.float64),g4.T.contiguous())
        a = input_matrix(g3,g4)
        torch.testing.assert_close(p @ g3.flatten(1).T,a.T)
        torch.testing.assert_close((x.reshape(-1,5,6) @ g4.T).flatten(1),x @ p)

    def test_weighting_and_gauge(self):
        p = Moment.from_queries([self.randn(2,30),self.randn(6,30)])
        z = torch.eye(30,dtype=torch.float64)
        expected = sum(x.T @ x / len(x) for x in [p.rows[:2]*2,p.rows[2:]*(12**.5)])/2
        torch.testing.assert_close(p.gram(z),expected)
        n = [(1.,Moment.from_queries([self.randn(50,30)]))]
        g3,g4 = self.randn(4,5,3),self.randn(3,6)
        a = input_matrix(g3,g4)
        m = self.randn(4,4)
        t = self.randn(3,3)
        torch.testing.assert_close(objective(m@a,p,n,.1),objective(a,p,n,.1))
        torch.testing.assert_close(input_matrix(g3@t,torch.linalg.solve(t,g4)),a)

    def test_g3_exact_and_alternating(self):
        g4 = self.randn(3,6)
        p = Moment.from_queries([self.randn(40,30)])
        n = [(1.,Moment.from_queries([self.randn(80,30)]))]
        vals,g3 = g3_step(g4,p,n,.1,5)
        a = input_matrix(g3,g4)
        torch.testing.assert_close(objective(a,p,n,.1),vals.sum())
        z = (p.rows.reshape(-1,5,6) @ g4.T).flatten(1)
        w = (n[0][1].rows.reshape(-1,5,6) @ g4.T).flatten(1)
        noise = w.T@w + .1*torch.kron(torch.eye(5),(g4@g4.T).contiguous())
        vectors = g3.flatten(1).T
        torch.testing.assert_close((z.T@z)@vectors,(noise@vectors)*vals)
        out = fit(p,n,.1,g4,modes=5,rounds=2,steps=5)
        self.assertTrue(all(b>=a-1e-8*abs(a) for a,b in zip(out['trace'],out['trace'][1:])))
        torch.testing.assert_close(torch.tensor(out['objective']),torch.tensor(sum(out['eigenvalues'])))

    def test_row_orthogonalization_preserves_subspace(self):
        g3,g4 = self.randn(4,5,3),self.randn(3,6)
        old = input_matrix(g3,g4)
        new = input_matrix(orthogonalize(g3,g4),g4)
        torch.testing.assert_close(new@new.T,torch.eye(4,dtype=new.dtype))
        torch.testing.assert_close(torch.linalg.svdvals(new@row_basis(old).T),torch.ones(4,dtype=new.dtype))

    def test_unrestricted_woodbury_both_background_sizes(self):
        p = Moment.from_queries([self.randn(12,30)])
        n = [(2.,Moment.from_queries([self.randn(19,30)]))]
        for rows in (8,40):
            bg = Moment.from_queries([self.randn(rows,30)])
            inverse = BackgroundInverse(bg,.6,.2,30)
            vals,u = unrestricted(p,n,inverse)
            sn = .2 * torch.eye(30,dtype=p.rows.dtype) + .6*bg.rows.T@bg.rows + 2*n[0][1].rows.T@n[0][1].rows
            expected,_ = generalized_top(p.rows.T@p.rows,sn,4)
            torch.testing.assert_close(vals,expected)
            torch.testing.assert_close(objective(u,p,n+[(.6,bg)],.2),expected.sum())

    def test_conformal_empirical_rank_and_small_support(self):
        basis = {1:torch.tensor([[1.,0.]])}
        router = ScopeRouter(basis,{1:torch.tensor([[1.,0.]])})
        features = torch.tensor([[float(x),1.] for x in range(20)])
        self.assertEqual(router.calibrate([features[0]],.05),float('inf'))
        positives = 0
        for leave in range(len(features)):
            rows = [x for i,x in enumerate(features) if i!=leave]
            router.calibrate(rows,.10)
            positives += router.route(features[leave])[0][0] is not None
        self.assertLessEqual(positives/len(features),.10)

    def test_teacher_positions_and_zero_residual_identity(self):
        labels = torch.tensor([[-100,-100,-100,5,6]])
        x = self.randn(1,5,30)
        feats = normalized_positions(x,labels)
        hook_mask = torch.zeros_like(labels,dtype=torch.bool)
        hook_mask[:, :-1] = labels[:,1:] != -100
        positions = torch.where(hook_mask[0])[0]
        torch.testing.assert_close(positions,torch.tensor([2,3]))
        torch.testing.assert_close(x[0,hook_mask[0]],x[0,2:4])
        target = x[0,2:4].float()
        torch.testing.assert_close(feats,target/(target.square().mean(-1,keepdim=True).sqrt()+1e-6))
        a,b = self.randn(4,30),torch.zeros(7,4,dtype=x.dtype)
        base = self.randn(5,7)
        torch.testing.assert_close(base[2:4] + (feats.double() @ a.T) @ b.T,base[2:4],rtol=0,atol=0)

    def test_leakage_contract(self):
        base = dict(id='1',owner=1,role='train_eq',source_sha256='source',image_sha256='source',
                    question='same question',answer='a',family='train')
        plan = dict(protected_image_sha256=['protected'],train_families=['train'],
                    holdout_families=['holdout'],rows=[base])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'m.json'
            p.write_text(json.dumps(plan))
            self.assertEqual(load_manifest(p)['rows'],[base])
            indirect = dict(base,id='3',role='official_ind',question='official indirect')
            p.write_text(json.dumps(dict(plan,rows=[indirect])))
            self.assertEqual(load_manifest(p)['rows'],[indirect])
            negative = dict(base,id='4',role='cal_ind',question='cal negative',relation='independent')
            p.write_text(json.dumps(dict(plan,rows=[negative])))
            with self.assertRaises(ValueError):
                load_manifest(p)
            p.write_text(json.dumps(dict(plan,rows=[negative],calibration_owners=[1])))
            self.assertEqual(load_manifest(p)['calibration_owners'],[1])
            for other in (dict(base,id='2',role='hold_pos',family='holdout'),
                          dict(base,id='2',source_sha256='protected'),
                          dict(base,id='2',family='holdout')):
                p.write_text(json.dumps(dict(plan,rows=[base,other])))
                with self.assertRaises(ValueError):
                    load_manifest(p)

    def test_scope_and_unknown_counts(self):
        scores = [{'role':'hold_pos','score':2.},{'role':'hold_ind','score':1.},
                  {'role':'hold_vis','score':2.}]
        self.assertEqual(scope_metrics(scores),{'delta':.5,'delta_ind':1.,'delta_vis':0.,'auc_ind':1.,'auc_vis':0.})
        val = routing_report([dict(id='a',owner=1,role='official_ind',image='i',
                                   route='P4',core=None,selected=1)])
        self.assertEqual(val['groups']['P4']['official_ind']['core_unknown'],1)


if __name__=='__main__':
    unittest.main()
