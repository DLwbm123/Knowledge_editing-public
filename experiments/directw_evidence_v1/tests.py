"""20 requirement groups with real FP64 CPU calculations and independent oracles.

Run from repository root using the neutral stdin entry in README.md
The recorded command uses a neutral stdin runner to keep process argv private.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import torch
from scipy.optimize import minimize
from .numerics import FrozenGGN, cg, hard_qp_reference, position_weights, protection_kl, solve_step
from .editor import (MatrixRuntime, ProtectionGroup, Constraint, EditConfig, build_constraints,
                     edit_one, geometry, mean_answer_logprob, BRANCHES)
from .fixture import TinyModel
from .contracts import (audit_data, method_eligibility, validate_cache, HistoryMemory, MemoryEntry,
                        cleanup, AttemptLedger, digest, enforce_storage)
from .gate import require_external_approval, default_config, gated_load, cli, BINDINGS, advance_state
from .export import export_native, load_original_matrix, verify_clean_reload

SEED = 20260930
METRICS: dict[str, dict] = {}
torch.set_num_threads(1)


def binding() -> dict[str, str]:
    return {k: "fixture" for k in ("model", "weight_version", "input", "prefix", "mask", "dtype", "backend", "config", "teacher_version")}


def row(role: str = "EDIT_FIT", **extra) -> dict:
    return dict(id="a", source_group="s", patient_group=None, image_hash="i", question_hash="q", answer_hash="a",
                fact_family="f", role=role, permission_basis="synthetic fixture", available_at_edit_index=0,
                scope_evidence="VERIFIED_OUT_OF_SCOPE", annotation_source="synthetic", ever_developed=False,
                ever_scored=False, teacher_reference=None, source_split="train", fit_permission_verified=True, **extra)


class Checks(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(SEED)
        self.dtype = torch.float64
        self.w = torch.randn(3, 4, dtype=self.dtype) * .15
        self.x = torch.randn(2, 3, 4, dtype=self.dtype)
        self.logits = lambda w: torch.tanh(self.x @ w.T)
        self.mask = torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
        self.weights = position_weights(self.mask, ("a","b"))
        self.f = FrozenGGN(self.logits, self.w, self.weights)

    def metric(self, **values) -> None:
        METRICS[self._testMethodName] = values

    def dense(self):
        jac = torch.func.jacrev(self.logits)(self.w).reshape(-1, self.w.numel())
        p = self.logits(self.w).softmax(-1).reshape(-1,3)
        blocks = [float(weight) * (torch.diag(prob)-prob[:,None]*prob[None,:])
                  for prob, weight in zip(p, self.weights.flatten())]
        return jac.T @ torch.block_diag(*blocks) @ jac

    def test_01_explicit_ggn(self):
        v = torch.randn_like(self.w)
        error = float((self.f(v).flatten() - self.dense() @ v.flatten()).abs().max())
        self.assertLess(error, 1e-8); self.metric(max_error=error, atol=1e-8, rtol=1e-6)

    def test_02_symmetry_psd(self):
        u,v = torch.randn_like(self.w),torch.randn_like(self.w)
        error = abs(float((u*self.f(v)).sum()-(v*self.f(u)).sum()))
        minimum = float(torch.linalg.eigvalsh(self.dense()).min())
        self.assertLess(error,1e-8); self.assertGreaterEqual(minimum,-1e-8)
        self.assertGreater(float((v*(self.f(v)+.1*v)).sum()),0)
        self.metric(max_error=error, minimum_eigenvalue=minimum, damping=.1)

    def test_03_zero_gradient_nonzero_fisher(self):
        anchor = self.logits(self.w).softmax(-1).detach()
        w = self.w.clone().requires_grad_(True)
        grad = torch.autograd.grad(protection_kl(self.logits(w),anchor,self.weights),w)[0]
        curvature = float(self.f(torch.ones_like(w)).norm())
        self.assertLess(float(grad.norm()),1e-8); self.assertGreater(curvature,1e-5)
        self.metric(max_error=float(grad.norm()), fisher_norm=curvature)

    def test_04_finite_difference(self):
        v = torch.randn_like(self.w); v /= v.norm()
        anchor = self.logits(self.w).softmax(-1).detach()
        predicted = float(.5*(v*self.f(v)).sum()); errors=[]
        for epsilon in (1e-2,3e-3,1e-3,3e-4):
            actual = float(protection_kl(self.logits(self.w+epsilon*v),anchor,self.weights))/epsilon**2
            errors.append(abs(actual-predicted))
        self.assertLess(errors[-1],1e-4)
        self.assertLess(errors[-1],errors[0]); self.metric(max_error=errors[-1], errors=errors, curvature=predicted, atol=1e-4)

    def test_05_preservation_linear_term(self):
        anchor = self.logits(self.w+.2).softmax(-1).detach()
        w=self.w.clone().requires_grad_(True)
        loss=protection_kl(self.logits(w),anchor,self.weights)
        g=torch.autograd.grad(loss,w)[0]
        after=protection_kl(self.logits(w-.1*g),anchor,self.weights)
        self.assertGreater(float(g.norm()),1e-5); self.assertLess(float(after),float(loss))
        a=torch.zeros(1,w.numel(),dtype=self.dtype); b=torch.tensor([-1.],dtype=self.dtype)
        step=solve_step(lambda v:v,a,b,g,nu=10)
        self.assertTrue(torch.allclose(step.direction,-g))
        self.metric(max_error=float((step.direction+g).abs().max()), before=float(loss),after=float(after),gradient_norm=float(g.norm()))

    def test_06_cg_and_failure(self):
        r=torch.randn(8,8,dtype=self.dtype); q=r.T@r+torch.eye(8,dtype=self.dtype)
        b=torch.randn(8,dtype=self.dtype)
        solution=cg(lambda v:q@v,b,max_iter=16)
        error=float((solution.value-torch.linalg.solve(q,b)).abs().max())
        self.assertEqual(solution.status,"CONVERGED"); self.assertLess(error,1e-8)
        self.assertEqual(cg(lambda v:q@v,b,max_iter=1).status,"NOT_CONVERGED")
        calls=[0]
        def changing(v):
            calls[0]+=1; return (q+calls[0]*torch.eye(8,dtype=self.dtype))@v
        self.assertEqual(cg(changing,b).status,"NONSTATIONARY")
        self.metric(max_error=error,residual=solution.relative_residual,iterations=solution.iterations)

    def test_07_hard_soft_independent_primal(self):
        q=torch.tensor([[2.,.3],[.3,1.]],dtype=self.dtype)
        a=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]],dtype=self.dtype)
        b=torch.tensor([1.,1.,.2],dtype=self.dtype); g=torch.tensor([.2,-.4],dtype=self.dtype); nu=7.
        status,_=hard_qp_reference(q,a,b,g); self.assertEqual(status,"INFEASIBLE")
        status,_=hard_qp_reference(q,a[[0,2]],b[[0,2]],g); self.assertEqual(status,"FEASIBLE")
        status,_=hard_qp_reference(q,torch.stack((a[0],a[0])),torch.tensor([1.,1.],dtype=self.dtype),g)
        self.assertEqual(status,"FEASIBLE")
        result=solve_step(lambda v:q@v,a,b,g,nu=nu,cg_rtol=1e-12,dual_tol=1e-10)
        self.assertEqual(result.status,"CONVERGED")
        import numpy as np
        Q,A,B,G=q.numpy(),a.numpy(),b.numpy(),g.numpy()
        objective=lambda z: .5*z[:2]@Q@z[:2]+G@z[:2]+.5*nu*(z[2:]@z[2:])
        jac=lambda z: np.r_[Q@z[:2]+G,nu*z[2:]]
        oracle=minimize(objective,np.r_[np.zeros(2),np.maximum(B,0)],jac=jac,method="SLSQP",
            bounds=[(None,None)]*2+[(0,None)]*3,
            constraints=[dict(type="ineq",fun=lambda z:A@z[:2]+z[2:]-B,jac=lambda z:np.c_[A,np.eye(3)])],
            options=dict(ftol=1e-12,maxiter=2000))
        self.assertTrue(oracle.success,oracle.message)
        error=float(np.max(np.abs(np.r_[result.direction.numpy(),result.slack.numpy()]-oracle.x)))
        self.assertLess(error,1e-6); self.assertLess(max(result.kkt.values()),1e-7)
        nonconverged=solve_step(lambda v:q@v,a,b,g,nu=nu,cg_max_iter=1)
        self.assertEqual(nonconverged.status,"NOT_CONVERGED")
        self.metric(max_error=error,kkt=result.kkt,independent_oracle="SciPy SLSQP primal d,xi",atol=1e-6)

    def runtime_case(self, protection_same=False):
        model=TinyModel(); runtime=MatrixRuntime(model,"mlp.down_proj.weight")
        target=dict(tokens=torch.zeros(1,2,2,dtype=self.dtype),image=torch.tensor([[1.,0.]],dtype=self.dtype))
        protect=target if protection_same else dict(tokens=target["tokens"],image=torch.tensor([[0.,1.]],dtype=self.dtype))
        logits=lambda w:runtime.logits(w,protect)
        mask=torch.ones(1,2,dtype=torch.bool)
        group=ProtectionGroup("bg",logits,logits(runtime.weight).softmax(-1).detach(),mask,("s",),1.,0. if protection_same else 1.,binding(),runtime.capture_input(protect))
        constraint=Constraint("edit","EDIT_FIT",lambda w:runtime.logits(w,target)[0,0,0],.1)
        config=EditConfig(.1,1000.,.05,2.,1.,max_steps=8,cg_rtol=1e-10,tolerance=1e-6)
        return runtime,target,[constraint],[group],config

    def test_08_clip_backtrack_all_branches(self):
        successful=[]
        for branch in BRANCHES[:-1]:
            r,inputs,cs,gs,cfg=self.runtime_case()
            if branch=="W_FT": cfg.ft_lr=10.; cfg.tau=1e-8
            result=edit_one(r,cs,gs,cfg,branch)
            self.assertEqual(result.status,"ACCEPTED",(branch,result))
            self.assertTrue(any(a.get("clipped") for a in result.attempts))
            self.assertLess(max(result.final_violations.values()),cfg.tolerance)
            successful.append(branch)
        r,_,cs,gs,cfg=self.runtime_case(True)
        result=edit_one(r,cs,gs,cfg,"W_FUNCTIONAL_QP")
        self.assertEqual(result.status,"BACKTRACK_REJECTED"); self.assertTrue(result.rollback)
        self.assertEqual(len(result.attempts),len(cfg.factors))
        self.assertTrue(torch.equal(r.weight,r.base))
        self.metric(branches=successful, rejection_attempts=len(result.attempts),max_error=max(result.final_violations.values()), semantic_tolerance=cfg.tolerance)

    def test_09_visual_both_endpoints(self):
        r,inputs,_,gs,cfg=self.runtime_case()
        with torch.no_grad():r.model.mlp.down_proj.bias[1]=.2
        r=MatrixRuntime(r.model,"mlp.down_proj.weight")
        plus=inputs; minus=dict(tokens=inputs["tokens"],image=-inputs["image"])
        neg_logits=r.bind_inputs(minus)
        gs=[ProtectionGroup("negative",neg_logits,neg_logits(r.weight).softmax(-1).detach(),
            torch.ones(1,2,dtype=torch.bool),("s",),1.,.02,binding(),r.capture_input(minus))]
        self.assertEqual(int(r.normal_logits(minus)[0,0].argmax()),1)
        def score(w,i,y):
            return r.logits(w,i).log_softmax(-1)[0,0,y]
        p=lambda w:score(w,plus,0)-score(w,plus,1)
        n=lambda w:score(w,minus,1)-score(w,minus,0)
        pair=dict(id="pair",verified=True,scope="OUT_OF_SCOPE",negative_base_correct=True,
                  plus_margin=p,minus_margin=n,endpoint_margin=.1,visual_margin=.3,negative_protection_group="negative")
        cs=build_constraints([], [pair], "W_EVIDENCE_QP")
        no_cross=build_constraints([], [pair], "W_FUNCTIONAL_QP")
        self.assertEqual(len(cs),3); self.assertEqual(len(no_cross),2)
        # Deliberate high-slack-penalty feasibility fixture, never a native proposal.
        cfg.max_steps=12;cfg.nu=1e8
        result=edit_one(r,cs,gs,cfg,"W_EVIDENCE_QP")
        self.assertEqual(result.status,"ACCEPTED",result)
        self.assertTrue(all(float(c.score(r.weight))>=c.threshold-cfg.tolerance for c in cs))
        self.assertLessEqual(float(gs[0].loss(r.weight)),gs[0].budget+1e-12)
        before=r.weight.detach().clone()
        bad_weight=before.clone();bad_weight[0,2]+=2.;bad_weight[2,2]-=2.
        old_correct=float(score(before,minus,1));bad_correct=float(score(bad_weight,minus,1))
        self.assertLess(bad_correct,old_correct)
        self.assertGreater(float(cs[2].score(bad_weight)),float(cs[2].score(before)))
        self.assertGreater(float(gs[0].loss(bad_weight)),gs[0].budget)
        force_wrong=Constraint("wrong_negative","EDIT_FIT",lambda w:r.logits(w,minus)[0,0,2],.8)
        blocked=edit_one(r,[*cs,force_wrong],gs,cfg,"W_EVIDENCE_QP")
        self.assertNotEqual(blocked.status,"ACCEPTED");self.assertTrue(torch.equal(before,r.weight))
        # Algebraic adversarial scores: cross=.5 passes while positive endpoint=-.1 fails.
        bad=build_constraints([], [dict(pair,plus_margin=lambda w:w.sum()*0-.1,minus_margin=lambda w:w.sum()*0+.6)],"W_EVIDENCE_QP")
        self.assertGreater(float(bad[2].score(r.weight)),bad[2].threshold)
        self.assertLess(float(bad[0].score(r.weight)),bad[0].threshold)
        before=r.weight.detach().clone()
        rejected=edit_one(r,bad,gs,cfg,"W_EVIDENCE_QP")
        self.assertNotEqual(rejected.status,"ACCEPTED"); self.assertTrue(torch.equal(before,r.weight))
        self.metric(max_error=max(result.final_violations.values()),two_endpoint_guard=rejected.status,
                    cross_redundant=False, note="cross=.3 exceeds two .1 endpoint margins; not redundant")

    def test_10_image_diagnostics_and_linear_bound(self):
        r,inputs,_,_,_=self.runtime_case()
        neg=dict(tokens=inputs["tokens"],image=-inputs["image"])
        p,n=r.capture_input(inputs),r.capture_input(neg)
        metrics=geometry(p,n,torch.ones_like(r.weight),-torch.ones_like(r.weight))
        zero=dict(tokens=inputs["tokens"],image=torch.zeros_like(inputs["image"]))
        self.assertGreater(metrics["distance"],0); self.assertAlmostEqual(metrics["gradient_cosine"],-1.,places=12)
        self.assertEqual(float(r.capture_input(zero).norm()),0.)
        self.assertEqual(float((p-r.capture_input(neg)).norm()),metrics["distance"])
        minimum_gap=float("inf")
        for _ in range(100):
            dw=torch.randn(3,4,dtype=self.dtype); k1,k2=torch.randn(4,dtype=self.dtype),torch.randn(4,dtype=self.dtype)
            residual=torch.randn(3,dtype=self.dtype); rho=float(torch.linalg.matrix_norm(dw,ord=2))
            lhs=float((dw@k1-residual).norm()+(dw@k2).norm()); rhs=max(0.,float(residual.norm())-rho*float((k1-k2).norm()))
            minimum_gap=min(minimum_gap,lhs-rhs); self.assertGreaterEqual(lhs+1e-8,rhs)
        self.metric(max_error=0., geometry=metrics,minimum_bound_gap=minimum_gap)

    def test_11_predictor_mask_shift_eos(self):
        logits=torch.randn(2,7,5,dtype=self.dtype)
        labels=torch.tensor([[-100,-100,-100,1,2,4,-100],[-100,-100,3,4,-100,-100,-100]])
        score=mean_answer_logprob(logits,labels,eos_id=4)
        l=logits.log_softmax(-1)
        expected=torch.stack(((l[0,2,1]+l[0,3,2])/2,l[1,1,3]))
        error=float((score-expected).abs().max()); self.assertLess(error,1e-8)
        with_eos=mean_answer_logprob(logits,labels,eos_id=4,include_eos=True)
        self.assertTrue(torch.allclose(with_eos[1],(l[1,1,3]+l[1,2,4])/2))
        changed=logits.clone(); changed[0,0]=100.;changed[0,-1]=-50.
        self.assertTrue(torch.equal(score,mean_answer_logprob(changed,labels,eos_id=4)))
        with self.assertRaises(ValueError):mean_answer_logprob(logits,torch.full_like(labels,-100),eos_id=4)
        self.metric(max_error=error,eos_policy="excluded content, separate optional EOS",prefix_positions=3)

    def test_12_data_rejections(self):
        cases=[row(evaluation_origin=True),row(available_override=True)]
        cases[1]["available_at_edit_index"]=3
        cases += [row("VIS_PAIR_FIT",pair={}),row("PAST_EDIT_MEMORY",corrected_history=True,prefix_hash="p")]
        cases[-1]["teacher_reference"]={"model":"base","kind":"BASE","prefix_hash":"p"}
        cases.append(row(teacher_reference_override=True)); cases[-1]["teacher_reference"]={"model":"wrong"}
        reasons=[]
        for entry in cases:
            audited=audit_data([entry],edit_index=0,test_ids={"a"} if entry.get("evaluation_origin") else set(),expected_model="base")
            self.assertEqual(audited["status"],"FAIL"); reasons.extend(f["reason"] for f in audited["findings"])
        self.metric(rejected_reasons=reasons,max_error=0.)

    def test_13_missing_visual_branch_only(self):
        rows=[row(role) for role in ("EDIT_FIT","GEN_FIT","PROTECT_BG_FIT","PROTECT_NEAR_FIT")]
        for i,r in enumerate(rows):r.update(id=str(i),question_hash=str(i))
        audited=audit_data(rows,edit_index=0,test_ids=set(),expected_model="base")
        eligible=method_eligibility(audited)
        self.assertEqual(eligible["W_EVIDENCE_QP"],"BLOCKED")
        self.assertEqual(eligible["W_FUNCTIONAL_QP"],"PENDING_NATIVE_CHECK")
        # A confirmation edit's original legal support and same-image paraphrase
        # need not be from different images; development cohorts MUST be separate.
        fit=row(selection_cohort="CONFIRM",benchmark_same_edit_sharing_verified=True)
        probe=row("TEST_CONFIRM",selection_cohort="CONFIRM")
        probe.update(id="probe",question_hash="different-paraphrase")
        legal=audit_data([fit,probe],edit_index=0,test_ids={"probe"},expected_model="base")
        self.assertEqual(legal["status"],"PASS")
        fit["selection_cohort"]="DEV"
        leaked=audit_data([fit,probe],edit_index=0,test_ids={"probe"},expected_model="base")
        self.assertEqual(leaked["status"],"FAIL")
        self.metric(branches=eligible,max_error=0.)

    def test_14_only_original_matrix(self):
        r,inputs,cs,gs,cfg=self.runtime_case()
        before={k:v.clone() for k,v in r.model.state_dict().items()}
        result=edit_one(r,cs,gs,cfg,"W_FUNCTIONAL_QP")
        self.assertEqual(result.status,"ACCEPTED")
        changed=[k for k,v in r.model.state_dict().items() if not torch.equal(v,before[k])]
        self.assertEqual(changed,["mlp.down_proj.weight"])
        self.assertEqual(set(before),set(r.model.state_dict()))
        self.assertEqual(r.hooks(),(0,)*len(list(r.model.modules())))
        evil,_,ecs,egs,ecfg=self.runtime_case()
        original=evil.normal_logits
        def mutate(inputs):
            with torch.no_grad():evil.model.fixed.add_(1.)
            return original(inputs)
        evil.normal_logits=mutate
        denied=edit_one(evil,ecs,egs,ecfg,"W_FUNCTIONAL_QP")
        self.assertEqual(denied.status,"EXCEPTION_ROLLED_BACK")
        self.assertEqual(float(evil.model.fixed),1.)
        self.metric(changed=changed,new_deployment_parameters=0,max_error=0.)

    def test_15_native_export_clean_process(self):
        r,inputs,cs,gs,cfg=self.runtime_case(); edit_one(r,cs,gs,cfg,"W_FUNCTIONAL_QP")
        functional=r.logits(r.weight,inputs); normal=r.normal_logits(inputs)
        error=float((functional-normal).abs().max());self.assertLess(error,1e-8)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"export";manifest=export_native(r,path,base_binding="b"*64)
            clean=TinyModel();load_original_matrix(clean,path,base_binding="b"*64)
            self.assertEqual(verify_clean_reload(r.model,clean,inputs)["status"],"PASS")
            self.assertTrue(torch.equal(r.model.generate(**inputs),clean.generate(**inputs)))
            # Standalone process imports only torch + synthetic model, NOT editor/export.
            script='''import json,sys,torch
from pathlib import Path
from experiments.directw_evidence_v1.fixture import TinyModel
p=Path(sys.stdin.readline().strip()); m=json.loads((p/'manifest.json').read_text())
model=TinyModel(); parameter=dict(model.named_parameters())[m['path']]
with torch.no_grad(): parameter.copy_(torch.load(p/'matrix.pt',weights_only=True))
inputs=dict(tokens=torch.zeros(1,2,2,dtype=torch.float64),image=torch.tensor([[1.,0.]],dtype=torch.float64))
assert not any(k.endswith('.editor') for k in sys.modules)
print(json.dumps(dict(logits=model(**inputs).tolist(),generation=model.generate(**inputs).tolist(),keys=list(model.state_dict()))))
'''
            # Neutral entrypoint outside the checkout; no method names in argv.
            runner=Path(directory)/"a.py";runner.write_text(script)
            child=subprocess.run([sys.executable,str(runner)],input=str(path)+"\n",capture_output=True,text=True,
                                 env={**os.environ,"PYTHONPATH":str(Path.cwd())},check=True)
            decoded=json.loads(child.stdout)
            self.assertEqual(decoded["logits"],normal.tolist());self.assertEqual(decoded["generation"],r.model.generate(**inputs).tolist())
            with self.assertRaises(ValueError):load_original_matrix(TinyModel(),path,base_binding="c"*64)
            wrong=TinyModel()
            with torch.no_grad():wrong.head.weight.add_(.1)
            with self.assertRaises(ValueError):load_original_matrix(wrong,path,base_binding="b"*64)
        # Real deployment dtype rounding is actually exercised on this tiny model.
        half=MatrixRuntime(TinyModel(torch.bfloat16),"mlp.down_proj.weight")
        half.write(torch.ones_like(half.weight))
        half.write(half.weight.float()+1e-5)
        self.assertEqual(half.last_rounding["zeroed_fraction"],1.)
        self.metric(max_error=error,clean_subprocess=True,synthetic_generation_equal=True,
                    bf16_zeroed_fraction=half.last_rounding["zeroed_fraction"],native="PENDING_NATIVE_CHECK")

    def test_16_exception_reset_and_sequential(self):
        r,inputs,cs,gs,cfg=self.runtime_case(); initial=r.weight.detach().clone()
        calls=[0]
        original=r.normal_logits
        def raises(inputs):
            calls[0]+=1
            if calls[0]>2:raise RuntimeError("injected real-forward error")
            return original(inputs)
        r.normal_logits=raises
        result=edit_one(r,cs,gs,cfg,"W_FUNCTIONAL_QP")
        self.assertEqual(result.status,"EXCEPTION_ROLLED_BACK"); self.assertTrue(torch.equal(initial,r.weight))
        self.assertEqual(result.attempts[-1]["status"],"EXCEPTION_ROLLED_BACK")
        r.normal_logits=original
        result=edit_one(r,cs,gs,cfg,"W_FUNCTIONAL_QP");self.assertEqual(result.status,"ACCEPTED")
        first=r.weight.detach().clone();cs[0].threshold=.2
        result=edit_one(r,cs,gs,cfg,"W_FUNCTIONAL_QP");self.assertEqual(result.status,"ACCEPTED")
        self.assertFalse(torch.equal(first,r.weight));r.reset_single();self.assertTrue(torch.equal(initial,r.weight))
        # Ctrl-C after write restores before being re-raised to the caller.
        calls[0]=0
        def interrupted(inputs):
            calls[0]+=1
            if calls[0]>2:raise KeyboardInterrupt()
            return original(inputs)
        r.normal_logits=interrupted
        with tempfile.TemporaryDirectory() as directory:
            logfile=Path(directory)/"attempts.jsonl"
            with self.assertRaises(KeyboardInterrupt):edit_one(r,cs,gs,cfg,"W_FUNCTIONAL_QP",attempt_log=logfile)
            events=[json.loads(line) for line in logfile.read_text().splitlines()]
            self.assertEqual(events[-1]["status"],"EXCEPTION_ROLLED_BACK")
            self.assertTrue(any(e["event"]=="ATTEMPT" for e in events))
        self.assertTrue(torch.equal(initial,r.weight))
        self.metric(max_error=0.,exception_rollback=True,interrupt_rollback=True,single_reset=True,sequential_inheritance=True)

    def test_17_fixed_rng_cache_history(self):
        random_logits=lambda w:self.logits(w)+.1*torch.randn(2,3,3,dtype=self.dtype)
        frozen=FrozenGGN(random_logits,self.w,self.weights);v=torch.randn_like(self.w)
        rng=torch.random.get_rng_state();one=frozen(v);two=frozen(v)
        self.assertTrue(torch.equal(one,two));self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        expected=binding()
        for key in expected:
            actual=dict(expected);actual[key]="wrong"
            with self.assertRaises(ValueError):validate_cache(actual,expected)
        memory=HistoryMemory(2,4,4096)
        payload=dict(role="PAST_EDIT_MEMORY",fit_permission_verified=True,teacher_kind="ACCEPTED")
        for i in range(4):
            e=MemoryEntry(str(i),str(i%2),str(i),i,2,torch.tensor([[.5,.5]],dtype=self.dtype),payload,binding())
            memory.add(e,current_index=i,accepted=True)
        self.assertLessEqual(memory.accounting()["inputs"],2);self.assertLessEqual(memory.accounting()["positions"],4)
        with self.assertRaises(ValueError):memory.at(0)
        future=MemoryEntry("f","s","f",9,2,torch.tensor([[.5,.5]]),payload,binding())
        with self.assertRaises(ValueError):memory.add(future,current_index=4,accepted=True)
        conflict_memory=HistoryMemory(2,4,4096)
        old=MemoryEntry("old","s","fact",0,2,torch.tensor([[.5,.5]],dtype=self.dtype),payload,binding())
        conflict_memory.add(old,current_index=0,accepted=True)
        new=MemoryEntry("new","s","fact",1,2,torch.tensor([[.2,.8]],dtype=self.dtype),dict(payload,conflict=True),binding())
        with self.assertRaises(ValueError):conflict_memory.add(new,current_index=1,accepted=True)
        conflict_memory.add(new,current_index=1,accepted=True,supersedes=("old",))
        self.assertEqual(list(conflict_memory.entries),["new"])
        self.assertTrue(torch.equal(conflict_memory.entries["new"].teacher,new.teacher))
        self.metric(max_error=float((one-two).abs().max()),history=memory.accounting(),evictions=len(memory.evictions))

    def test_18_gate_before_loader(self):
        bindings={k:digest(k) for k in BINDINGS}
        config=default_config();called=[]
        loader=lambda:called.append(1)
        with self.assertRaises(PermissionError):gated_load("NATIVE_SMOKE",bindings,None,config,loader)
        with self.assertRaises(PermissionError):cli(["--stage","PILOT","--run"])
        config.update(execution_stage="NATIVE_SMOKE",current_state="NATIVE_SMOKE_ALLOWED",allow_native_model_execution=True,
            allow_gpu=True,authorized_gpu_hours=1,leased_gpu_uuids=["fixture-uuid"],model_binding="m",editable_weight_path="p",
            data_manifest_digest="d",approved_protocol_digest="p",trust_radius=.1,behavior_thresholds={"fit":.1},preservation_budgets={"bg":.01},
            smoke_data_audit_status="PASS",mechanical_smoke_inputs=1,native_wall_seconds_cap=60,
            max_active_constraints=2,max_edit_steps=1)
        approval=dict(approved=True,approved_phases=["NATIVE_SMOKE"],approval_reference="test-receipt",bindings=bindings)
        trusted=dict(origin="USER_MESSAGE",original_text="synthetic authorization fixture only",reference="test-receipt",phases=["NATIVE_SMOKE"],bindings=bindings)
        # This positive fixture validates the gate, does not write any approval file or invoke loader.
        require_external_approval("NATIVE_SMOKE",bindings,approval,config,trusted_authorization=trusted)
        failures=0
        for key in BINDINGS:
            bad=dict(bindings);bad[key]="x"*64
            with self.assertRaises(PermissionError):gated_load("NATIVE_SMOKE",bad,approval,config,loader,trusted_authorization=trusted)
            failures+=1
        for field,value in [("authorized_gpu_hours",0),("leased_gpu_uuids",[]),("trust_radius",None),("preservation_budgets",None),
                            ("smoke_data_audit_status","BLOCKED"),("mechanical_smoke_inputs",0),("teacher_cache_limit_gib",0),
                            ("authorized_gpu_hours",float("inf")),("native_wall_seconds_cap",0), ("max_edit_steps",20)]:
            bad=dict(config);bad[field]=value
            with self.assertRaises(PermissionError):gated_load("NATIVE_SMOKE",bindings,approval,bad,loader,trusted_authorization=trusted)
            failures+=1
        with self.assertRaises(PermissionError):gated_load("PILOT",bindings,approval,config,loader,trusted_authorization=trusted)
        self.assertEqual(called,[]);self.metric(max_error=0.,binding_budget_rejections=failures,loader_calls=0)
        state=advance_state("IMPLEMENTING","CPU_TESTS_COMPLETE",cpu_passed=True)
        state=advance_state(state,"DATA_AUDIT_COMPLETE",audit_complete=True)
        state=advance_state(state,"WAITING_FOR_EXTERNAL_REVIEW",audit_complete=True)
        self.assertEqual(state,"WAITING_FOR_EXTERNAL_REVIEW")
        with self.assertRaises(PermissionError):advance_state(state,"PILOT_ALLOWED",stage="PILOT")

    def test_19_failed_missing_judge_denominator(self):
        ledger=AttemptLedger();ledger.record("a","ACCEPTED",judged=True,correct=True)
        ledger.record("b","TRAINING_FAILED");ledger.record("c","ROLLED_BACK");ledger.record("d","MISSING_JUDGE_KEY")
        with self.assertRaises(ValueError):ledger.record("unscored","ACCEPTED",correct=True)
        with self.assertRaises(ValueError):ledger.record("nonboolean","ACCEPTED",judged=True,correct="yes")
        result=ledger.summary();self.assertEqual(result["intended"],4);self.assertEqual(result["missing"],3)
        self.assertEqual(result["correctness_lower"],.25);self.assertEqual(result["correctness_upper"],1.)
        self.metric(max_error=0.,**result)

    def test_20_cleanup_scope_consumers_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"r";root.mkdir();inside=root/"a.pt";inside.write_text("fixture")
            outside=Path(directory)/"outside.pt";outside.write_text("private")
            with self.assertRaises(ValueError):cleanup(root,[outside],{"outside.pt":[]},completed=set(),evidence_complete=True)
            with self.assertRaises(ValueError):cleanup(root,[inside],{"a.pt":["generation"]},completed=set(),evidence_complete=True)
            with self.assertRaises(ValueError):cleanup(root,[inside],{"a.pt":["active_resume"]},completed={"active_resume"},evidence_complete=True)
            link=root/"link.pt";link.symlink_to(outside)
            with self.assertRaises(ValueError):cleanup(root,[link],{"link.pt":[]},completed=set(),evidence_complete=True)
            plan=cleanup(root,[inside],{"a.pt":["generation"]},completed={"generation"},evidence_complete=True)
            self.assertTrue(inside.exists());self.assertEqual(plan,["a.pt"])
            with self.assertRaises(ValueError):enforce_storage(root,5,10)
            cleanup(root,[inside],{"a.pt":[]},completed=set(),evidence_complete=True,dry_run=False)
            self.assertFalse(inside.exists());self.assertTrue(outside.exists())
        self.metric(max_error=0.,outside_preserved=True,dry_run=True)


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records=[];self.started={}
    def startTest(self,test):
        self.started[test.id()]=time.monotonic();super().startTest(test)
    def record(self,test,status,error=None):
        name=test._testMethodName
        self.records.append(dict(test=name,status=status,seed=SEED,dtype="float64 CPU",
            seconds=time.monotonic()-self.started[test.id()],metrics=METRICS.get(name,{}),error=error))
    def addSuccess(self,test):super().addSuccess(test);self.record(test,"PASS")
    def addFailure(self,test,err):super().addFailure(test,err);self.record(test,"FAIL",self._exc_info_to_string(err,test))
    def addError(self,test,err):super().addError(test,err);self.record(test,"FAIL",self._exc_info_to_string(err,test))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.record(test,"SKIPPED",reason)


def main(output: Path | None = None) -> int:
    result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    document=dict(scope="CPU SYNTHETIC ONLY; no native VLM performance",seed=SEED,
        command="/opt/miniconda3/bin/python - < neutral test runner",python=sys.version,torch=torch.__version__,
        counts={k:sum(r["status"]==k for r in result.records) for k in ("PASS","FAIL","SKIPPED","PENDING_NATIVE")},
        native_check_status="PENDING_NATIVE",tests=result.records)
    if output:output.write_text(json.dumps(document,indent=2)+"\n")
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    raise SystemExit(main(Path(os.environ["RESULT_PATH"]) if "RESULT_PATH" in os.environ else None))
