"""CPU regressions for role, phase and mixed precision boundaries; no native claims."""
import unittest
import torch
from .contracts import SMOKE, FIT, audit_data, method_eligibility, AttemptLedger, digest
from .gate import BINDINGS, default_config, require_external_approval
from .tests import row
from .numerics import cg, solve_step
from .editor import MatrixRuntime
from .fixture import TinyModel


class PhaseBChecks(unittest.TestCase):
    def test_nonpersistent_buffers_strict_functional_reset_and_alias(self):
        model=TinyModel()
        model.register_buffer('fixed',torch.tensor(1.,dtype=torch.float64),persistent=False)
        with torch.no_grad():model.mlp.down_proj.weight.fill_(.1)
        runtime=MatrixRuntime(model,'mlp.down_proj.weight')
        inputs=dict(tokens=torch.ones(1,2,2,dtype=torch.float64),image=torch.ones(1,2,dtype=torch.float64))
        self.assertNotIn('fixed',model.state_dict())
        self.assertIn('fixed',runtime.functional_state)
        self.assertTrue(torch.equal(runtime.logits(runtime.base,inputs),runtime.normal_logits(inputs)))
        fn=runtime.bind_inputs(inputs)
        _,tangent=torch.func.jvp(fn,(runtime.base,),(torch.ones_like(runtime.base),))
        self.assertTrue(torch.isfinite(tangent).all());self.assertGreater(float(tangent.norm()),0)
        snapshot=runtime.base_state;hooks=runtime.hooks()
        model.fixed.fill_(2.)
        with self.assertRaises(RuntimeError):runtime.audit(snapshot,hooks)
        runtime.reset_single();runtime.audit(snapshot,hooks)
        self.assertEqual(float(model.fixed),1.)
        model.register_buffer('shared_buffer',model.mlp.down_proj.weight.detach(),persistent=False)
        with self.assertRaises(ValueError):MatrixRuntime(model,'mlp.down_proj.weight')

    def test_smoke_excluded_from_fit(self):
        sample=row(SMOKE, mechanical_permission_verified=True)
        result=audit_data([sample],edit_index=0,test_ids=set(),expected_model=None)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['eligible_counts'],{})
        self.assertTrue(all(x=='BLOCKED' for x in method_eligibility(result).values()))
        renamed=row('EDIT_FIT');renamed['id']='renamed'
        crossed=audit_data([sample,renamed],edit_index=0,test_ids=set(),expected_model=None)
        self.assertEqual(crossed['eligible_counts'],{})
        self.assertIn('SMOKE_TO_SCIENTIFIC_ROLE',[x['reason'] for x in crossed['findings']])
        renamed.update(original_role=SMOKE)
        renamed['question_hash']='different'
        renamed_audit=audit_data([renamed],edit_index=0,test_ids=set(),expected_model=None)
        self.assertEqual(renamed_audit['eligible_counts'],{})

    def test_evaluation_cannot_be_smoke(self):
        for patch in [{'source_split':s} for s in ['validation','test','heldout']]+[{'original_role':s} for s in ['REG_EXPOSED','DEV_CAL','TEST_CONFIRM']]+[{'evaluation_origin':True}]:
            sample=row(SMOKE,mechanical_permission_verified=True);sample.update(patch)
            result=audit_data([sample],edit_index=0,test_ids=set(),expected_model=None)
            self.assertIn('EVALUATION_TO_SMOKE',[x['reason'] for x in result['findings']])

    def test_pilot_not_admitted_by_smoke(self):
        binding={k:digest(k) for k in BINDINGS}
        approval=dict(approved=True,approved_phases=['NATIVE_SMOKE','PILOT'],approval_reference='fixture',bindings=binding)
        trusted=dict(origin='USER_MESSAGE',original_text='Synthetic gate fixture only',reference='fixture',phases=['NATIVE_SMOKE','PILOT'],bindings=binding)
        config=default_config();config.update(execution_stage='NATIVE_SMOKE',current_state='NATIVE_SMOKE_ALLOWED',
            allow_native_model_execution=True,allow_gpu=True,authorized_gpu_hours=.5,leased_gpu_uuids=['fixture'],
            model_binding='m',editable_weight_path='w',data_manifest_digest='d',approved_protocol_digest='p',
            trust_radius=.01,behavior_thresholds={'mechanical':1e-3},preservation_budgets={'mechanical':1e-3},
            smoke_data_audit_status='PASS',mechanical_smoke_inputs=1,native_wall_seconds_cap=1800,
            max_active_constraints=2,max_edit_steps=1)
        require_external_approval('NATIVE_SMOKE',binding,approval,config,trusted_authorization=trusted)
        pilot=dict(config,execution_stage='PILOT',current_state='PILOT_ALLOWED',allow_training=True,native_smoke_status='PASS',
                   data_audit_status='PASS',legal_fit_inputs=1,preapproved_global_rule_digest='fixture')
        with self.assertRaises(PermissionError):
            require_external_approval('PILOT',binding,approval,pilot,trusted_authorization=trusted)
        pilot['scientific_role_counts']={r:1 for r in ['EDIT_FIT','GEN_FIT','PROTECT_BG_FIT','PROTECT_NEAR_FIT']}
        require_external_approval('PILOT',binding,approval,pilot,trusted_authorization=trusted)
        pilot['preapproved_global_rule_digest']=None
        with self.assertRaises(PermissionError):
            require_external_approval('PILOT',binding,approval,pilot,trusted_authorization=trusted)

    def test_mechanical_not_in_scientific_denominator(self):
        ledger=AttemptLedger()
        for patch in [dict(role=SMOKE),dict(result_kind='MECHANICAL_NATIVE_VALIDATION')]:
            with self.assertRaises(ValueError):ledger.record('s','ACCEPTED',**patch)
        with self.assertRaises(ValueError):ledger.record('s','MECHANICAL_NATIVE_VALIDATION')
        self.assertEqual(ledger.summary()['intended'],0)
        ledger.record('fit','ROLLED_BACK');self.assertEqual(ledger.summary()['intended'],1)

    def test_fp32_temporary_coordinate_bf16_deployment(self):
        torch.manual_seed(20260930)
        base=torch.randn(3,4).to(torch.bfloat16)
        coordinate=base.float().requires_grad_(True)
        x=torch.randn(2,4).to(torch.bfloat16)
        fn=lambda w:(x@w.to(torch.bfloat16).T).float()
        normal=(x@base.T).float()
        self.assertTrue(torch.equal(fn(coordinate),normal))
        grad=torch.autograd.grad(fn(coordinate).log_softmax(-1)[:,0].mean(),coordinate)[0]
        self.assertEqual(grad.dtype,torch.float32);self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)
        q=lambda v:2*v
        a=grad.flatten()[None,:];b=torch.tensor([.01]);g=torch.zeros_like(grad)
        solved=solve_step(q,a,b,g,cg_rtol=1e-4,cg_max_iter=4,dual_tol=1e-6)
        self.assertEqual(solved.status,'CONVERGED');self.assertEqual(solved.direction.dtype,torch.float32)
        requested=coordinate.detach()+solved.direction
        deployed=requested.to(torch.bfloat16)
        self.assertEqual(deployed.dtype,base.dtype)
        self.assertTrue(torch.isfinite(fn(deployed.float())).all())
        # Frozen CG tolerance is retained, rather than accepting BF16 stagnation.
        self.assertEqual(cg(q,grad,rtol=1e-4,max_iter=4).status,'CONVERGED')


if __name__=='__main__':
    unittest.main()
