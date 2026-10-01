"""CPU regressions for role, phase and mixed precision boundaries; no native claims."""
import unittest
import torch
from .contracts import SMOKE, FIT, audit_data, method_eligibility, AttemptLedger, digest
from .gate import BINDINGS, default_config, require_external_approval
from .tests import row, binding as teacher_binding, gate_fixture
from .numerics import cg, solve_step, FrozenGGN, protection_gradient, protection_kl
from .editor import MatrixRuntime
from .fixture import TinyModel


class PhaseBChecks(unittest.TestCase):
    def test_token_path_audit_separates_batch_prefix_mask_and_preserves_inputs(self):
        from types import SimpleNamespace
        from .native_io import token_path_audit
        labels=torch.tensor([[-100,1,2]])
        logits=torch.tensor([[[0.,2.,0.],[0.,0.,2.],[0.,0.,0.]]])
        class Model:
            def __call__(self,**kwargs):return SimpleNamespace(logits=logits)
            def generate(self,**kwargs):
                j=kwargs['inputs'].shape[1]-2
                score=torch.tensor([[3.,1.,0.]]) if j==0 else torch.tensor([[0.,0.,3.]])
                if 'attention_mask' in kwargs:score=torch.tensor([[0.,3.,0.]]) if j==0 else score
                return SimpleNamespace(scores=(score,),sequences=score.argmax(-1)[:,None])
        generation=dict(inputs=torch.tensor([[4,5]]),do_sample=False,max_new_tokens=8)
        prepared=({},labels,generation,{})
        batch=logits.clone();batch[0,0,1]=2.5
        rng=torch.get_rng_state().clone();original=generation['inputs'].clone()
        result=token_path_audit(Model(),prepared,batch,labels)
        p=result['positions'][0]
        self.assertEqual(p['batch_teacher']['margin'],2.5)
        self.assertEqual(p['single_teacher']['margin'],2.)
        self.assertEqual(p['prefix_generate']['margin'],-2.)
        self.assertEqual(p['prefix_generate_explicit_mask']['margin'],3.)
        self.assertFalse(p['native_generated_target']);self.assertFalse(p['mask_generated_same'])
        self.assertTrue(result['positions'][1]['native_generated_target'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(original,generation['inputs']))
        self.assertEqual(generation['max_new_tokens'],8)
        with self.assertRaises(ValueError):token_path_audit(Model(),prepared,batch,torch.full_like(labels,-100))

    def test_fp32_large_rhs_tolerance_against_fp64_and_failed_cg(self):
        torch.manual_seed(0)
        a=torch.randn(1,128)*10;b=torch.tensor([14.97]);g=torch.zeros(128)
        strict=solve_step(lambda v:100*v,a,b,g,cg_rtol=1e-4,cg_max_iter=16)
        calibrated=solve_step(lambda v:100*v,a,b,g,cg_rtol=1e-4,cg_max_iter=16,dual_tol=1e-6)
        oracle=solve_step(lambda v:100*v,a.double(),b.double(),g.double(),cg_rtol=1e-4,cg_max_iter=16)
        self.assertEqual(calibrated.status,'CONVERGED');self.assertEqual(oracle.status,'CONVERGED')
        self.assertLess(float((calibrated.direction.double()-oracle.direction).abs().max()),1e-7)
        self.assertTrue(all(s.status=='CONVERGED' for s in calibrated.cg))
        # Roundoff can vary by CPU; if the strict solve fails it must retain the
        # same nearly exact direction, not be a genuinely bad CG solution.
        if strict.status!='CONVERGED':
            self.assertTrue(all(s.status=='CONVERGED' for s in strict.cg))
            self.assertLess(float((strict.direction.double()-oracle.direction).abs().max()),1e-7)
        diagonal=torch.linspace(1.,1000.,128)
        failed=solve_step(lambda v:diagonal*v,a,b,g,cg_rtol=1e-4,cg_max_iter=1,dual_tol=1e-6)
        self.assertEqual(failed.status,'NOT_CONVERGED')
        self.assertTrue(any(s.status!='CONVERGED' for s in failed.cg))

    def test_target_margin_uses_worst_content_or_eos_and_masks_padding(self):
        from .native_io import minimum_target_margin
        logits=torch.tensor([[[4.,5.,0.],[3.,0.,2.],[99.,0.,0.],[0.,0.,0.]]],requires_grad=True)
        labels=torch.tensor([[-100,1,2,-100]])
        margin=minimum_target_margin(logits,labels)
        self.assertEqual(float(margin),-1.)
        margin.backward()
        expected=torch.zeros_like(logits);expected[0,1,0]=-1.;expected[0,1,2]=1.
        self.assertTrue(torch.equal(logits.grad,expected))
        improved=logits.detach().clone();improved[0,1,2]=5.
        self.assertEqual(float(minimum_target_margin(improved,labels)),1.)
        self.assertTrue(torch.equal(improved[0,:2].argmax(-1),labels[0,1:3]))
        with self.assertRaises(ValueError):minimum_target_margin(logits,torch.full_like(labels,-100))

    def test_readonly_prompt_diagnostic_replay_and_failure_ledger(self):
        import tempfile,json
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from . import prompt_diagnostic as diagnostic
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False)
            def named_parameters(self):return [(diagnostic.WEIGHT,self.weight)]
            def forward(self,**kwargs):return SimpleNamespace(logits=torch.zeros(1,3,3))
        def prepared(*args,question_suffix=''):
            return {},torch.tensor([[-100,1,2]]),dict(suffix=question_suffix),{}
        def generated(model,tokenizer,generation,reference,cap):
            return dict(tokens=[11 if generation['suffix'] else 10,2],text='yes',reference=reference,
                metrics=dict(normalized_exact_match=True,token_F1=1.,generated_tokens=2,eos_reached=True,hit_token_cap=False))
        with tempfile.TemporaryDirectory() as directory,patch.object(diagnostic,'require_external_approval'),\
             patch.object(diagnostic,'load',side_effect=lambda c:(Model(),SimpleNamespace(eos_token_id=2),None)),\
             patch.object(diagnostic,'prepare',side_effect=prepared),\
             patch.object(diagnostic,'generation_snapshot',side_effect=generated),\
             patch.object(torch.cuda,'max_memory_allocated',return_value=0):
            for mismatch in (False,True):
                root=Path(directory)/str(mismatch);root.mkdir()
                config=dict(bindings={},run_root=str(root),prompt_rows=[dict(answer='yes')],prior_GGN_calls=2430,
                    prior_native_seconds=10.,native_wall_seconds_cap=60,question_suffix=' short',
                    generation_max_new_tokens=64,expected_base_tokens={'0':[99] if mismatch else [10,2]})
                result=diagnostic.run(config,{},{});self.assertTrue(result['final_Base_unchanged'])
                self.assertEqual(result['state'],'STOPPED_ON_ERROR' if mismatch else 'EXPERIMENT_COMPLETE')
                self.assertEqual(result['completed'],0 if mismatch else 1)
                self.assertEqual(result['cumulative_GGN_calls'],2430)
                self.assertGreaterEqual(result['cumulative_native_seconds'],10.)
                self.assertEqual(json.loads((root/'STATUS.json').read_text())['state'],result['state'])
                self.assertTrue((root/'0.generation.private.json').exists())
                self.assertEqual((root/'FAILURE.private.json').exists(),mismatch)

    def test_generation_suffix_scoring_and_truncation(self):
        from types import SimpleNamespace
        from .native_io import generation_snapshot,answer_metrics
        self.assertTrue(answer_metrics('  YES!  ','Yes')['normalized_exact_match'])
        self.assertFalse(answer_metrics('No, not yes','Yes')['normalized_exact_match'])
        self.assertEqual(answer_metrics('kidney kidney','kidney')['token_F1'],2/3)
        self.assertEqual(answer_metrics('','yes')['token_F1'],0)
        with self.assertRaises(ValueError):answer_metrics('yes','...')
        class Model:
            def generate(self,**kw):
                assert kw['max_new_tokens']==2 and kw['do_sample'] is False
                assert kw['return_dict_in_generate'] and kw['output_scores']
                return SimpleNamespace(sequences=torch.tensor([[99,10,2]]),scores=(None,None))
        class Tokenizer:
            eos_token_id=2
            def decode(self,tokens,skip_special_tokens):
                assert tokens==[10,2] and skip_special_tokens
                return 'Yes.'
        rng=torch.random.get_rng_state().clone()
        result=generation_snapshot(Model(),Tokenizer(),dict(do_sample=False,max_new_tokens=8),'yes',2)
        self.assertTrue(result['metrics']['normalized_exact_match'])
        self.assertFalse(result['metrics']['hit_token_cap'])
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        Tokenizer.eos_token_id=3
        self.assertTrue(generation_snapshot(Model(),Tokenizer(),dict(do_sample=False),'yes',2)['metrics']['hit_token_cap'])
        with self.assertRaises(ValueError):generation_snapshot(Model(),Tokenizer(),{},'yes',65)

    def test_native_diagnostic_probes_are_read_only_and_do_not_change_edit(self):
        from .exploratory import probe_snapshot
        from .editor import edit_one
        results=[]
        for enabled in (False,True):
            runtime,inputs,c,g,cfg=self.fp32_case()
            cfg.rounding_mode='stochastic_bf16';cfg.constraint_value_mode='native_value'
            pin=dict(inputs,tokens=inputs['tokens'][:,:1].expand(1,4,2).clone())
            labels=torch.tensor([[-100,1,2,-100]])
            if enabled:
                rng=torch.random.get_rng_state().clone();weight=runtime.weight.detach().clone()
                before,anchor=probe_snapshot(runtime,pin,labels,2)
                self.assertEqual(anchor.shape,(1,3))
                self.assertAlmostEqual(before,float(anchor[0,1]),places=6)
                self.assertTrue(torch.equal(weight,runtime.weight))
                self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
            self.assertEqual(result.status,'ACCEPTED')
            results.append((runtime.weight.detach().clone(),result.attempts,result.ggn_calls))
            if enabled:
                after,post=probe_snapshot(runtime,pin,labels,2)
                repeat,repeated=probe_snapshot(runtime,pin,labels,2)
                self.assertEqual(after,repeat);self.assertTrue(torch.equal(post,repeated))
                self.assertGreater(float((anchor.exp()*(anchor-post)).sum(-1).mean()),0)
                self.assertNotEqual(before,after)
                runtime.audit(runtime.base_state,runtime.base_hooks)
                runtime.reset_single()
                reset,restored=probe_snapshot(runtime,pin,labels,2)
                self.assertEqual(reset,before);self.assertTrue(torch.equal(anchor,restored))
                with self.assertRaises(ValueError):probe_snapshot(runtime,pin,torch.full_like(labels,-100),2)
        self.assertTrue(torch.equal(results[0][0],results[1][0]))
        self.assertEqual(results[0][1:],results[1][1:])

    def test_stochastic_bf16_brackets_and_exact_grid_expectation(self):
        from .editor import stochastic_bf16_round
        # Exhaust all 256 evenly spaced draws, with dyadic interpolation fractions.
        lower=torch.tensor([-2.,-1.,0.,.5,1.,2.],dtype=torch.bfloat16)
        upper=torch.nextafter(lower,torch.full_like(lower,torch.inf))
        for p in (0.,.25,.5,.75,1.):
            value=lower.float()+p*(upper.float()-lower.float())
            x=value[None,:].expand(256,-1)
            u=((torch.arange(256,dtype=torch.float32)+.5)/256)[:,None].expand_as(x)
            y=stochastic_bf16_round(x,u)
            self.assertTrue(((y==lower)|(y==upper)).all())
            self.assertTrue(torch.equal(y,stochastic_bf16_round(x,u)))
            self.assertTrue(torch.allclose(y.double().mean(0),value.double(),atol=0,rtol=0))
        exact=torch.tensor([-2.,-0.,0.,1.,2.])
        self.assertTrue(torch.equal(stochastic_bf16_round(exact,torch.full_like(exact,.5)).float(),exact))
        for x,u in [(torch.tensor([float('nan')]),torch.tensor([.5])),
                    (torch.tensor([torch.finfo(torch.float32).max]),torch.tensor([.5])),
                    (torch.ones(1),torch.ones(1)),(torch.ones(1),torch.tensor([-.1])),
                    (torch.ones(1),torch.tensor([float('nan')])),
                    (torch.ones(2),torch.ones(1)*.5),(torch.ones(1).double(),torch.ones(1)*.5)]:
            with self.assertRaises(ValueError):stochastic_bf16_round(x,u)

    def test_stochastic_bf16_edit_repeatability_and_rollback(self):
        from .editor import edit_one
        runs=[]
        for _ in range(2):
            runtime,_,c,g,cfg=self.fp32_case()
            cfg.rounding_mode='stochastic_bf16';cfg.constraint_value_mode='native_value'
            cfg.record_trial_diagnostics=True
            rng=torch.random.get_rng_state().clone()
            result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
            self.assertEqual(result.status,'ACCEPTED',result)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertTrue(all(a['trial_diagnostics']['native_repeat_exact'] for a in result.attempts))
            runtime.audit(runtime.base_state,runtime.base_hooks)
            runs.append((runtime.weight.detach().clone(),result.attempts))
        self.assertTrue(torch.equal(runs[0][0],runs[1][0]));self.assertEqual(runs[0][1],runs[1][1])
        runtime,_,c,g,cfg=self.fp32_case()
        cfg.rounding_mode='stochastic_bf16';cfg.max_steps=1;c.threshold=100.
        result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
        self.assertTrue(result.attempts);self.assertTrue(result.rollback)
        self.assertNotEqual(result.status,'ACCEPTED');self.assertIsNone(result.error)
        self.assertTrue(torch.equal(runtime.weight,runtime.base))
        runtime.audit(runtime.base_state,runtime.base_hooks)
        runtime,_,c,g,cfg=self.fp32_case(deployment_dtype=torch.float32)
        cfg.rounding_mode='stochastic_bf16'
        with self.assertRaises(ValueError):edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')

    def test_explicit_native_fp32_control_preserves_initial_values(self):
        from .native_io import deployment_dtype
        from .editor import edit_one
        self.assertEqual(deployment_dtype({}),torch.bfloat16)
        self.assertEqual(deployment_dtype({'native_deployment_dtype':'float32'}),torch.float32)
        with self.assertRaises(ValueError):deployment_dtype({'native_deployment_dtype':'float16'})
        model=TinyModel(dtype=torch.bfloat16)
        initial={k:v.detach().float().clone() for k,v in model.state_dict().items()}
        model=model.to(deployment_dtype({'native_deployment_dtype':'float32'}))
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in model.state_dict().items()))
        runtime,_,c,g,cfg=self.fp32_case(deployment_dtype=torch.float32)
        cfg.constraint_value_mode='native_value';cfg.record_trial_diagnostics=True
        result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
        self.assertEqual(result.status,'ACCEPTED',result)
        self.assertEqual(runtime.weight.dtype,torch.float32)
        for a in result.attempts:
            d=a['trial_diagnostics']
            self.assertEqual(d['functional_requested'],d['functional_rounded'])
            self.assertEqual(d['functional_rounded'],a['actual_scores'])

    def test_trial_diagnostics_do_not_change_edit_and_record_real_forwards(self):
        from .editor import edit_one
        outputs=[]
        for enabled in (False,True):
            runtime,_,c,g,cfg=self.fp32_case()
            cfg.constraint_value_mode='native_value';cfg.record_trial_diagnostics=enabled
            result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
            self.assertEqual(result.status,'ACCEPTED',result)
            if enabled:
                for attempt in result.attempts:
                    d=attempt.pop('trial_diagnostics')
                    self.assertTrue(d['native_repeat_exact'])
                    self.assertEqual(len(d['functional_rounded']),1)
                    self.assertTrue(torch.isfinite(torch.tensor(d['affine_requested'])).all())
                    self.assertTrue(torch.isfinite(torch.tensor(d['functional_KL_rounded']['bg'])))
            outputs.append((runtime.weight.detach().clone(),result.attempts,result.ggn_calls))
        self.assertTrue(torch.equal(outputs[0][0],outputs[1][0]))
        self.assertEqual(outputs[0][1:],outputs[1][1:])

    def test_native_value_anchor_ignores_constant_surrogate_offset(self):
        from .editor import edit_one
        results=[]
        for offset in (0.,10.):
            runtime,_,c,g,cfg=self.fp32_case()
            original=c.score;c.score=lambda w,f=original: f(w)+offset
            cfg.constraint_value_mode='native_value'
            result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
            self.assertEqual(result.status,'ACCEPTED',result)
            self.assertGreaterEqual(float(c.normal_score()),c.threshold-cfg.tolerance)
            results.append(runtime.weight.detach().clone())
        self.assertTrue(torch.equal(*results))
        runtime,_,c,g,cfg=self.fp32_case()
        original=c.score;c.score=lambda w:original(w)+10.
        result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
        self.assertNotEqual(result.status,'ACCEPTED')
        self.assertTrue(result.rollback)

    def test_resume_preserves_negative_prefix_and_cumulative_budget(self):
        import json,tempfile
        from pathlib import Path
        from .exploratory import resumed_cases
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            config=dict(model_binding='base',editable_weight_path='w',data_manifest_digest='data',protocol={'tau':100},
                candidate_rows=[],exploratory_inputs=8,target_logprob_gain=.1,max_edit_steps=3,maximum_CG_iterations=16,
                run_root=str(p/'next'),resume_from=str(p),prior_GGN_calls=161,prior_native_seconds=990.)
            cases=[dict(index=str(i),Base_restored=True,status='BACKTRACK_REJECTED',rollback=True) for i in range(4)]
            previous=dict(state='STOPPED_ON_ERROR',final_Base_restored=True,cases=cases,completed=4,
                previous_native_seconds=144.,elapsed_seconds=846.,cumulative_GGN_calls=161)
            (p/'CONFIG.private.json').write_text(json.dumps(dict(config=config)))
            (p/'STATUS.json').write_text(json.dumps(previous))
            self.assertEqual(resumed_cases(config),cases)
            for patch in (dict(prior_GGN_calls=46),dict(prior_native_seconds=144.),dict(target_logprob_gain=.01),dict(run_root=str(p))):
                with self.assertRaises(ValueError):resumed_cases(dict(config,**patch))
            previous['cases'][0]['status']='EXCEPTION_ROLLED_BACK'
            (p/'STATUS.json').write_text(json.dumps(previous))
            with self.assertRaises(ValueError):resumed_cases(config)

    def test_exploratory_source_selection_and_explicit_waiver_gate(self):
        from .exploratory import select_cases
        rows=[dict(id=str(i),source_group=str(i%3),image_hash=str(i%3),question_hash=str(i),
                   role='CANDIDATE_ONLY',original_role='train',source_split='train') for i in range(12)]
        rows[0]['role']='SMOKE_MECHANICAL';rows[1]['source_split']='test'
        pairs=select_cases(rows,8)
        self.assertEqual(len(pairs),8)
        self.assertEqual(pairs,select_cases(list(reversed(rows)),8))
        self.assertTrue(all(a['id'] not in ('0','1') and b['id'] not in ('0','1') and a['image_hash']!=b['image_hash'] for a,b in pairs))
        c=default_config();c.update(execution_stage='EXPLORATORY_PILOT',current_state='EXPLORATORY_ALLOWED',
            allow_native_model_execution=True,allow_training=True,allow_gpu=True,authorized_gpu_hours=4,
            leased_gpu_uuids=['fixture'],model_binding='m',editable_weight_path='w',data_manifest_digest='d',
            trust_radius=.1,behavior_thresholds={'gain':.1},preservation_budgets={'reference':.001},
            result_kind='EXPLORATORY_TRAINING_ONLY',verified_training_source=True,exploratory_inputs=8,
            prelaunch_native_validation_required=True,waived_requirements=['scientific_fit_admission',
            'independent_calibration','previous_smoke_budget','external_review_stop'])
        b,a,t=gate_fixture(c,('EXPLORATORY_PILOT',))
        with self.assertRaises(PermissionError):require_external_approval('EXPLORATORY_PILOT',b,a,c,trusted_authorization=t)
        t['waived_requirements']=c['waived_requirements'][:]
        require_external_approval('EXPLORATORY_PILOT',b,a,c,trusted_authorization=t)
        self.assertEqual(c['data_audit_status'],'BLOCKED')
        with self.assertRaises(PermissionError):require_external_approval('PILOT',b,a,c,trusted_authorization=t)

    def test_ggn_safety_count_survives_exception_rollback(self):
        from .editor import edit_one
        r,_,c,g,cfg=self.fp32_case();cfg.max_ggn_calls=1
        result=edit_one(r,[c],[g],cfg,'W_FUNCTIONAL_QP')
        self.assertEqual(result.ggn_calls,1)
        self.assertEqual(result.status,'EXCEPTION_ROLLED_BACK')
        self.assertIn('GGN call safety ceiling',result.error)
        self.assertTrue(torch.equal(r.weight,r.base))

    def fp32_case(self, deployment_dtype=torch.bfloat16):
        from .editor import Constraint, ProtectionGroup, EditConfig
        model=TinyModel(dtype=deployment_dtype);model.requires_grad_(False)
        runtime=MatrixRuntime(model,'mlp.down_proj.weight')
        inputs=dict(tokens=torch.zeros(1,2,2,dtype=deployment_dtype),image=torch.tensor([[1.,0.]],dtype=deployment_dtype))
        functional=runtime.bind_inputs(inputs,arithmetic_dtype=torch.float32)
        normal=runtime.bind_normal_inputs(inputs)
        group=ProtectionGroup('bg',functional,functional(runtime.base.float()).softmax(-1).detach(),
            torch.ones(1,2,dtype=torch.bool),('source',),1.,1.,dict(teacher_binding(),dtype='torch.float32'),
            normal_logits=normal,deployment_anchor=normal().float().softmax(-1).detach(),
            deployment_binding=dict(teacher_binding(),dtype=str(deployment_dtype)))
        constraint=Constraint('edit','EDIT_FIT',lambda w:functional(w)[0,0,0],.1,normal_score=lambda:normal().float()[0,0,0])
        # CPU fixture only; these are not new native tolerances or permissions.
        config=EditConfig(.1,1000.,.05,2.,1.,max_steps=8,cg_max_iter=8,cg_rtol=1e-5,
                          tolerance=.002,arithmetic_dtype=torch.float32)
        return runtime,inputs,constraint,group,config

    def test_fp32_edit_uses_bf16_normal_acceptance(self):
        from .editor import edit_one
        for branch in ('W_FT','W_EUCLIDEAN_QP','W_KEY_QP','W_FUNCTIONAL_QP'):
            with self.subTest(branch=branch):
                runtime,inputs,c,g,cfg=self.fp32_case()
                g.keys=runtime.capture_input(inputs)
                if branch=='W_FT':cfg.ft_lr=10.;cfg.tau=1e-8
                original=runtime.normal_logits;calls=[]
                def counted(inputs):
                    result=original(inputs);calls.append(result.dtype);return result
                runtime.normal_logits=counted
                result=edit_one(runtime,[c],[g],cfg,branch)
                self.assertEqual(result.status,'ACCEPTED',result)
                self.assertGreater(result.accepted_steps,0)
                self.assertGreater(len(calls),2)
                self.assertEqual(set(calls),{torch.bfloat16})
                self.assertGreaterEqual(float(c.normal_score()),c.threshold-cfg.tolerance)
                self.assertEqual(runtime.weight.dtype,torch.bfloat16)
                runtime.audit(runtime.base_state,runtime.hooks())
                c.normal_score=None
                with self.assertRaisesRegex(ValueError,'explicit normal-forward'):
                    edit_one(runtime,[c],[g],cfg,branch)

    def test_fp32_satisfied_does_not_override_failed_bf16_score(self):
        from .editor import edit_one
        runtime,inputs,c,g,cfg=self.fp32_case()
        with torch.no_grad():runtime.weight.fill_(.2)
        functional=runtime.bind_inputs(inputs,arithmetic_dtype=torch.float32)
        smooth=functional(runtime.weight.float())[0,0,0].detach()
        actual=runtime.normal_logits(inputs).float()[0,0,0].detach()
        sign=1. if smooth>actual else -1.
        self.assertGreater(float((smooth-actual).abs()),1e-6)
        c.score=lambda w:sign*functional(w)[0,0,0]
        c.normal_score=lambda:sign*runtime.normal_logits(inputs).float()[0,0,0]
        c.threshold=float(sign*(smooth+actual)/2)
        g.anchor=functional(runtime.weight.float()).softmax(-1).detach()
        g.deployment_anchor=runtime.normal_logits(inputs).float().softmax(-1).detach()
        cfg.tolerance=1e-8;cfg.max_steps=1
        before=runtime.weight.detach().clone()
        self.assertGreater(float(c.score(before.float())),c.threshold)
        self.assertLess(float(c.normal_score()),c.threshold)
        result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
        self.assertNotEqual(result.status,'ACCEPTED',result)
        self.assertTrue(torch.equal(before,runtime.weight))

    def test_full_buffer_restore_on_exception_after_bf16_write(self):
        from .editor import MatrixRuntime,edit_one
        runtime,inputs,c,g,cfg=self.fp32_case()
        runtime.model.register_buffer('scratch',torch.tensor(7.),persistent=False)
        # Rebind runtime so this is legitimate edit-start state, not a new buffer.
        runtime=MatrixRuntime(runtime.model,'mlp.down_proj.weight')
        functional=runtime.bind_inputs(inputs,arithmetic_dtype=torch.float32)
        c.score=lambda w:functional(w)[0,0,0]
        c.normal_score=lambda:runtime.normal_logits(inputs).float()[0,0,0]
        g.logits=functional;g.normal_logits=runtime.bind_normal_inputs(inputs)
        before={k:v.clone() for k,v in runtime.model.state_dict().items()}
        original=runtime.normal_logits
        def fail_after_write(inputs):
            if runtime.version:
                runtime.model.scratch.add_(3)
                raise RuntimeError('controlled post-write failure')
            return original(inputs)
        runtime.normal_logits=fail_after_write
        result=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
        self.assertEqual(result.status,'EXCEPTION_ROLLED_BACK',result)
        self.assertTrue(result.rollback)
        self.assertTrue(result.attempts)
        self.assertEqual(float(runtime.model.scratch),7.)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in runtime.model.state_dict().items()))
        runtime.audit(runtime.base_state,runtime.hooks())
        runtime.reset_single();self.assertEqual(float(runtime.model.scratch),7.)
        # If the registry itself changes, do not report a successful rollback.
        def break_registry(inputs):
            runtime.model.register_buffer('unexpected',torch.tensor(1.),persistent=False)
            return original(inputs)
        runtime.normal_logits=break_registry
        failed=edit_one(runtime,[c],[g],cfg,'W_FUNCTIONAL_QP')
        self.assertEqual(failed.status,'ROLLBACK_FAILED')
        self.assertFalse(failed.rollback)

    def test_bound_execution_changes_and_cumulative_budget_fail_before_loader(self):
        import copy
        from .gate import gated_load
        config=default_config();config.update(execution_stage='NATIVE_SMOKE',current_state='NATIVE_SMOKE_ALLOWED',
            allow_native_model_execution=True,allow_gpu=True,authorized_gpu_hours=.5,leased_gpu_uuids=['fixture'],
            model_binding='m',editable_weight_path='w',data_manifest_digest='d',trust_radius=.1,
            behavior_thresholds={'mechanical':1e-4},preservation_budgets={'mechanical':.001},
            smoke_data_audit_status='PASS',mechanical_smoke_inputs=1,native_wall_seconds_cap=1600,
            max_active_constraints=2,max_edit_steps=1,prior_native_seconds=144.393211,prior_GGN_calls=46,
            maximum_GGN_calls=2,damping=100.)
        bindings,approval,trusted=gate_fixture(config)
        called=[];loader=lambda:called.append(True)
        # Positive check, no model loader.
        require_external_approval('NATIVE_SMOKE',bindings,approval,config,trusted_authorization=trusted)
        for field,value in [('maximum_CG_iterations',400),('maximum_GGN_calls',480),('damping',200.),
                            ('cg_rtol',.1),('native_wall_seconds_cap',1800),('prior_GGN_calls',0)]:
            with self.subTest(changed=field):
                changed=copy.deepcopy(config);changed[field]=value
                with self.assertRaisesRegex(PermissionError,'actual execution'):
                    gated_load('NATIVE_SMOKE',bindings,approval,changed,loader,trusted_authorization=trusted)
        changed=copy.deepcopy(config);changed['protocol']['cg_rtol']=.2
        with self.assertRaises(PermissionError):
            gated_load('NATIVE_SMOKE',bindings,approval,changed,loader,trusted_authorization=trusted)
        # Even a synthetic fresh binding cannot admit internally over-budget config.
        for patch in [dict(maximum_GGN_calls=3),dict(native_wall_seconds_cap=1700),dict(maximum_CG_iterations=400)]:
            changed=dict(config,**patch);b,a,t=gate_fixture(changed)
            with self.assertRaises(PermissionError):gated_load('NATIVE_SMOKE',b,a,changed,loader,trusted_authorization=t)
        self.assertEqual(called,[])

    def test_analytic_protection_pullback_stationary_and_nonstationary(self):
        torch.manual_seed(21)
        x=torch.randn(2,3,dtype=torch.float64)
        point=torch.randn(3,4,dtype=torch.float64)
        fn=lambda w:x@w
        anchor=fn(point).softmax(-1).detach();weights=torch.ones(2,dtype=torch.float64)/2
        self.assertEqual(int(torch.count_nonzero(protection_gradient(fn,point,anchor,weights))),0)
        moved=(point+.1*torch.randn_like(point)).requires_grad_(True)
        actual=protection_gradient(fn,moved,anchor,weights)
        reference=torch.autograd.grad(protection_kl(fn(moved),anchor,weights),moved)[0]
        self.assertGreater(float(actual.norm()),0)
        self.assertTrue(torch.allclose(actual,reference,atol=1e-12,rtol=1e-12))

    def test_reused_inverse_verified_against_current_operator(self):
        q=torch.diag(torch.tensor([1.,2.,3.],dtype=torch.float64));op=lambda v:q@v
        a=torch.tensor([[1.,1.,0.],[0.,1.,1.]],dtype=torch.float64)
        g=torch.tensor([.2,-.1,.3],dtype=torch.float64);b=torch.tensor([.4,.1],dtype=torch.float64)
        saved=[cg(op,v,max_iter=4) for v in [g,*a]]
        fresh=solve_step(op,a,b,g,cg_max_iter=4)
        reused=solve_step(op,a,b,g,cg_max_iter=4,inverse_solutions=saved)
        self.assertEqual(reused.status,'CONVERGED')
        self.assertTrue(torch.allclose(fresh.direction,reused.direction,atol=1e-10,rtol=1e-10))
        self.assertTrue(all(s.iterations==0 and s.calls==1 for s in reused.cg))
        stale=solve_step(lambda v:2*q@v,a,b,g,cg_max_iter=4,inverse_solutions=saved)
        self.assertEqual(stale.status,'NOT_CONVERGED')
        self.assertTrue(any(s.relative_residual>1e-4 for s in stale.cg))

    def test_bf16_cast_does_not_define_a_smooth_fp32_function(self):
        point=torch.tensor([1.],requires_grad=True)
        quantized=lambda w:w.to(torch.bfloat16).float().square().sum()
        derivative=torch.autograd.grad(quantized(point),point)[0]
        eps=1e-4
        fd=(quantized(point.detach()+eps)-quantized(point.detach()-eps))/(2*eps)
        self.assertEqual(float(derivative),2.)
        self.assertEqual(float(fd),0.)

    def test_fp32_functional_arithmetic_preserves_bf16_deployment(self):
        torch.manual_seed(20261001)
        model=TinyModel(dtype=torch.bfloat16)
        with torch.no_grad():model.mlp.down_proj.weight.normal_(0,.2)
        runtime=MatrixRuntime(model,'mlp.down_proj.weight')
        inputs=dict(tokens=torch.randn(1,2,2).bfloat16(),image=torch.randn(1,2).bfloat16())
        fn=runtime.bind_inputs(inputs,arithmetic_dtype=torch.float32)
        point=runtime.base.float();weights=torch.ones(1,2)/2
        F=FrozenGGN(fn,point,weights);u=torch.randn_like(point);v=torch.randn_like(point)
        self.assertEqual(fn(point).dtype,torch.float32)
        self.assertLess(abs(float((u*F(v)).sum()-(v*F(u)).sum())),1e-5)
        eps=.001
        _,jv=torch.func.jvp(fn,(point,),(u,))
        fd=(fn(point+eps*u)-fn(point-eps*u))/(2*eps)
        self.assertTrue(torch.allclose(fd,jv,atol=1e-4,rtol=1e-3))
        runtime.audit(runtime.base_state,runtime.hooks())
        self.assertEqual(runtime.weight.dtype,torch.bfloat16)
        self.assertTrue(torch.equal(runtime.weight,runtime.base))
        with self.assertRaises(ValueError):fn(runtime.base)

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
        config=default_config();config.update(execution_stage='NATIVE_SMOKE',current_state='NATIVE_SMOKE_ALLOWED',
            allow_native_model_execution=True,allow_gpu=True,authorized_gpu_hours=.5,leased_gpu_uuids=['fixture'],
            model_binding='m',editable_weight_path='w',data_manifest_digest='d',approved_protocol_digest='p',
            trust_radius=.01,behavior_thresholds={'mechanical':1e-3},preservation_budgets={'mechanical':1e-3},
            smoke_data_audit_status='PASS',mechanical_smoke_inputs=1,native_wall_seconds_cap=1800,
            max_active_constraints=2,max_edit_steps=1)
        binding,approval,trusted=gate_fixture(config,('NATIVE_SMOKE','PILOT'))
        require_external_approval('NATIVE_SMOKE',binding,approval,config,trusted_authorization=trusted)
        pilot=dict(config,execution_stage='PILOT',current_state='PILOT_ALLOWED',allow_training=True,native_smoke_status='PASS',
                   data_audit_status='PASS',legal_fit_inputs=1,preapproved_global_rule_digest='fixture')
        binding,approval,trusted=gate_fixture(pilot,('NATIVE_SMOKE','PILOT'))
        with self.assertRaises(PermissionError):
            require_external_approval('PILOT',binding,approval,pilot,trusted_authorization=trusted)
        pilot['scientific_role_counts']={r:1 for r in ['EDIT_FIT','GEN_FIT','PROTECT_BG_FIT','PROTECT_NEAR_FIT']}
        binding,approval,trusted=gate_fixture(pilot,('NATIVE_SMOKE','PILOT'))
        require_external_approval('PILOT',binding,approval,pilot,trusted_authorization=trusted)
        pilot['preapproved_global_rule_digest']=None
        binding,approval,trusted=gate_fixture(pilot,('NATIVE_SMOKE','PILOT'))
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
