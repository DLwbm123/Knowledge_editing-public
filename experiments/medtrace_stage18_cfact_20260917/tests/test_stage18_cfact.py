import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook
from methods.medtrace.selective_write import LowRankExpert,optimizer_for,predictor_mask,full_vocab_kl
from scripts.medtrace.stage18_cfact import (update,source_batch,check_cache,rng_state,resume,assert_base_off,state_hash,extra_schedule)
from scripts.medtrace.stage18_support import validate_task
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.run_selective_write import save
from m3bench_repro.editors.routing import MemoryRouter


@dataclass
class Record:
    question:str='native'
    image_path:Path=Path('/native.jpg')
    target:str='native answer'
    official_rephrase:str='fit'


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layer=torch.nn.Linear(12,8); self.requires_grad_(False)
    def forward(self,inputs_embeds,**_): return SimpleNamespace(logits=self.layer(inputs_embeds))


def toy():
    torch.manual_seed(17); model=Model()
    cp=AsymmetricCPExpert(12,8,4); cp.rho.data.fill_(.3)
    expert=LowRankExpert(cp,17,rank=4)
    def batch(label,offset):
        return SimpleNamespace(inputs_embeds=torch.randn(1,6,12)+offset,labels=torch.tensor([[-100,-100,-100,label,2,-100]]),target_token_ids=(label,2))
    batches=[batch(3,0.),batch(3,.2),batch(4,1.2),batch(5,-.8)]
    def loss(b):
        logits=model(inputs_embeds=b.inputs_embeds).logits
        return torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,8),b.labels[:,1:].reshape(-1),ignore_index=-100)
    runtime=SimpleNamespace(model=model,compute_loss=loss)
    u=batch(6,2.); mask=predictor_mask(u.labels)
    teacher=({'inputs_embeds':u.inputs_embeds},u.labels,mask,model(inputs_embeds=u.inputs_embeds).logits[mask].detach().log_softmax(-1),{'role':'U_fit'})
    return runtime,expert,batches,teacher


class Stage18Test(unittest.TestCase):
    def test_dev_dispatch_requires_exact_reviewed_cohort(self):
        from scripts.medtrace.stage18_smoke import validate_dispatch
        from scripts.medtrace.stage18_cfact import BRANCHES
        tasks=dict(scope='REVIEWED_EXISTING_DATA_DEV_V2',tasks=[dict(canonical_edit_id='fictional')])
        tasks['freeze_id']=digest(tasks['tasks'])
        cfg=dict(mode='EXPLORATORY_DEV_PILOT',branches=list(BRANCHES),freeze_id=tasks['freeze_id'])
        gate=dict(training_freeze=tasks['freeze_id'],formal_eligible=False,selected=[dict(candidate_id='fictional',Base_wrong=True,native_supported=True,relations_supported=True,support_supported=True)])
        self.assertFalse(validate_dispatch(cfg,tasks,gate))
        for field in ('Base_wrong','native_supported','relations_supported','support_supported'):
            bad=copy.deepcopy(gate);bad['selected'][0][field]=False
            with self.assertRaises(ValueError):validate_dispatch(cfg,tasks,bad)
        with self.assertRaises(ValueError):validate_dispatch(dict(cfg,mode='SOURCE_LABEL_SMOKE'),tasks,gate)
        with self.assertRaises(ValueError):validate_dispatch(dict(cfg,freeze_id='changed'),tasks,gate)

    def test_pilot_metrics_keep_failed_native_and_shared_H_denominator(self):
        from scripts.medtrace.stage18_score import aggregate,score_key
        rows=[dict(edit_id='a',role='H_eval',query_id='shared',image='image',base_correct=True,post_correct=True,native_correct=False,agreement=True,on=False),
              dict(edit_id='b',role='H_eval',query_id='shared',image='image',base_correct=True,post_correct=True,native_correct=True,agreement=True,on=False)]
        value=aggregate(rows)
        self.assertEqual(value['PairCorrect']['edit_macro'],.5)
        self.assertEqual(value['PairCorrect']['supported_edits'],2)
        self.assertEqual(value['H_eval']['unique_QA'],1)
        self.assertIsNone(value['U_eval']['retention']['edit_macro'])
        q=dict(image_sha256='image',question='Q?',reference='Yes');out=dict(raw_answer='Yes')
        self.assertNotEqual(score_key(q,out),score_key(dict(q,reference='No'),out))
        self.assertEqual(score_key(q,out),score_key(q,dict(out,seconds=1.)))

    def test_actual_H_G_gradient_and_zero_weight(self):
        runtime,template,batches,teacher=toy(); states={}
        for role,index,weight in [('none',None,1),('H0',2,0),('G0',3,0),('H',2,1),('G',3,1)]:
            e=copy.deepcopy(template); hook=MedTraceLayerHook(runtime.model.layer,e);hook.attach()
            try:
                value=update(runtime,hook,e,optimizer_for(e,runtime.model),*batches[:2],teacher,
                    None if index is None else (batches[index],{'role':role}),extra_weight=weight)
                if index is not None:
                    grad=value['terms']['extra']['weighted_gradient_norm']
                    self.assertGreater(grad,0) if weight else self.assertEqual(grad,0)
                states[role]=copy.deepcopy(e.state_dict())
            finally: hook.detach()
        for k in states['none']:
            self.assertTrue(torch.equal(states['none'][k],states['H0'][k]))
            self.assertTrue(torch.equal(states['none'][k],states['G0'][k]))
        self.assertTrue(any(not torch.equal(states['none'][k],states['H'][k]) for k in states['none']))
        self.assertTrue(any(not torch.equal(states['H'][k],states['G'][k]) for k in states['H']))
        self.assertTrue(all(p.grad is None for p in runtime.model.parameters()))

    def test_source_own_image_question_answer(self):
        got=source_batch(SimpleNamespace(build_edit_batch=lambda r:r),Record(),dict(image_path='/H.jpg',question='H question',reference='own answer'))
        self.assertEqual((str(got.image_path),got.question,got.target,got.official_rephrase),('/H.jpg','H question','own answer',''))

    def test_resume_same_next_update_and_rng(self):
        r,e,b,t=toy(); opt=optimizer_for(e,r.model);hook=MedTraceLayerHook(r.model.layer,e);hook.attach()
        with tempfile.TemporaryDirectory() as d:
            try:
                update(r,hook,e,opt,*b[:2],t,(b[2],{'role':'H_fit'})); binding={'W0':'test','seed':17}
                save(Path(d)/'x.pt',dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),step=1,curve=[],**rng_state()))
                expected_rng=torch.rand(3)
                update(r,hook,e,opt,*b[:2],t,(b[2],{'role':'H_fit'})); expected=copy.deepcopy(e.state_dict())
                step,_=resume(Path(d)/'x.pt',binding,e,opt); self.assertEqual(step,1);self.assertTrue(torch.equal(torch.rand(3),expected_rng))
                update(r,hook,e,opt,*b[:2],t,(b[2],{'role':'H_fit'}))
                self.assertTrue(all(torch.equal(expected[k],v) for k,v in e.state_dict().items()))
                with self.assertRaises(ValueError): resume(Path(d)/'x.pt',{'W0':'changed'},e,opt)
            finally:hook.detach()

    def test_KL_direction_full_vocab_shift_EOS_and_cache(self):
        labels=torch.tensor([[-100,-100,4,2,-100]]); attention=torch.tensor([[1,1,1,1,0]])
        self.assertEqual(predictor_mask(labels,attention).nonzero().tolist(),[[0,1],[0,2]])
        q=torch.tensor([[1.,2.,4.,0.,-1.]]).log_softmax(-1); logits=torch.tensor([[3.,0.,1.,2.,5.]],requires_grad=True)
        expected=(q.exp()*(q-logits.log_softmax(-1))).sum()
        self.assertTrue(torch.allclose(full_vocab_kl(logits,q),expected))
        full_vocab_kl(logits,q).backward(); self.assertTrue(torch.all(logits.grad!=0))
        cache={'binding':{'temperature':1,'runtime':'A','mask':[1,2]},'logp':q}
        self.assertTrue(torch.equal(check_cache(cache,cache['binding']),q))
        for key,val in [('temperature',2),('runtime','B'),('mask',[2,3])]:
            with self.assertRaises(ValueError): check_cache(cache,dict(cache['binding'],**{key:val}))

    def test_base_route_isolation_OFF_zero_and_global_nearest(self):
        r,e,b,_=toy(); base=r.model(inputs_embeds=b[0].inputs_embeds).logits
        hook=MedTraceLayerHook(r.model.layer,e);hook.attach()
        try:
            assert_base_off(r); self.assertTrue(torch.equal(base,r.model(inputs_embeds=b[0].inputs_embeds).logits))
            hook.set_teacher_routing(b[0].labels)
            with self.assertRaises(ValueError):assert_base_off(r)
            e.B.data.zero_();self.assertTrue(torch.equal(base,r.model(inputs_embeds=b[0].inputs_embeds).logits))
            hook.clear_request_routing(); assert_base_off(r)
        finally:hook.detach()
        router=MemoryRouter('euclidean');router.add('first',torch.tensor([0.]),.1);router.add('second',torch.tensor([1.]),10)
        decision=router.route(torch.tensor([.4])); self.assertFalse(decision.activated);self.assertEqual(decision.nearest_logical_edit_id,'first')
        router.radii[0]=10; self.assertEqual(router.route(torch.tensor([.5])).logical_edit_id,'first')

    def test_pilot_isolation_checks_future_native_and_hash_alias(self):
        from scripts.medtrace.stage18_pilot import isolation
        row=lambda n,h,role:dict(dataset='SLAKE',image_path=f'/imgs/xmlab{n}/source.jpg',image_sha256=h,role=role)
        native=row(1,'one','native');aux=row(2,'two','H_fit');evaluation=row(3,'three','H_eval')
        self.assertEqual(len(isolation([native,aux],[evaluation])[0]),2)
        with self.assertRaises(ValueError):isolation([native,aux,row(2,'two','native')],[evaluation])
        with self.assertRaises(ValueError):isolation([native,aux],[row(9,'two','H_eval')])

    def test_base_deduplicates_without_accepting_label_conflicts(self):
        from scripts.medtrace.stage18_base import rows_for
        row=dict(image_sha256='fictional',question='Question?',reference='Yes')
        task=dict(native=row,H_fit=[dict(row)],G_fit=[],U_fit=[])
        packet=dict(candidate_packages=[dict(training=task,evaluation=[dict(row)])])
        self.assertEqual(len(rows_for(packet)),1)
        packet['candidate_packages'][0]['evaluation'][0]['reference']='No'
        with self.assertRaises(ValueError):rows_for(packet)

    def test_original_binary_labels_are_candidates_not_rewritten_answers(self):
        from scripts.medtrace.stage18_support import conflict
        a=dict(dataset='SLAKE',image_path='/imgs/xmlab9001/source.jpg',source_group='a',question='Does the picture contain heart?',reference='No')
        b=dict(a,image_path='/imgs/xmlab9002/source.jpg',source_group='b',reference='Yes')
        self.assertTrue(conflict(a,b))
        self.assertFalse(conflict(a,dict(b,reference='No')))
        self.assertFalse(conflict(a,dict(b,question='Does the picture contain liver?')))
        self.assertFalse(conflict(a,dict(b,image_path=a['image_path'])))
        self.assertEqual(a['reference'],'No')

    def test_existing_source_identity_covers_derived_images(self):
        from scripts.medtrace.stage18_existing import source_groups
        self.assertEqual(source_groups(dict(dataset='SLAKE',rows=[dict(image_path='/derived/xmlab9001/source_blur.jpg'),dict(image_path='/imgs/xmlab9001/source.jpg')])),{('SLAKE','xmlab9001')})

    def test_qualification_keeps_review_and_base_gates(self):
        from scripts.medtrace.stage18_pilot import qualify
        n=dict(image_sha256='fictional-native',question='Question?',reference='A')
        h=dict(n,image_sha256='fictional-fit',reference='B')
        e=dict(h,image_sha256='fictional-eval',role='H_eval')
        t=dict(canonical_edit_id='fictional-edit',native=n,H_fit=[h])
        packet=dict(candidate_packages=[dict(candidate_id=t['canonical_edit_id'],training=t,evaluation=[e])])
        packet['freeze_id']=digest(packet)
        reviews=dict(records=[dict(review_id=digest([t['canonical_edit_id'],role,row])[:20],verdict='SUPPORTED')
            for role,row in [('H_fit',h),('H_eval',e)]])
        base=dict(packet_binding=digest(packet),records=[dict(query_id=digest([n['image_sha256'],n['question']]),source=n)])
        verdicts=dict(source_binding=digest(base),decisions=[dict(opaque_query_id=digest(base['records'][0]),is_correct=False)])
        self.assertTrue(qualify(packet,reviews,base,verdicts)[0]['strict_image_level_DEV_eligible'])
        reviews['records'][0]['verdict']='UNVERIFIED'
        self.assertFalse(qualify(packet,reviews,base,verdicts)[0]['strict_image_level_DEV_eligible'])
        verdicts['source_binding']='wrong'
        with self.assertRaises(ValueError):qualify(packet,reviews,base,verdicts)

    def test_schema_rejects_eval_missingH_and_wrong_fit(self):
        # Fictional image/qid identities; these fixtures are not source patient records.
        n=dict(dataset='SLAKE',image_path='/imgs/xmlab9001/source.jpg',image_sha256='n',source_group='SLAKE:xmlab9001',question='What is the largest organ in the picture?',reference='Lung',role='native',source_qid=None)
        h=dict(n,image_path='/imgs/xmlab9002/source.jpg',image_sha256='h',source_group='SLAKE:xmlab9002',reference='Liver',role='H_fit',source_qid=900010)
        g=dict(h,question='What modality is used to take this image?',reference='CT',role='G_fit',source_qid=900007)
        u=dict(g,question='Does the picture contain liver?',reference='Yes',role='U_fit',source_qid=900011);q=n['question']
        t=dict(canonical_edit_id='id',order=1,seed=int(digest([20260912,'id'])[:8],16),native=n,H_fit=[h],G_fit=[g],U_fit=[u],fit_questions=[f'Please answer the following question: {q}',f'Question: {q}',f'{q} Please provide an answer.',f'Please respond to this question: {q}'])
        validate_task(t); self.assertEqual(extra_schedule(t),[0]*320)
        for bad in [dict(t,H_eval=[]),dict(t,H_fit=[]),dict(t,fit_questions=['eval']*4),dict(t,H_fit=[dict(h,reference='Lung')]),dict(t,H_fit=[dict(h,eval_answer='x')])]:
            with self.assertRaises(ValueError):validate_task(bad)

if __name__=='__main__': unittest.main()
