import tempfile
import unittest
from pathlib import Path
import build as b

class SafetyTests(unittest.TestCase):
    def edit(self,q='What pathology is seen in this image?',target='brain tumor',position=3):
        return dict(original_question=q,target_answer=target,question_operator=b.operator(q),position=position)
    def source(self,q='What is the pathology?',answer='edema',**kw):
        return dict(source_dataset='VQA-RAD',original_question=q,original_answer=answer,source_question_id='x',source_text=q+'\n'+answer,**kw)
    def test_unmentioned_not_negative(self):
        self.assertEqual(b.decide(self.edit(),self.source())[0],'UNKNOWN')
    def test_coexistent_diseases(self):
        self.assertEqual(b.decide(self.edit(target='brain tumor'),self.source(answer='edema'))[1],'UNKNOWN')
    def test_synonymous_answers(self):
        self.assertTrue(b.compatible('MR-FLAIR','MR FLAIR'))
    def test_negation_uncertainty_side_time_retained(self):
        q='Is there possibly no lesion on the left after surgery?'
        self.assertIn('possibly',b.qualifiers(q));self.assertIn('left',b.qualifiers(q))
        self.assertEqual(b.decide(self.edit(q,'No'),self.source('Is there a lesion?','Yes'))[0],'UNKNOWN')
    def test_incomplete_list(self):
        self.assertEqual(b.decide(self.edit('Which organs belong to the digestive system?','rectum, colon'),self.source(answer='stomach'))[0],'UNKNOWN')
    def test_derived_label_not_gold(self):
        self.assertEqual(b.decide(self.edit(),self.source(answer='brain tumor',metadata_origin='LLM_DERIVED'))[0],'MODEL_ONLY')
    def test_bad_span(self):
        self.assertFalse(b.support_check(['not in original'],'original'));self.assertFalse(b.support_check([],''))
    def test_bound_span(self):
        self.assertTrue(b.support_check(['original'],'the original evidence'))
    def test_eval_not_fit(self):
        row=dict(assigned_role='EVAL',evidence_level='SOURCE_VERIFIED')
        self.assertFalse(b.admit(row,'FIT'))
    def test_sealed_loader(self):
        with self.assertRaises(ValueError):b.fit_load('/tmp/sealed_eval/h_eval_sealed.jsonl')
    def test_protected_or_future_not_fit(self):
        row=dict(assigned_role='FIT',evidence_level='SOURCE_VERIFIED',source_role_check='FAIL',source_supports_h_answer='YES',target_relation='CONTRADICTS',same_question_operator=True,citation_check='PASS')
        self.assertFalse(b.admit(row,'FIT'))
    def test_model_only_excluded(self):
        self.assertFalse(b.admit(dict(assigned_role='FIT',evidence_level='MODEL_ONLY'),'FIT'))
    def test_same_image_rewrite_cannot_cross_roles(self):
        rows=[dict(source_group_id='case1',image_sha256='same',assigned_role='FIT'),dict(source_group_id='case1',image_sha256='same',assigned_role='EVAL')]
        with self.assertRaises(ValueError):b.validate_role_isolation(rows)
    def test_training_source_cannot_be_independent_eval(self):
        rows=[dict(source_group_id='case1',image_sha256='a',assigned_role='FIT'),dict(source_group_id='case1',image_sha256='b',assigned_role='EVAL')]
        with self.assertRaises(ValueError):b.validate_role_isolation(rows)
    def test_unexecuted_api_no_pass(self):
        d=b.decide(self.edit(),self.source());self.assertEqual(d[0],'UNKNOWN')
    def test_modality_task_not_diagnosis(self):
        self.assertEqual(b.decide(self.edit(),self.source('What is the modality?','CT'))[0],'UNKNOWN')
    def test_ct_and_mri_not_synonyms(self):
        e=self.edit('What is the imaging modality?','MRI',49)
        self.assertEqual(b.decide(e,self.source('What image modality is this?','CT'))[0],'SOURCE_VERIFIED')
    def test_compatible_suprasellar(self):
        self.assertTrue(b.compatible('sella and suprasellar cistern','Suprasellar cistern'))

if __name__=='__main__':unittest.main()
