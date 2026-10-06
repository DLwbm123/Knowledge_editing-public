import unittest
from rules import automatic_decision,contrast_state,ct_relation,image_acquisition

class Regression(unittest.TestCase):
    def decide(self,text,target='Contrast CT with GI and IV contrast',**extra):
        return automatic_decision(dict(question_operator='ct_contrast',target_answer=target),dict(source_dataset='PMC_PRIMARY',source_text=text,**extra))
    def test_noncontrast_is_compatible(self):
        self.assertEqual(self.decide('Non-contrast CT shows a lesion.','Non-contrast CT')['relation'],'COMPATIBLE')
    def test_mixed_unbound(self):
        self.assertEqual(self.decide('(A) Non-contrast CT. (B) Contrast-enhanced CT.')['level'],'UNKNOWN')
    def test_other_exam(self):
        self.assertEqual(self.decide('Current ultrasound image. A previous non-contrast CT showed a lesion.')['level'],'UNKNOWN')
    def test_current_ct_unknown_other_exam_enhanced(self):
        self.assertEqual(self.decide('CT image shows a lesion. A contrast-enhanced MRI was also acquired.')['level'],'UNKNOWN')
    def test_explicit_current_image(self):
        self.assertEqual(self.decide('Non-contrast CT shows a lesion.')['level'],'SOURCE_VERIFIED')
    def test_qualifiers_not_binary(self):
        self.assertEqual(ct_relation(contrast_state('CT with oral contrast'),contrast_state('CT with IV contrast')),'UNKNOWN')
        self.assertEqual(contrast_state('CT without IV contrast')['routes'],{'IV':'NO'})
    def test_route_explicit_negation(self):
        self.assertEqual(ct_relation(contrast_state('CT without IV contrast'),contrast_state('CT with IV contrast')),'CONTRADICTS')
    def test_noncontrast_not_blanket_oral_negation(self):
        self.assertEqual(self.decide('Non-contrast CT shows a lesion.','CT with oral contrast')['level'],'UNKNOWN')
    def test_oral_without_iv_can_be_noncontrast_acquisition(self):
        self.assertEqual(ct_relation(contrast_state('CT with oral contrast without IV contrast'),contrast_state('Non-contrast CT')),'UNKNOWN')
    def test_mixed_bound_selected_panel(self):
        text='(A) Non-contrast CT. (B) Contrast-enhanced CT.'
        binding=dict(inspection='IMAGE_LABELS_CHECKED',image_labels=['a','b'],panels={'a':dict(modality='CT',span='Non-contrast CT'),'b':dict(modality='CT',span='Contrast-enhanced CT')})
        self.assertEqual(self.decide(text,panel_binding=binding)['state']['state'],'MIXED')
        self.assertEqual(self.decide(text,panel_binding=binding,bound_panel='a')['level'],'SOURCE_VERIFIED')
    def test_no_unchecked_all_panels(self):
        binding=dict(inspection='NOT_RUN',image_labels=['a'],panels={})
        self.assertEqual(self.decide('Non-contrast CT.',panel_binding=binding)['level'],'UNKNOWN')
    def test_modality_mentioned_negated(self):
        s=dict(source_text='Not a CT image; an MRI was previously acquired.')
        self.assertIsNone(image_acquisition(s)[0])
        self.assertIsNone(image_acquisition(dict(source_text='CT was not performed.'))[0])
        self.assertIsNone(image_acquisition(dict(source_text='Possible CT image.'))[0])
    def test_modality_comparison_multiframe(self):
        s=dict(source_text='(A) CT chest. (B) MRI abdomen.')
        self.assertIsNone(image_acquisition(s)[0])
    def test_generated_answer_second_check(self):
        d=self.decide('CT without IV contrast.','CT with IV contrast')
        self.assertEqual(d['relation'],'CONTRADICTS');self.assertEqual(d['state']['routes']['IV'],'NO')

if __name__=='__main__':unittest.main()
