import tempfile
import unittest
from pilot import audit_payload,valid_spans
from runner import Runner
from audit import intersections
from finalize import classify_result

class Boundaries(unittest.TestCase):
    def test_auditor_whitelist(self):
        edit=dict(edit_id='fictional',original_question='Which toy?',target_answer='cube',attribute_relation='object',qualifiers_full_text='Which toy?')
        source=dict(source_id='fictional-source',source_dataset='FICTIONAL',source_text='A sphere.',evidence_ids=['s1'],evidence_origin='ORIGINAL_ANNOTATION',image_binding='single imaginary drawing')
        candidate=dict(h_question='Which toy?',h_answer='sphere',evidence_ids=['s1'],evidence_spans=['A sphere.'],candidate_verdict='SUPPORTED',decisive_reason='GENERATOR_SECRET',confidence=.99)
        payload=audit_payload(edit,source,candidate)
        self.assertNotIn('GENERATOR_SECRET',str(payload));self.assertNotIn('candidate_verdict',str(payload));self.assertNotIn('confidence',str(payload))
        self.assertEqual(set(payload),{'edit_record','candidate_H','source_record'})
    def test_quote_guard(self):
        self.assertTrue(valid_spans(['sphere'],'A sphere.'));self.assertFalse(valid_spans(['cube'],'A sphere.'))
    def test_unknown_outbound_does_not_execute(self):
        with tempfile.TemporaryDirectory() as p:
            runner=Runner(p,'/usr/bin/true')
            self.assertEqual(runner.call('A',{},'fictional',{},'UNKNOWN')['status'],'OUTBOUND_BLOCKED')
            self.assertEqual(runner.count,0)
    def test_disagreement_preserved_unknown_and_no_verified_upgrade(self):
        row=dict(construction_status="MODEL_REVIEW_COMPLETED",generator_output={"candidate_verdict":"REFUTED"},audit_output={"independent_verdict":"UNKNOWN"},evidence_level="REJECTED",final_status="REJECTED")
        self.assertEqual(classify_result(row)["evidence_level"],"UNKNOWN")
        row.update(generator_output={"candidate_verdict":"SUPPORTED"},audit_output={"independent_verdict":"PASS"},evidence_level="SOURCE_GROUNDED",final_status="ACCEPTED_GROUNDED_REVIEWED")
        result=classify_result(row)
        self.assertEqual(result["evidence_level"],"SOURCE_GROUNDED");self.assertFalse(result["clinical_verified"])
    def test_role_intersection_is_ids_not_pair_counts(self):
        def row(role,group,grade='SOURCE_VERIFIED'):
            return dict(edit_id='e1',position=1,assigned_role=role,source_group_id=group,evidence_level=grade)
        result=intersections([row('FIT','f'),row('FIT','f')],[row('EVAL','a'),row('EVAL','b')])
        vv=next(x for x in result if x['intersection']=='FIT_verified_EVAL_verified')
        self.assertEqual((vv['fit_groups'],vv['eval_groups'],vv['group_overlap']),(1,2,0))

if __name__=='__main__':unittest.main()
