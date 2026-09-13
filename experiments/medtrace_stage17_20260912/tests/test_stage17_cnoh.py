import unittest
from scripts.medtrace.stage17_cnoh import binding_commit, training_task


class Adapter(unittest.TestCase):
    def test_recovery_preserves_completed_provenance_only(self):
        cfg=dict(N=146,code_commit='new',recovery=dict(completed_prefix=94,previous_code_commit='old'))
        self.assertEqual(binding_commit(cfg,94),'old')
        self.assertEqual(binding_commit(cfg,95),'new')
        with self.assertRaises(ValueError): binding_commit(dict(cfg,N=94),94)

    def test_preserve_frozen_support_and_reject_missing_u(self):
        t=dict(edit_id='native',order=1,seed=17,native={'query_id':'native'},
            U_fit=[{'question':'frozen U'}],fit_questions=['a','b','c','d'],
            fit_status='SUPPORTED',roles={'U_fit':True})
        out=training_task(t)
        self.assertEqual(out['U'],t['U_fit'])
        self.assertEqual((out['canonical_edit_id'],out['seed']),('native',17))
        with self.assertRaises(ValueError): training_task(dict(t,U_fit=[]))


if __name__=='__main__': unittest.main()
