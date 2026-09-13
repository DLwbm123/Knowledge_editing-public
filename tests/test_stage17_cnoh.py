import unittest
from scripts.medtrace.stage17_cnoh import training_task


class Adapter(unittest.TestCase):
    def test_preserve_frozen_support_and_reject_missing_u(self):
        t=dict(edit_id='native',order=1,seed=17,native={'query_id':'native'},
            U_fit=[{'question':'frozen U'}],fit_questions=['a','b','c','d'],
            fit_status='SUPPORTED',roles={'U_fit':True})
        out=training_task(t)
        self.assertEqual(out['U'],t['U_fit'])
        self.assertEqual((out['canonical_edit_id'],out['seed']),('native',17))
        with self.assertRaises(ValueError): training_task(dict(t,U_fit=[]))


if __name__=='__main__': unittest.main()
