import unittest
from scripts.medtrace.stage17_report import metric


class Metrics(unittest.TestCase):
    def test_frozen_denominator_and_macro(self):
        rows=[dict(edit='a',base=True,correct=False),dict(edit='a',base=False,correct=True),
              dict(edit='b',base=True,correct=True),dict(edit='b',base=True,correct=False)]
        result,_=metric(rows,'correct',lambda r:r['base'])
        self.assertEqual((result['numerator'],result['probes'],result['macro']),(1,3,.25))
        self.assertIsNone(metric([],'correct')[0]['macro'])


if __name__=='__main__': unittest.main()
