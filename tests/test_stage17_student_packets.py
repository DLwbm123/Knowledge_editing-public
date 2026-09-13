"""Routed reuse needs matching tokens/state, not just the answer string."""
import copy
import unittest
from scripts.medtrace.stage17_student_packets import mode_sources


class Routing(unittest.TestCase):
    def test_strict_effective_output(self):
        base = dict(raw_answer='same',raw_token_ids=[1])
        forced = dict(raw_answer='same',raw_token_ids=[2])
        row = dict(FORCED_ON=forced,R0=forced,RC=base,canonical_cap=1024,
            distance=0.9,radius_R0=1.0,route_on=dict(R0=True,RC=False))
        self.assertEqual(mode_sources(row,base),dict(FORCED_ON='student',R0='student',RC='Base'))
        for key, value in [('RC',forced),('distance',float('nan')),('route_on',dict(R0=True,RC=True))]:
            broken = copy.deepcopy(row); broken[key] = value
            with self.assertRaises(ValueError): mode_sources(broken,base)


if __name__ == '__main__':
    unittest.main()
