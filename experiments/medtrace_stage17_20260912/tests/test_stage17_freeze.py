"""Accepted Base identity and verdict coverage are a pre-training trust boundary."""
import unittest
from scripts.medtrace.stage17_freeze import accepted
from scripts.medtrace.stage17_prepare import digest


class Freeze(unittest.TestCase):
    def test_fixed_routes(self):
        from scripts.medtrace.stage17_single import routes
        self.assertEqual(routes(0.8, 1), {'R0':True, 'RC':False})
        self.assertEqual(routes(1, 1), {'R0':True, 'RC':False})
        self.assertEqual(routes(1.1, 1), {'R0':False, 'RC':False})

    def test_binding_and_coverage(self):
        b = dict(query_id='q', judge=dict(protocol='p', model='m', immutable_snapshot=None))
        opaque = digest(b)
        v = dict(query_id='q', opaque_query_id=opaque, protocol='p', judge_model='m',
                 immutable_snapshot=None, is_correct=False)
        self.assertEqual(accepted({opaque:b}, [v]), {'q':False})
        for bad in ([v,v], [dict(v,is_correct=0)], [dict(v,query_id='other')]):
            with self.assertRaises(ValueError):
                accepted({opaque:b}, bad)
        with self.assertRaises(ValueError):
            accepted({opaque:dict(b, changed=True)}, [v])


if __name__ == '__main__':
    unittest.main()
