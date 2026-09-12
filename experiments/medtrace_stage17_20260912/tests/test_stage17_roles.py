import unittest
from scripts.medtrace.stage17_roles import filter_events, identity_closure


class Roles(unittest.TestCase):
    def test_role_closure_and_order_without_scores(self):
        rows=[dict(dataset='SLAKE',image_path='/imgs/xmlab1/source.jpg',image_sha256='h1',query_id='q1'),
              dict(dataset='SLAKE',image_path='/imgs/xmlab2/source.jpg',image_sha256='h1',query_id='q2'),
              dict(dataset='SLAKE',image_path='/imgs/xmlab2/source_blur.jpg',image_sha256='h2',query_id='q3'),
              dict(dataset='SLAKE',image_path='/imgs/xmlab3/source.jpg',image_sha256='h2',query_id='q4'),
              dict(dataset='SLAKE',image_path='/imgs/xmlab4/source.jpg',image_sha256='h3',query_id='q5')]
        blocked=identity_closure({('SLAKE','xmlab1')},rows)
        self.assertEqual(blocked,{('SLAKE','xmlab1'),('SLAKE','xmlab2'),('SLAKE','xmlab3')})
        events=[dict(event_id='e1',edit_query_id='q4'),dict(event_id='e2',edit_query_id='q5')]
        kept,removed=filter_events(events,rows,blocked)
        self.assertEqual(kept,events[1:]);self.assertEqual(removed[0]['event_id'],'e1')


if __name__=='__main__':
    unittest.main()
