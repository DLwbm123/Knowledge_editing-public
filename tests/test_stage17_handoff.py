import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
from scripts.medtrace.stage17_handoff import all_phases

class HandoffTest(TestCase):
    def test_no_completion_until_every_phase_validates(self):
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.medtrace.stage17_handoff.validate_phase') as validate:
            root=Path(tmp)
            self.assertFalse(all_phases(root, {}, ('grace',)))
            validate.assert_not_called()
            for mode in ('single','sequential'):
                p=root/'private'/('grace_'+mode);p.mkdir(parents=True)
                (p/'CLEANUP.json').write_text('{}')
            self.assertTrue(all_phases(root, {}, ('grace',)))
            self.assertEqual(validate.call_count,2)
            validate.side_effect=ValueError('binding mismatch')
            with self.assertRaises(ValueError):all_phases(root, {}, ('grace',))

if __name__=='__main__':main()
