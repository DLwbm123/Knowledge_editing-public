import tempfile
import json
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
from scripts.medtrace.stage17_handoff import all_phases, tick

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


    def test_legacy_dispatch_uses_same_frozen_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';campaign=root/'campaign'
            for p in [base,campaign]:
                (p/'private').mkdir(parents=True);(p/'public').mkdir()
            bc=dict(freeze_id='f',N=1,order=['e1']);cc=dict(freeze_id='f',N=1)
            (base/'private/DISPATCH.json').write_text(json.dumps(bc))
            (campaign/'private/DISPATCH.json').write_text(json.dumps(cc))
            for name in ['EXTERNAL_LORA','EXTERNAL_BASELINES']:
                for suffix in ['','_IMPORTED']:
                    (campaign/'private'/(name+suffix+'.json')).write_text('{}')
            with patch('scripts.medtrace.stage17_handoff.all_phases',return_value=True), patch('scripts.medtrace.stage17_handoff.validate_phase') as validate:
                self.assertTrue(tick(dict(baseline=str(base),campaign=str(campaign))))
                self.assertEqual(validate.call_args.args[1]['order'],['e1'])
            cc['freeze_id']='different';(campaign/'private/DISPATCH.json').write_text(json.dumps(cc))
            with self.assertRaisesRegex(ValueError,'cohort mismatch'):
                tick(dict(baseline=str(base),campaign=str(campaign)))

if __name__=='__main__':main()
