"""The future-child gate must never invoke training or accept unassigned work."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from scripts.medtrace import stage17_split as split


class SplitGateTest(unittest.TestCase):
    def test_gate_requires_assigned_completed_validated_phase(self):
        api = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = dict(run=str(root), operational_assignment='BELORA_GPU3_SPLIT_V1')
            with patch.object(split, 'frozen_api', return_value=(None, api)), patch.dict(
                    os.environ, JOB_METHOD='belora', JOB_MODE='single'):
                (root / 'STOP').touch()
                with self.assertRaisesRegex(RuntimeError, 'Split stopped'):
                    split.gate(cfg)
                api.validate_phase.assert_not_called()
                (root / 'STOP').unlink()
                phase = root / 'private/belora_single'
                phase.mkdir(parents=True)
                (phase / 'CLEANUP.json').write_text('{}')
                split.gate(cfg)
                api.validate_phase.assert_called_once_with(root, cfg, 'single', method='belora')
                api.validate_phase.side_effect = ValueError('bad binding')
                with self.assertRaisesRegex(ValueError, 'bad binding'):
                    split.gate(cfg)
                with patch.dict(os.environ, JOB_METHOD='grace'):
                    with self.assertRaisesRegex(ValueError, 'Unassigned'):
                        split.gate(cfg)
        with self.assertRaises(ValueError):
            split.selected({})


if __name__ == '__main__':
    unittest.main()
