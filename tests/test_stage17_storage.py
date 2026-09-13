import tempfile
import unittest
from pathlib import Path
from scripts.medtrace.stage17_single import checkpoint_budget


class Storage(unittest.TestCase):
    def test_budget_counts_missing_checkpoints_not_completed_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tasks=[dict(order=1),dict(order=2)]
            reserve=8*1024**3
            self.assertEqual(checkpoint_budget(root,tasks,100),reserve+200)
            directory=root/'private/single_BE/e001'; directory.mkdir(parents=True)
            (directory/'state.pt').write_bytes(b'existing')
            self.assertEqual(checkpoint_budget(root,tasks,100),reserve+100)
            (directory/'COMPLETE.json').write_text('{}')
            self.assertEqual(checkpoint_budget(root,tasks,100),reserve+100)
            with self.assertRaises(ValueError): checkpoint_budget(root,tasks,0)


if __name__=='__main__': unittest.main()
