"""Trust-boundary check: cache mismatch fails, empty valid generation is not discarded."""
import unittest
import json
from pathlib import Path
import subprocess
import tempfile
import sys
from scripts.medtrace.stage17_prepare import check_raw, digest, group


class Binding(unittest.TestCase):
    def test_binding_and_source_variants(self):
        query = dict(query_id='q', question='Q?', image_path='/imgs/xmlab8/source.jpg', image_sha256='a')
        raw = dict(error=None, runtime='official', image_sha256='a', prompt_token_ids=[1,-200],
            raw_generated_token_ids=[], generated_token_count=0, model_answer_raw='')
        tokenizer = type('Tokenizer', (), {'decode':lambda self, ids, **kw: ''})()
        check_raw(query, dict(query), raw, tokenizer, [1,-200])
        with self.assertRaises(ValueError):
            check_raw(query, dict(query, question='changed'), raw, tokenizer, [1,-200])
        with self.assertRaises(ValueError):
            check_raw(query, query, dict(raw, error='OOM'), tokenizer, [1,-200])
        with self.assertRaises(ValueError):
            check_raw(query, query, raw, tokenizer, [9,-200])
        self.assertEqual(group(dict(dataset='SLAKE', image_path='/imgs/xmlab8/source_blur.jpg')),
            ('SLAKE','xmlab8'))
        self.assertNotEqual(digest(raw), digest(dict(raw, model_answer_raw='different')))

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS process isolation')
    def test_actual_os_isolation_without_model_call(self):
        from scripts.medtrace.stage17_judge import profile, flags
        with tempfile.TemporaryDirectory(prefix='job.',dir='/private/tmp') as folder:
            root=Path(folder); work=root/'one'; sibling=root/'two'
            work.mkdir(); sibling.mkdir()
            (work/'input').write_text('allowed'); (sibling/'input').write_text('denied')
            policy=work/'policy'; policy.write_text(profile(work,[work,sibling],root/'private',root/'source'))
            for path, allowed in [(work/'input',True),(sibling/'input',False)]:
                proc=subprocess.run(['/usr/bin/sandbox-exec','-f',str(policy),'/bin/cat',str(path)],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                self.assertEqual(proc.returncode==0,allowed)
            config=flags(work)
            self.assertIn('features.shell_tool=false',config)
            self.assertIn('approval_policy="never"',config)
            self.assertIn('permissions.judge.network.enabled=false',config)


if __name__ == '__main__':
    unittest.main()
