import json
from pathlib import Path
import tempfile
import unittest

from scripts.medtrace.stage17_external import receive, validate_phase


class ExternalTests(unittest.TestCase):
    def test_only_complete_bound_cleaned_phases_are_adopted_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'private').mkdir()
            cfg=dict(freeze_id='f',runtime_lock={},code_commit='c',methods={},order=['e'],N=1)
            def put(p,value):
                p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value))
            put(root/'private/EXTERNAL_LORA.json',cfg)
            incoming=root/'private/external-incoming'
            for mode in ('single','sequential'):
                directory=incoming/'private'/f'lora_{mode}'
                phase=dict(freeze_id='f',method='lora',mode=mode,runtime={},code_commit='c',
                    method_lock={},order=['e'],prefixes=[1])
                put(directory/'BINDING.json',phase)
                put(directory/'COMPLETE.json',dict(status='GENERATED_NOT_SCORED',N=1,phase=phase))
                put(directory/'CLEANUP.json',dict(status='DELETED'))
            with self.assertRaises(ValueError):
                validate_phase(incoming,dict(cfg,code_commit='wrong'),'single')
            collision=root/'private/lora_single'; collision.mkdir()
            with self.assertRaises(RuntimeError): receive(root)
            self.assertFalse((root/'private/EXTERNAL_LORA_IMPORTED.json').exists())
            collision.rmdir(); receive(root)
            self.assertEqual(json.loads((root/'private/EXTERNAL_LORA_IMPORTED.json').read_text()),cfg)
            validate_phase(root,cfg,'single'); validate_phase(root,cfg,'sequential')


if __name__=='__main__': unittest.main()
