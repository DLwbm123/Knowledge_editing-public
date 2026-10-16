"""Recovery must reuse bound outputs without model calls or budget changes."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from . import launch


def run():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with patch.dict(os.environ, RUN_ROOT=temp, DATA_ROOT=temp+'/data', RECOVERY_ID='1'):
            from experiments.purew_original146_v1 import common
            with patch.object(common, 'RUN', root), patch.object(launch, 'helpers', return_value=(common, root)):
                assert launch.image_path('/old/derived_inputs/a.png') == root/'private/derived_inputs/a.png'
                assert launch.image_path('/old/DataP/a.png') == root/'data/a.png'
                row = dict(opaque_Base_id='id', image_path='/old/derived_inputs/a.png')
                bindings = {'id': {'locked': True}}
                phase = dict(arm='U', mode='base', prefix=0, owner=0, code='new')
                prior = dict(phase, code='old')
                dest = root/'private/outputs/a.json'
                old = dict(binding=dict(input=row, judge_input=bindings['id'], phase=prior), R0=dict(raw_answer='kept', raw_token_ids=[1,2]))
                common.write(dest, old)
                common.write(dest.with_suffix('.STARTED.json'), dict(phase=prior))
                common.write(root/'private/RECOVERY1.json', dict(authorized=True, previous_commit='old'))
                original = dest.read_bytes()
                result = launch.generate(None, bindings, row, dest, phase, None)
                assert result == old and dest.read_bytes() == original
                try:
                    launch.generate(None, bindings, dict(row, image_path='changed'), dest, phase, None)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('Changed binding accepted')
                dest.unlink()
                try:
                    launch.generate(None, bindings, row, dest, phase, None)
                except FileExistsError:
                    pass
                else:
                    raise AssertionError('Orphan attempt replayed')
    return dict(status='PASS', derived_and_DataP_paths=True, bound_output_reused_without_model_or_budget=True,
                changed_binding_rejected=True, orphan_attempt_rejected=True)
