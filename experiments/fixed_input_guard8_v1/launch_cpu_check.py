"""No-model checks for the fresh launch waiver and frozen comparison reader."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


def run():
    with tempfile.TemporaryDirectory(prefix='e.',dir=os.environ.get('CHECK_TMPDIR')) as temporary:
        root = Path(temporary)
        os.environ.update(RUN_ROOT=str(root),TRAIN_PYTHON='python',NEUTRAL_ENTRY='e.py')
        from . import launch, complete, worker
        from .executor import Budget
        from .configuration import CAPS
        c, _ = launch.helpers()
        c.write(root/'RUN_MANIFEST.json',dict(deadline_epoch=None,min_free_bytes=0,GPU_seconds_limit=None))
        c.write(root/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=10**12,gpu_sessions=[],Judge_attempts=0))
        c.write(root/'PLAN_CONFIG.json',dict(hardware=dict(UUIDs={'7':'synthetic-uuid'})))
        complete.remove_resource_gates()
        c.write(root/'budget.json',dict(start_epoch=1,counts={k:0 for k in CAPS}))
        budget = Budget(root/'budget.json');budget.check();c.budget()
        with patch.object(complete.subprocess,'check_output',return_value='synthetic-uuid\n') as query:
            assert c.available(7)
            assert query.call_args.args[0] == ['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader']
        with patch.object(complete.subprocess,'check_output',return_value='wrong-uuid\n'):
            try:c.available(7)
            except ValueError:pass
            else:raise AssertionError('GPU identity binding lost')
        assert launch.GPU_SECONDS == worker.GPU_SECONDS == float('inf')
        budget.data['counts']['forwards'] = CAPS['forwards']+1
        try:budget.check()
        except ValueError:pass
        else:raise AssertionError('Count cap lost')
        (root/'STOP').touch()
        try:c.budget()
        except RuntimeError:pass
        else:raise AssertionError('Manual STOP lost')
        (root/'STOP').unlink();(root/'logs').mkdir()
        class Child:
            pid = 123
            def wait(self,timeout):
                assert timeout is None,'Controller still applies a wall deadline'
                return 0
        with patch.object(launch,'plan',return_value=[('U','base',0)]),patch.object(launch.subprocess,'Popen',return_value=Child()):
            launch.controller()
        assert c.read(root/'private/GENERATION_COMPLETE.json')['status']=='PASS'
    from .compare import long_outputs
    old = Path('/data/bmw/Knowledge_editing/outputs/fixed-input-guard8-v1-20261016/run')
    output = long_outputs(old)
    assert output['P']['prefix8']['protection']['cap_hits']==27
    assert output['P']['prefix8']['protection']['cap_hits_wrong']==26
    return dict(status='PASS',expired_clock_and_huge_gpu_usage_permitted=True,uuid_only_admission=True,
                controller_wait_has_no_deadline=True,worker_gpu_seconds_unlimited=True,count_caps_and_manual_stop_preserved=True,
                old_reference_length_reader_verified=True,models_loaded=0,GPU_calls=0)


if __name__=='__main__':print(json.dumps(run()))
