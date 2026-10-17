"""Finish only the unexecuted P reload under the user's resource-gate waiver."""
import os
from pathlib import Path
import shutil
import subprocess
import time
from . import launch
from .configuration import CAPS
from .executor import Budget


def remove_resource_gates():
    c, root = launch.helpers()
    def check(self):
        if set(self.data['counts']) != set(CAPS):
            raise ValueError('Budget schema mismatch')
        for key, cap in CAPS.items():
            if type(self.data['counts'][key]) is not int or not 0 <= self.data['counts'][key] <= cap:
                raise ValueError('Invalid counter: '+key)
    def budget():
        if (root/'STOP').exists():
            raise RuntimeError('Explicit stop requested')
        if shutil.disk_usage(root).free < c.read(root/'RUN_MANIFEST.json')['min_free_bytes']:
            raise OSError('Disk reserve reached')
    def available(gpu):
        budget()
        actual = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if actual != c.read(root/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]:
            raise ValueError('GPU identity changed')
        return True
    Budget.check = check
    launch.GPU_SECONDS = float('inf')
    c.budget, c.available = budget, available


def main():
    c, root = launch.helpers()
    assert c.read(root/'public/RESOURCE_GATE_WAIVER.json')['authorized'] is True
    assert not (root/'private/FINAL_CONSUMED_P.json').exists()
    assert not list((root/'private/outputs/P/reload/8').glob('*.json'))
    assert all(s.get('ended_epoch') for s in c.read(root/'RESOURCE_LEDGER.json')['gpu_sessions'])
    remove_resource_gates()
    config = c.read(root/'private/config.json')
    c.write(root/'public/PROGRESS.json',dict(status='RUNNING',stage='reload_P_8',epoch=time.time()))
    try:
        launch.auxiliary('P','reload',8,config)
        assert c.read(root/'private/FINAL_CONSUMED_P.json')['status']=='PASS'
        active=root/'private/fixed8/P/active.pt'
        assert not active.is_symlink()
        receipt=dict(bytes=active.stat().st_size,reason='All insertion/prefix/native-reload consumers durable',retained='current native export')
        active.unlink()
        c.write(root/'public/OPTIMIZER_RETIRED_P.json',receipt)
        c.write(root/'private/GENERATION_COMPLETE.json',dict(status='PASS',scoring_pending=True))
        c.write(root/'public/PROGRESS.json',dict(status='GENERATION_COMPLETE_SCORING_PENDING',epoch=time.time()))
    except BaseException:
        import traceback
        c.write(root/'private/failures/completion.json',dict(traceback=traceback.format_exc(),epoch=time.time()))
        c.write(root/'public/PROGRESS.json',dict(status='STOPPED_COMPLETION_FAILURE',epoch=time.time()))
        raise


if __name__=='__main__':main()
