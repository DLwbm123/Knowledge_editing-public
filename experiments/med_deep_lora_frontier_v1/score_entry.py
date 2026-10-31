import os
from pathlib import Path
import runpy
mode=os.environ['ENTRY_MODE']
if mode not in ('score_guard','score_driver'):
    raise ValueError('Unknown score stage')
runpy.run_path(str(Path(os.environ['FIXED_WORKTREE'])/'experiments/med_deep_lora_frontier_v1'/(mode+'.py')),run_name='__main__')
