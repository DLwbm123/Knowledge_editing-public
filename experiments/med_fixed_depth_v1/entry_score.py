"""Neutral local score entry."""
import os
from pathlib import Path
import runpy

mode = os.environ.get('ENTRY_MODE', 'score_guard')
if mode not in ('score_guard', 'score_worker'):
    raise ValueError('Unknown entry mode')
runpy.run_path(str(Path(os.environ['FIXED_WORKTREE']) / 'experiments/med_fixed_depth_v1' / (mode + '.py')), run_name='__main__')
