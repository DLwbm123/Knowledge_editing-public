"""Neutral entry for the finite report dependency."""
import os
from pathlib import Path
import runpy

runpy.run_path(str(Path(os.environ['FIXED_WORKTREE']) / 'experiments/med_fixed_depth_v1/report_tail.py'), run_name='__main__')
