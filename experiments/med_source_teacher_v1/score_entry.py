"""Neutral local entry for the independent scoring guard and worker."""
import os
from pathlib import Path
import runpy

mode = os.environ['ENTRY_MODE']
if mode not in ('score_guard', 'score_executor'):
    raise ValueError('Unknown local score stage')
runpy.run_path(str(Path(os.environ['TEACHER_SCORE_ROOT']) / 'tools' / (mode + '.py')), run_name='__main__')
