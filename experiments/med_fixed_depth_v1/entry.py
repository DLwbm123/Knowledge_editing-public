"""Neutral process entry; state and dispatch are passed through the environment."""
import os
from pathlib import Path
import runpy

mode = os.environ.get('ENTRY_MODE', 'guard')
if mode not in ('guard', 'worker'):
    raise ValueError('Unknown entry mode')
runpy.run_path(str(Path(os.environ['EXPERIMENT_ROOT']) / 'tools' / (mode + '.py')), run_name='__main__')
