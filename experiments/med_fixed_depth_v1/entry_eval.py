"""Neutral entry for the first-pair generation dependency."""
import os
from pathlib import Path
import runpy

mode = os.environ.get('ENTRY_MODE', 'first_eval_guard')
if mode not in ('first_eval_guard', 'first_eval'):
    raise ValueError('Unknown entry mode')
runpy.run_path(str(Path(os.environ['EXPERIMENT_ROOT']) / 'tools' / (mode + '.py')), run_name='__main__')
