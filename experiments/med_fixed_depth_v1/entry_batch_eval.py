"""Neutral process entry for the fixed four-GPU evaluation stage."""
import os
from pathlib import Path
import runpy

mode = os.environ.get('ENTRY_MODE', 'batch_eval_guard')
if mode not in ('batch_eval_guard', 'batch_eval_worker'):
    raise ValueError('Unknown entry mode')
runpy.run_path(str(Path(os.environ['EXPERIMENT_ROOT']) / 'tools' / (mode + '.py')), run_name='__main__')
