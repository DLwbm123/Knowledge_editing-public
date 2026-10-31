"""Neutral entry; method and paths are supplied only via environment."""
import os
from pathlib import Path
import runpy

mode = os.environ['ENTRY_MODE']
if mode not in ('guard', 'train', 'eval', 'packet'):
    raise ValueError('Unknown execution stage')
filename = {'guard': 'guard.py', 'train': 'train_worker.py', 'eval': 'eval_worker.py', 'packet': 'packet_worker.py'}[mode]
runpy.run_path(str(Path(os.environ['EXPERIMENT_ROOT']) / 'tools' / filename), run_name='__main__')
