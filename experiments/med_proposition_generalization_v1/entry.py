import os
from pathlib import Path
import runpy
mode = os.environ['ENTRY_MODE']
if mode not in ('guard', 'worker'):
    raise ValueError('Unknown mode')
runpy.run_path(str(Path(os.environ['EXPERIMENT_ROOT']) / 'tools' / (mode + '.py')), run_name='__main__')
