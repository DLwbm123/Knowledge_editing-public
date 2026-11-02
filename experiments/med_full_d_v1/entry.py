import os,runpy
from pathlib import Path
mode=os.environ['ENTRY_MODE']
if mode not in ('guard','worker'):raise ValueError('Invalid mode')
runpy.run_path(str(Path(os.environ['EXPERIMENT_ROOT'])/'tools'/f'{mode}.py'),run_name='__main__')
