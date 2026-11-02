import os,runpy
from pathlib import Path
mode=os.environ['ENTRY_MODE']
if mode not in ('guard','worker'):raise ValueError('unknown stage')
runpy.run_path(str(Path(os.environ['DIAG_ROOT'])/'tools'/(mode+'.py')),run_name='__main__')
