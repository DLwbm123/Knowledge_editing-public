"""Install under a neutral absolute path; private adjacent ENV supplies bindings."""
import json
import os
from pathlib import Path
import runpy

here = Path(__file__).parent
os.environ.update(json.loads((here / 'ENV.json').read_text()))
root = Path(os.environ['V107_ROOT'])
runpy.run_path(str(root / ('guard.py' if os.environ.get('ACTION_107') == 'guard' else 'train.py')),
              run_name='__main__')
