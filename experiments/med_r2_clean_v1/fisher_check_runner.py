import os
from pathlib import Path
import subprocess

root = Path(os.environ['AUDIT_ROOT'])
sources = [root/'experiments/med_r2_clean_v1/fisher_slice.py',
           root/'experiments/med_r2_clean_v1/test_fisher_slice.py']
program = 'import signal; signal.alarm(120)\n' + '\n'.join(p.read_text() for p in sources)
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                         'pro5000', '/data/bmw/envs/v0/bin/python -'],
                        input=program, text=True, capture_output=True, timeout=130)
Path(os.environ['AUDIT_OUTPUT']).write_text(result.stdout if result.returncode == 0 else
                                           'ENGINEERING_FAILURE\n' + result.stdout + result.stderr)
if result.returncode:
    raise SystemExit(result.returncode)
