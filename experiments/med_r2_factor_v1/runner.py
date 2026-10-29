import os
from pathlib import Path
import subprocess

root = Path(os.environ['AUDIT_ROOT'])
sources = [root / 'experiments/medical_proposition_v133/med133_curvature.py',
           root / 'experiments/med_r2_factor_v1/two_direction.py',
           root / 'experiments/med_r2_factor_v1/check.py']
program = "import signal; signal.alarm(120); __name__='synthetic_check'\n" + '\n'.join(p.read_text() for p in sources)
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                         'pro5000', '/data/bmw/envs/v0/bin/python -'],
                        input=program, text=True, capture_output=True, timeout=130)
out = Path(os.environ['AUDIT_OUTPUT'])
out.write_text(result.stdout if result.returncode == 0 else
               'ENGINEERING_FAILURE\n' + result.stdout + result.stderr)
if result.returncode:
    raise SystemExit(result.returncode)
