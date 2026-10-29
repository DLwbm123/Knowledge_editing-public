"""Independent wall-time, storage and free-space guard for the read-only audit."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

out=Path(os.environ['AUDIT_OUTPUT']); started=time.monotonic()
assert not out.exists() and shutil.disk_usage(out.parent).free >= 8*1024**3
p=subprocess.Popen([sys.executable,'-c','import os,runpy; runpy.run_path(os.environ["AUDIT_WORKER"],run_name="__main__")'],env=os.environ.copy())
while p.poll() is None:
    if time.monotonic()-started >= 1800 or shutil.disk_usage(out.parent).free < 8*1024**3 or (out.exists() and out.stat().st_size >= 20_000_000):
        p.kill(); p.wait(); raise RuntimeError('resource guard stopped audit')
    time.sleep(.1)
if p.returncode:
    raise SystemExit(p.returncode)
assert out.exists() and out.stat().st_size < 20_000_000
print('AUDIT_GUARD_EXIT=0')
