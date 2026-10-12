"""Reuse isolated scorer; only read-only SSH observations get bounded transport retries."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(os.environ['SCORER_BASE']).parent))
import scorer as base

remote=base.remote


def observe(op):
    for attempt in range(3):
        try:
            return remote(op)
        except subprocess.CalledProcessError as exc:
            if exc.returncode!=255 or op['action'] not in ('next','status','reserved') or attempt==2:
                raise
            time.sleep(2**(attempt+1))


base.remote=observe
try:
    base.main()
except Exception:
    data=dict(epoch=time.time(),traceback=traceback.format_exc(),new_requests_stopped=True)
    (base.LOCAL/'SCORER_FAILURE.json').write_text(__import__('json').dumps(data))
    try:
        base.publish_file('SCORER_FAILURE.json',__import__('json').dumps(data).encode())
    except Exception:
        pass
    raise
