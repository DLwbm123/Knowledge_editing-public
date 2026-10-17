"""Run CPU-only preparation in a disposable directory; never installs a run."""
import json
from pathlib import Path
import subprocess
import time


def main():
    here = Path(__file__).resolve().parent
    workspace = here.parents[1]
    previous = workspace / 'experiments/purew_ecw_consistent3_v1'
    state = workspace / '.local_run/ecw_r1_preparation_20261015'
    state.mkdir(parents=True, exist_ok=True)
    files = {name: (previous / name).read_text() for name in ['worker.py', 'source_loss.py', 'medical.py', 'runtime.py']}
    files.update({'core.py': (here / 'core.py').read_text(),
                  'legacy_core.py': (previous / 'core.py').read_text(),
                  'cpu_check_legacy.py': (previous / 'cpu_check.py').read_text()})
    protocol = json.loads((workspace / 'reports/purew_ecw_consistent3_v1_20261015/PROTOCOL.json').read_text())
    started = time.time()
    receipt = dict(epoch=started, action='CPU_ONLY_NO_INSTALL_NO_GPU_NO_JUDGE')
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'pro5000',
            "/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
            input=json.dumps((here / 'cpu_check.py').read_text()) + '\n' + json.dumps(dict(files=files, protocol=protocol)) + '\n',
            capture_output=True, text=True, timeout=120)
        receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        receipt.update(status='CPU_TIMEOUT_DO_NOT_BLINDLY_RETRY', error=str(exc))
        raise
    finally:
        receipt['elapsed_wall_seconds'] = time.time() - started
        (state / ('CPU_' + str(int(started)) + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    if result.returncode:
        raise SystemExit(result.returncode)
    validation = json.loads(result.stdout)
    validation.pop('disposable_root', None)
    out = workspace / 'reports/purew_ecw_r1_v1_20261015'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'CPU_VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n')


if __name__ == '__main__':
    main()
