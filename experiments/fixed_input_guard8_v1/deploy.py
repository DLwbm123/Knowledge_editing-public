"""Publish source first, then one neutral remote install; unknown outcomes are retained."""
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    workspace = Path(__file__).resolve().parents[2]
    report = workspace/'reports/fixed_input_guard8_v1_20261016'
    state = workspace/'.local_run/fixed_input_guard8_v1_20261016'
    state.mkdir(parents=True, exist_ok=True)
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=workspace, text=True).strip()
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=workspace, text=True).strip()
    paths = list((workspace/'experiments/fixed_input_guard8_v1').glob('*.py'))
    paths += [workspace/'experiments'/p for p in ('fixed_input_guard_v1/guard.py', 'purew_original146_v1/common.py',
                                                'purew_original146_v1/contract.py', 'purew_medical_evidence100_v1/access.py')]
    payload = dict(commit=sha, gpu=7, uuid='GPU-578b486c-6eae-13b0-23be-353ed09220ec',
                   authorization=json.loads((report/'launch/AUTHORIZATION.json').read_text()),
                   protocol=json.loads((report/'PROTOCOL.json').read_text()), cpu=json.loads((report/'launch/CPU_CHECK.json').read_text()),
                   files={str(p.relative_to(workspace)):p.read_text() for p in paths})
    with (state/'INSTALL_ATTEMPT.json').open('x') as handle:
        json.dump(dict(commit=sha, epoch=time.time(), gpu=7), handle)
    code = (Path(__file__).parent/'bootstrap.py').read_text()
    try:
        result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',
                                 "python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],
                                input=json.dumps(code)+'\n'+json.dumps(payload)+'\n', text=True,
                                capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        (state/'INSTALL_UNCERTAIN.json').write_text(json.dumps(dict(reason='Inspect remote state before any further installation')))
        raise
    (state/'INSTALL_RESPONSE.json').write_text(json.dumps(dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2))
    assert result.returncode == 0, result.stderr
    startup = json.loads(result.stdout)
    (state/'STARTUP.json').write_text(json.dumps(startup, indent=2))
    print(json.dumps(startup, indent=2))


if __name__ == '__main__':
    main()
