"""Prepare a coverage-template union, not permission to repeat completed work.

An explicit reuse audit and missing-work gate must precede execution. Inherited
assets are read-only. The four-method template includes diagnostic FIT slots
which completed external methods are not required to regenerate.
"""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ['V131_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not ROOT.exists(), 'No implicit overwrite or restart'
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    mount = subprocess.check_output(['findmnt', '-T', str(ROOT.parent), '-J'], text=True)
    fs = json.loads(mount)['filesystems'][0]
    assert fs['source'].startswith('/dev/') and fs['target'] == '/data'
    assert os.statvfs(ROOT.parent).f_bavail * os.statvfs(ROOT.parent).f_frsize > 108_000_000_000
    ROOT.mkdir()
    probe = ROOT / 'write_read_probe.txt'
    probe.write_text('v131 storage probe\n')
    assert probe.read_text() == 'v131 storage probe\n'
    probe.unlink()
    sys.path.insert(0, str(PARENT / 'private/tools'))
    import contract
    packet = read(PARENT / 'private/M13_SOURCE_PACKET.json')
    assert contract.source_admission(packet)['status'] == 'PASS'
    previous_plan = read(PARENT / 'private/EVAL_PLAN146.json')
    queries = {r['query']['query_id']: r['query'] for r in previous_plan}
    plan = contract.evaluation_plan(packet, queries)
    prototype = [r for r in plan if r['method'] == 'M13']
    assert len(prototype) == 7494
    methods = ['M13P_160', 'BALANCEDIT', 'MEMIT_MM', 'ALPHAEDIT_MM']
    complete = [dict(r, method=method) for method in methods for r in prototype]
    complete += [r for r in plan if r['method'] == 'BASE']
    assert len(complete) == len({contract.consumer_id(r) for r in complete}) == 32069
    assert all(Path(p).is_file() for p in packet['native_fit_image_paths'].values())
    for method in methods:
        (ROOT / method).mkdir()
    write(ROOT / 'EVAL_PLAN.json', complete)
    write(ROOT / 'ASSET_REUSE.json', dict(
        source_packet=str(PARENT / 'private/M13_SOURCE_PACKET.json'),
        fisher=str(PARENT / 'private/OPERATOR_FISHER.pt'),
        calibration_rows=146, C_targets=146, source_images=125,
        router_bindings=str(PARENT / 'private/ROUTER_BINDINGS146.json'),
        old_single_outputs='Identity audit only; no aggregate scores substituted',
        old_M13P_weights='Retired; retrain fixed recipe',
        old_BalancEdit_outputs='FP16 runtime incompatible; historical only',
        source_admission='PASS', storage_write_read=True, mount=fs))
    write(ROOT / 'PREPARE_COMPLETE.json', dict(time=time.time(), methods=methods,
        planning_only=True, launch_authorized=False,
        consumers=len(complete), per_method=7494, Base=2093,
        counts=dict(collections.Counter(r['method'] for r in complete))))


if __name__ == '__main__':
    main()
