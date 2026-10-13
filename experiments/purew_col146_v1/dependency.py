"""Read-only predecessor barrier; neither P qualification nor GPU utilization is a gate."""
import json
from pathlib import Path


def ready(root):
    root=Path(root)
    for name in ('TRAINING_GENERATION_COMPLETE','FINAL_CONSUMED'):
        if not (root/'private'/(name+'.json')).exists():return False
    for action in ('gpu','reload'):
        p=root/'private'/('EXIT_'+action+'.json')
        if not p.exists() or json.loads(p.read_text())['exit_code']!=0:return False
        start=json.loads((root/'private'/('START_'+action+'.json')).read_text())
        stat=Path('/proc')/str(start['pid'])/'stat'
        if stat.exists() and stat.read_text().split()[21]==str(start['start_ticks']):return False
    ledger=json.loads((root/'RESOURCE_LEDGER.json').read_text())
    return not any(not x.get('ended_epoch') for x in ledger['gpu_sessions'])


def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        r=Path(folder);(r/'private').mkdir()
        assert not ready(r)
        for n in ('TRAINING_GENERATION_COMPLETE','FINAL_CONSUMED'):(r/'private'/(n+'.json')).write_text('{}')
        for a in ('gpu','reload'):
            (r/'private'/('EXIT_'+a+'.json')).write_text('{"exit_code":0}')
            (r/'private'/('START_'+a+'.json')).write_text('{"pid":999999999,"start_ticks":"0"}')
        (r/'RESOURCE_LEDGER.json').write_text('{"gpu_sessions":[{"ended_epoch":1}]}')
        assert ready(r)
        (r/'RESOURCE_LEDGER.json').write_text('{"gpu_sessions":[{}]}');assert not ready(r)
        (r/'RESOURCE_LEDGER.json').write_text('{"gpu_sessions":[]}')
        (r/'private/EXIT_reload.json').write_text('{"exit_code":1}');assert not ready(r)
    return dict(status='PASS',requires_both_zero_exits_and_retired_leases=True,P_qualification_not_a_dependency=True)
