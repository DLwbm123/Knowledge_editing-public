"""Adopt a single timed-out installation by receipts; never install or replay a GPU stage."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import deploy


def snapshot():
    code="""from pathlib import Path
import json,time
r=Path('/data/bmw/Knowledge_editing/outputs/purew-fs3-v1-20261014/run')
read=lambda p:json.loads(p.read_text())
d={'manifest':read(r/'RUN_MANIFEST.json'),'version':read(r/'private/GPU_SOURCE_VERSION.json'),
   'failures':[p.name for p in (r/'private/failures').glob('*.json')]}
s=r/'private/START_CONTROLLER.json'
if s.exists():
 x=read(s);p=Path('/proc')/str(x['pid'])
 d['start']=x;d['identity_matches']=p.exists() and p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==x['start_ticks'] and p.joinpath('cmdline').read_bytes().split(b'\\0')[-2].decode()==x['entry']
print(json.dumps(d))
"""
    return json.loads(deploy.remote(code))


def main():
    state=deploy.STATE;attempt=json.loads((state/'INSTALL_ATTEMPT.json').read_text())
    assert not (state/'SCORERS.json').exists(),'Existing consumer identities require review'
    with (state/'ADOPTION_STARTED.json').open('x') as f:json.dump(dict(epoch=time.time(),install_commit=attempt['commit']),f)
    while True:
        d=snapshot();(state/'ADOPTION_STATUS.json').write_text(json.dumps(d,indent=2)+'\n')
        assert d['version']['commit']==attempt['commit'] and not d['failures']
        assert time.time()<d['manifest']['deadline_epoch'],'Original wall deadline retained'
        if 'start' in d:
            assert d['identity_matches'],'Retained controller identity must match exactly'
            startup=dict(status='EXISTING_INSTALL_ADOPTED',start=d['start'],activation_epoch=d['manifest']['starting_epoch'],run_root=json.loads((deploy.OUT/'PROTOCOL.json').read_text())['run_root'])
            os.environ['WORKSPACE']=str(deploy.W)
            import consumers
            consumers.start_consumers(startup)
            (state/'INSTALL_ADOPTION_COMPLETE.json').write_text(json.dumps(dict(epoch=time.time(),startup=startup,install_repeated=False),indent=2)+'\n')
            return
        time.sleep(30)


if __name__=='__main__':
    try:main()
    except BaseException:
        (deploy.STATE/'ADOPTION_FAILURE.json').write_text(json.dumps(dict(epoch=time.time(),traceback=traceback.format_exc())))
        raise
