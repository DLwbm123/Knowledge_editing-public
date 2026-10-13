"""Private bounded acquisition receiver; neutral entry and RUN_ROOT environment."""
import base64
import fcntl
import json
import os
import pathlib
import sys
import time

r=pathlib.Path(os.environ['RUN_ROOT'])
h=json.loads(sys.stdin.buffer.readline());n=h['name'];kind=h.get('kind','metadata')
assert pathlib.Path(n).name==n and kind in ('metadata','image')
out=r/'private/primary_evidence'/n
lock=(r/'private/.acquisition.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
p=r/'private/PRIMARY_DOWNLOAD_LEDGER.json';ledger=json.loads(p.read_text())
manifest=json.loads((r/'RUN_MANIFEST.json').read_text())
if h.get('mode')=='read':
    print(json.dumps(dict(exists=out.exists(),data=base64.b64encode(out.read_bytes()).decode() if out.exists() else None,
                          remaining=manifest['caps']['image_bytes' if kind=='image' else 'metadata_bytes']-ledger[kind+'_download_bytes'])))
else:
    data=sys.stdin.buffer.read();rec=h['receipt'];size=int(rec.get('source_download_bytes',0));assert size>=0
    ledger[kind+'_download_bytes']+=size
    item={**rec,'name':n,'kind':kind,'stored_bytes':len(data),'receiver_epoch':time.time()}
    cap=manifest['caps']['metadata_bytes' if kind=='metadata' else 'image_bytes']
    admitted=ledger[kind+'_download_bytes']<=cap and time.time()<manifest['deadline_epoch'] and not (r/'STOP').exists()
    item['storage_admitted']=admitted;ledger['files'].append(item)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(p)
    assert admitted,'First-clock or download cap reached; attempt cost preserved'
    out.parent.mkdir(exist_ok=True)
    with out.open('xb') as f:f.write(data)
    print(json.dumps(dict(name=n,kind=kind,stored_bytes=len(data),download_bytes=size)))
