import tempfile
from pathlib import Path
from close_lifecycle import candidates
with tempfile.TemporaryDirectory() as d:
 r=Path(d);a=r/'initializers';a.mkdir();p=a/'e001.pt';p.write_bytes(b'x')
 b=r/'adapters/s20260929/A0';b.mkdir(parents=True);(b/'e001.pt').write_bytes(b'x')
 assert candidates(r)==[p]
 p.unlink();p.symlink_to(b/'e001.pt')
 try:candidates(r)
 except AssertionError:pass
 else:raise AssertionError('symlink accepted')
print('lifecycle scope check PASS')
