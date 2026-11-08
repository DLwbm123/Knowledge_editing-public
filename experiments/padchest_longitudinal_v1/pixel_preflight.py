"""Reject destructive direct uint16-to-RGB conversion before any model call."""
import json,os,time
from pathlib import Path
import numpy as np
from PIL import Image

root=Path(os.environ['LONG_ROOT']);cfg=json.loads((root/'CONFIG.private.json').read_text());clock=json.loads((root/'CLOCK.json').read_text())
assert time.time()<clock['hard_end']-1200
pairs=[json.loads(p.read_text()) for p in (root/'pairs').glob('pair_*.json')]
paths=sorted({p for pair in pairs for p in pair['paths']});assert len(paths)==24
stats=[]
for path in paths:
 raw=Path(cfg['dataset'])/'images'/Path(path).name
 with Image.open(raw) as image:
  a=np.asarray(image);assert a.ndim==2 and np.isfinite(a).all()
 low,high=int(a.min()),int(a.max());assert 0<=low<high<=65535
 expected=np.rint((a.astype(np.float64)-low)*(255.0/(high-low))).astype(np.uint8)
 with Image.open(path) as image:
  actual=np.asarray(image);assert image.mode=='RGB' and actual.shape==a.shape+(3,)
  assert all(np.array_equal(actual[:,:,c],expected) for c in range(3)),'Real CXR does not match the fixed label-independent display map'
 stats.append(dict(white_fraction=float(np.mean(actual==255)),exact_fixed_display_map=True))
result=dict(status='PASS',images=len(paths),exact_expected_mapping=sum(x['exact_fixed_display_map'] for x in stats),RGB_images=len(paths),
 average_white_fraction=float(np.mean([x['white_fraction'] for x in stats])),rule='per-image full-range affine min/max rounded uint8 RGB; existing rendered images reused',
 new_image_files=0,model_calls=0,Judge_calls=0,time=time.time())
with (root/'PIXEL_ADMISSION.json').open('x') as f:json.dump(result,f,indent=2)
