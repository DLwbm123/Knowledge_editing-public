"""Pixel and role admission independent of all new model outputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

def check(root,cfg):
 old=Path(cfg['corrected_root']);data=json.loads((old/'T5_INPUTS.private.json').read_text());aux=json.loads((root/'AUX_INPUTS.private.json').read_text())
 roles=json.loads((root/'ROLE_BINDINGS.private.json').read_text());s,a,p=[set(roles[k]) for k in ['source_patients','aux_patients','probe_patients']]
 assert (len(s),len(a),len(p))==(4,4,6) and not(s&a or s&p or a&p)
 queries=[t['native'] for t in data['tasks']]+[r['query'] for r in data['plan'] if r['role'] in ['CONTRAST','SAME_STATE']]+aux['queries']
 paths={x for q in queries for x in json.loads(Path(q['image_path']).read_text())['paths']};assert len(paths)==28
 white=[]
 for path in paths:
  with Image.open(Path(cfg['raw_dataset'])/'images'/Path(path).name) as im:raw=np.asarray(im)
  assert raw.ndim==2 and np.isfinite(raw).all();low,high=int(raw.min()),int(raw.max());assert 0<=low<high<=65535
  expected=np.rint((raw.astype(np.float64)-low)*(255.0/(high-low))).astype(np.uint8)
  with Image.open(path) as im:
   rendered=np.asarray(im);assert im.mode=='RGB' and rendered.shape==expected.shape+(3,)
   assert all(np.array_equal(rendered[:,:,c],expected) for c in range(3))
  white.append(float(np.mean(rendered==255)))
 result=dict(status='PASS',images=28,exact_RGB_display_mapping=28,source_patients=4,auxiliary_patients=4,probe_patients=6,patient_role_overlap=0,mean_white_fraction=float(np.mean(white)),method_outputs_used_for_selection=False)
 with (root/'DATA_ADMISSION.json').open('x') as f:json.dump(result,f,indent=2)
 return result
