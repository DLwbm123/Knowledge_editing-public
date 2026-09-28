"""Run against a prepared private extension via RUN_ROOT; emits no private inputs."""
import json,os,sys
from pathlib import Path
root=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(root),str(root/'source_patch'),str(root/'source')]
import torch
from bindings import check_payload
imports=json.loads((root/'IMPORTED_BINDINGS.json').read_text())
for rel,meta in imports.items():
 payload=torch.load(root/rel,map_location='cpu',weights_only=True)
 check_payload(payload,rel,meta['parent_expected'])
 try:check_payload(payload,rel,dict(meta['parent_expected'],actual_step=79))
 except AssertionError:pass
 else:raise AssertionError('Mismatched teacher step accepted')
print('PASS',len(imports),'verified teachers; mismatched steps rejected')
