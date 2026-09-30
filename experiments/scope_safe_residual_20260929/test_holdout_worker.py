"""Validate role and duplicate guards without the GPU runtime."""
import ast
from pathlib import Path
module=ast.parse(Path(__file__).with_name('holdout_worker.py').read_text())
fn=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='evaluation_task')
import hashlib
ns={'hashlib':hashlib};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<selfcheck>','exec'),ns)
f=ns['evaluation_task'];row=dict(purpose='LOCALITY_STRESS_HOLDOUT',scope='negative',image_sha256='a',question='q')
assert f([row])['order']==0 and len(f([row])['evaluation'])==1
for bad in [[row,row],[dict(row,purpose='CHECK')],[dict(row,scope='positive')]]:
 try:f(bad)
 except AssertionError:pass
 else:raise AssertionError('bad holdout accepted')
print('holdout role/duplicate self-check PASS')
