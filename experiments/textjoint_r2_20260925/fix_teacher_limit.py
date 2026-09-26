"""Align shared KL prefix validation with the frozen runtime generation limit."""
import ast
import os
from pathlib import Path

ROOT=Path(os.environ['RUN_ROOT'])
path=ROOT/'source/scripts/medtrace/run_selective_write.py'
old='    if not tokens or len(tokens) > 128:\n'
new='    if not tokens or len(tokens) > int(runtime.generation_config.get("max_new_tokens", 128)):\n'
text=path.read_text()
assert text.count(old)==1 or text.count(new)==1
if old in text:
    backup=ROOT/'private/teacher_limit_original.py'
    assert not backup.exists()
    backup.write_text(text)
    path.write_text(text.replace(old,new,1))

# Test the actual patched function and causal mask on CPU, without loading weights.
import torch
from types import SimpleNamespace
mask_path=ROOT/'source/methods/medtrace/selective_write.py'
ns={'torch':torch}
for source,name in [(mask_path,'predictor_mask'),(path,'teacher_batch')]:
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
raw=dict(input_ids=torch.tensor([[7,8,9]]),attention_mask=torch.ones(1,3,dtype=torch.long),images=None,image_sha256='synthetic')
def expand(**kw):
    n=kw['raw_input_ids'].shape[1]
    return torch.zeros(1,n,2),kw['attention_mask'],None,kw['labels']
runtime=SimpleNamespace(device='cpu',generation_config={'max_new_tokens':1024},
    adapter=SimpleNamespace(prepare_inputs=lambda *args:raw),_expand_multimodal=expand)
row=dict(image_path='synthetic',question='synthetic',eqkey='synthetic')
for length in [1,128,158,1024]:
    tokens=[11]*(length-1)+[2]
    _,labels,mask,binding=ns['teacher_batch'](runtime,row,tokens)
    assert int(mask.sum())==length and binding['tokens']==tokens
    assert mask[0,2] and not mask[0,-1] and labels[0,-1]==2
for tokens in [[],[11]*1025]:
    try:ns['teacher_batch'](runtime,row,tokens)
    except ValueError:pass
    else:raise AssertionError('Invalid prefix accepted')
print('PASS: full 158/1024-token prefixes, EOS, causal shift, empty/oversized rejection')
