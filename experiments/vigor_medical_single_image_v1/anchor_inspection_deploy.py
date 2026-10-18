"""One-off read-only inspection of the historical gradient anchor; no weight update."""
import inspect,json,subprocess
from pathlib import Path
from experiments.vigor_medical_single_image_v1.diagnose_deploy import remote_diagnostic
s=inspect.getsource(remote_diagnostic).replace('vigor-minimal-diagnostic4-v1-20261018/run','vigor-minimal-diagnostic4-v1-20261018/anchor_inspection')
entry='''import json,os,torch
from experiments.vigor_medical_single_image_v1.diagnose import main
import experiments.vigor_medical_single_image_v1.diagnose as d
from experiments.vigor_medical_single_image_v1.editor import paired_batches,editable_fp32,gradient_pair,direction
def inspect_anchor(runtime,record,index,factor,requested,historical=None):
 layer=runtime.get_module('model.layers.0.mlp.down_proj')
 with editable_fp32(runtime.model,layer,readonly=True):
  pair=paired_batches(runtime,record)
  losses,grads=gradient_pair(runtime,layer,pair)
  _,geo=direction(*grads,*(factor[k].to(layer.weight.device) for k in ('a','b')))
  print(json.dumps(dict(current=losses,E=geo['E'],historical=historical)),flush=True)
 raise RuntimeError('READ_ONLY_ANCHOR_INSPECTION_FINISHED_NO_UPDATE')
d.candidate=inspect_anchor
d.sum_target_factor=lambda *a:torch.load('/data/bmw/Knowledge_editing/outputs/vigor-minimal-diagnostic4-v1-20261018/run/private/sum_target_factor.pt',weights_only=True)['factor']
main(json.load(open(os.environ['CONFIG_FILE'])))
'''
a=s.index('    entry.write_text(');b=s.index('\n    with (root/',a)
s=s[:a]+'    entry.write_text('+repr(entry)+')'+s[b:]
root=Path.cwd();files=list((root/'experiments/vigor_medical_single_image_v1').glob('*.py'));files.append(root/'experiments/medtrace_stage12_v2_20260911/m3bench_repro/editors/llava_runtime.py')
payload=dict(files={str(p.relative_to(root)):p.read_text() for p in files},commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
code=s+'\nprint(json.dumps(remote_diagnostic(json.loads(sys.stdin.readline()))))'
r=subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',capture_output=True,text=True)
print(r.stdout,r.stderr)
