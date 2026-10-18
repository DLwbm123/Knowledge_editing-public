"""Fixed read-only operator probe deployment; never updates weights."""
import inspect,json,subprocess
from pathlib import Path
from experiments.vigor_medical_single_image_v1.diagnose_deploy import remote_diagnostic
s=inspect.getsource(remote_diagnostic).replace('vigor-minimal-diagnostic4-v1-20261018/repaired_candidates','vigor-minimal-diagnostic4-v1-20261018/operator_probe')
s=s.replace("implementation_commit=payload['commit'],","implementation_commit=payload['commit'], existing_sum_factor='/data/bmw/Knowledge_editing/outputs/vigor-minimal-diagnostic4-v1-20261018/run/private/sum_target_factor.pt',")
s=s.replace('vigor_medical_single_image_v1.diagnose import main','vigor_medical_single_image_v1.investigate import main')
root=Path.cwd();files=list((root/'experiments/vigor_medical_single_image_v1').glob('*.py'));files.append(root/'experiments/medtrace_stage12_v2_20260911/m3bench_repro/editors/llava_runtime.py')
payload=dict(files={str(p.relative_to(root)):p.read_text() for p in files},commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
code=s+'\nprint(json.dumps(remote_diagnostic(json.loads(sys.stdin.readline()))))'
r=subprocess.run(['ssh','pro5000',"python3 -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n'+json.dumps(payload)+'\n',capture_output=True,text=True)
print(r.stdout,r.stderr)
