"""CPU-only production chain test; receipts preserve each attempt including failures."""
import json,subprocess,time
from pathlib import Path

def main():
    here=Path(__file__).resolve().parent;w=here.parents[1];out=w/'reports/purew_projected_objective_first1_v1_20261015'
    state=w/'.local_run/projected_objective_first1_v1_20261015';state.mkdir(parents=True,exist_ok=True)
    p=json.loads((out/'PROTOCOL.json').read_text());budget=dict(p['budgets'])
    budget.update(maximum_formal_attempt_slots=40,maximum_committed_updates=42,maximum_forward_calls=30000,maximum_backwards=600,maximum_generations=184)
    effective=dict(p,budgets=budget,active_arms=['PM','PU'],reference_run='/data/bmw/Knowledge_editing/outputs/purew-objective-scope-v1-20261015/run',K_reference_run='/data/bmw/Knowledge_editing/outputs/purew-protect100-v1-20261013/run')
    files={p.name:p.read_text() for p in here.glob('*.py')}
    files['cpu_check_legacy.py']=(w/'experiments/purew_ecw_consistent3_v1/cpu_check.py').read_text()
    code=(here/'cpu_check.py').read_text()+ '\n'+(here/'integration_check.py').read_text()
    start=time.time();receipt=dict(epoch=start,action='CPU_PRODUCTION_NO_INSTALL_NO_MODEL_NO_JUDGE')
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n'+json.dumps(dict(files=files,protocol=effective))+'\n',text=True,capture_output=True,timeout=180)
        receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        receipt.update(status='CPU_TIMEOUT',error=str(exc));raise
    finally:
        receipt['wall_seconds']=time.time()-start;(state/('CPU_'+str(int(start))+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    assert result.returncode==0
    report=json.loads(result.stdout.splitlines()[-1]);report.pop('disposable_root',None)
    (out/'PRODUCTION_CPU_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
