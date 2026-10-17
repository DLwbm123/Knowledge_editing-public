"""CPU-only production chain test; receipts preserve each attempt including failures."""
import json,subprocess,time
from pathlib import Path

def main():
    here=Path(__file__).resolve().parent;w=here.parents[1];out=w/'reports/purew_hinge_evidence_first1_v1_20261016'
    state=w/'.local_run/hinge_evidence_first1_v1_20261016';state.mkdir(parents=True,exist_ok=True)
    from configuration import effective_protocol
    p=json.loads((out/'PROTOCOL.json').read_text());effective=effective_protocol(p)
    files={p.name:p.read_text() for p in here.glob('*.py')}
    code=(here/'cpu_check.py').read_text()+ '\n'+"\nimport subprocess\nformal_entry=root/'e.py';formal_entry.write_text(payload['files']['formal_check.py'])\nformal=subprocess.run([sys.executable,str(formal_entry)],env=dict(os.environ,PYTHONPATH=str(tools)+':'+str(parent/'source')),text=True,capture_output=True)\nassert formal.returncode==0,formal.stderr\nassert json.loads(formal.stdout)['status']=='PASS'\n"+(here/'integration_check.py').read_text()
    start=time.time();receipt=dict(epoch=start,action='CPU_PRODUCTION_NO_INSTALL_NO_MODEL_NO_JUDGE')
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps(code)+'\n'+json.dumps(dict(files=files,protocol=effective))+'\n',text=True,capture_output=True,timeout=240)
        receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        receipt.update(status='CPU_TIMEOUT',error=str(exc));raise
    finally:
        receipt['wall_seconds']=time.time()-start;(state/('CPU_'+str(int(start))+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    if result.returncode!=0:
        (out/'PRODUCTION_CPU_VALIDATION.json').write_text(json.dumps(dict(status='FAIL',protocol_version=p['version'],attempt_epoch=start,production_CPU_only=True),indent=2)+'\n')
    assert result.returncode==0
    report=json.loads(result.stdout.splitlines()[-1]);report.pop('disposable_root',None)
    (out/'PRODUCTION_CPU_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
