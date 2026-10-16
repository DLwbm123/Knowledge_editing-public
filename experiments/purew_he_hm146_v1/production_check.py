"""Explicit CPU-only source check; never installs or launches an experiment."""
import json,subprocess,time
from pathlib import Path


def main():
    here=Path(__file__).resolve().parent;w=here.parents[1]
    out=w/'reports/purew_he_hm146_v1_20261016';state=w/'.local_run/he_hm146_preparation_20261016';state.mkdir(parents=True,exist_ok=True)
    payload={k:json.loads((out/name).read_text()) for k,name in [('protocol','PROTOCOL.json'),('authorization','EXECUTION_AUTHORIZATION.json'),('hardware','HARDWARE_AUTHORIZATION.json')]}
    assert not payload['authorization']['execution']
    payload['files']={p.name:p.read_text() for p in here.glob('*.py')}
    for name,code in payload['files'].items():compile(code,name,'exec')
    receipt=dict(started_epoch=time.time(),kind='CPU_ONLY_NO_INSTALL_NO_MODEL_NO_JUDGE')
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=json.dumps((here/'cpu_check.py').read_text())+'\n'+json.dumps(payload)+'\n',text=True,capture_output=True,timeout=600)
        receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    finally:
        receipt['wall_seconds']=time.time()-receipt['started_epoch']
        (state/('CPU_'+str(int(receipt['started_epoch']))+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout.splitlines()[-1]);report['protocol_version']=payload['protocol']['version']
    (out/'CPU_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
