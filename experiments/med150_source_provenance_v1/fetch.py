"""Fetch only eight preregistered detection annotations, without QA or pixels."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.private/med150_source_provenance_v1_20261028'

def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30 minute source audit limit')));signal.alarm(1800)
    assert not OUT.exists();OUT.mkdir(parents=True,mode=0o700)
    selection=json.loads((ROOT/'.private/med150_source_metadata_v1_20261028/CANDIDATES_FROZEN.json').read_text())['selections']
    assert len(selection)==8;names=[x['source_group'] for x in selection]
    code='names='+repr(names)+'\n'+'''from pathlib import Path
import json
root=Path('/data/bmw/DataP/knowledge_editing/data/m3bench/SLAKE/imgs').resolve();result=[];total=0
for n in names:
 d=(root/n).resolve();assert d.is_relative_to(root)
 p=d/'detection.json';mask=d/'mask.png';assert p.is_file() and mask.is_file()
 size=p.stat().st_size;assert size<=2000000;total+=size;assert total<20000000
 result.append(dict(size=size,mask_exists=mask.is_file(),detection=json.loads(p.read_text())))
print(json.dumps(result,ensure_ascii=False))
'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','/data/bmw/envs/v0/bin/python','-'],input=code,text=True,capture_output=True,timeout=120)
    if p.returncode:raise RuntimeError('Source metadata extraction failure: '+p.stderr[-1500:])
    assert len(p.stdout.encode())<20_000_000
    data=json.loads(p.stdout);assert len(data)==8 and all(x['mask_exists'] for x in data)
    target=OUT/'ANNOTATIONS_PRIVATE.json';fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    receipt=dict(status='DETECTION_ANNOTATIONS_FETCHED_UNREVIEWED',time=time.time(),groups=8,detection_files=8,mask_files_present=8,
        mask_pixels_read=0,qa_files_opened=0,base_calls=0,method_calls=0,source_file_bytes=sum(x['size'] for x in data),clinical_confirmation=False)
    (OUT/'FETCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
