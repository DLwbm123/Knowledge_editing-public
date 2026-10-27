"""Freeze the selected source/predicate bindings and stage eight private images."""
import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.private/med150_source_visual_v1_20261028'
TEMP=Path('/private/tmp/j.20261028.9')


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Visual review 30 minute limit')));signal.alarm(1800)
    assert not OUT.exists() and not TEMP.exists();OUT.mkdir(parents=True,mode=0o700);TEMP.mkdir(mode=0o700)
    sc=os.statvfs(TEMP);assert sc.f_bavail*sc.f_frsize>8*1024**3
    selected=json.loads((ROOT/'.private/med150_source_metadata_v1_20261028/CANDIDATES_FROZEN.json').read_text())['selections']
    packet=json.loads((ROOT/'.private/med150_source_review_v1_20261028/SOURCE_PACKET_PRIVATE.json').read_text())['records']
    results=json.loads(Path('/private/tmp/y.20261028.4/SOURCE_BASE_FEASIBILITY_PRIVATE.json').read_text())['groups']
    assert len(selected)==len(results)==8
    rows=[]
    for index,(g,res) in enumerate(zip(selected,results),1):
        assert res['group_index']==index
        images={packet[str(q)]['img_name'] for q in g['qid_pool']};assert len(images)==1
        choice=res.get('selected') or {};target=choice.get('target_qid');protect=choice.get('protect_qids',[])
        assert target is None or (target in g['qid_pool'] and all(q in g['qid_pool'] for q in protect))
        rows.append(dict(group_index=index,image_name=next(iter(images)),target_qid=target,protect_qids=protect,status=res['status']))
    assert sum(x['target_qid'] is not None for x in rows)==6
    (OUT/'INPUTS_FROZEN.private.json').write_text(json.dumps(dict(time=time.time(),groups=rows,model_outputs_used_for_new_selection=False),indent=2)+'\n')
    names=[r['image_name'] for r in rows]
    code='names='+repr(names)+'\n'+'''from pathlib import Path
import base64,json
root=Path('/data/bmw/DataP/knowledge_editing/data/m3bench/SLAKE/imgs').resolve();out=[];total=0
for n in names:
 p=(root/n).resolve();assert p.is_relative_to(root) and p.is_file()
 size=p.stat().st_size;total+=size;assert total<20000000
 out.append(dict(size=size,data=base64.b64encode(p.read_bytes()).decode()))
print(json.dumps(out))
'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','/data/bmw/envs/v0/bin/python','-'],input=code,text=True,capture_output=True,timeout=120)
    assert p.returncode==0,p.stderr[-1500:];images=json.loads(p.stdout);assert len(images)==8
    from PIL import Image
    total=0
    for i,img in enumerate(images,1):
        b=base64.b64decode(img['data'],validate=True);assert len(b)==img['size'];total+=len(b);assert total<20000000
        dest=TEMP/f'{i:02d}.png';fd=os.open(dest,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as f:f.write(b)
        with Image.open(dest) as im:im.verify()
    receipt=dict(status='FROZEN_IMAGES_STAGED_FOR_REVIEW',time=time.time(),groups=8,potential_groups=6,total_bytes=total,source_sizes_match=True,decoded=True,byte_identity_verified=False,external_downloads=0,model_calls=0,clinical_validation=False)
    (OUT/'STAGING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
